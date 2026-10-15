"""新闻文本十分类：清洗、TF-IDF、四种模型的参数比较和可选测试集预测。"""

import argparse
import os
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


DATA_DIR = Path(__file__).resolve().parent
RESULT_DIR = DATA_DIR / "results"
RANDOM_STATE = 42
VALIDATION_RATIO = 0.2
MODEL_NAMES = ("naive_bayes", "svm", "logistic_regression", "mlp")

CONTACT_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.IGNORECASE
)
ATTRIBUTION_PATTERN = re.compile(r"^(?:In article\b.*|On .+ wrote:)\s*$", re.IGNORECASE)
PUNCTUATION_PATTERN = re.compile(r"[^A-Za-z0-9\s]")
STOP_WORDS = ENGLISH_STOP_WORDS - {"no", "not", "nor", "never"}


def load_data():
    """读取有标签训练集和无标签测试集。"""
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")
    if not {"text", "target"}.issubset(train_df.columns) or "text" not in test_df:
        raise ValueError("CSV 列名应包含训练集 text、target 和测试集 text。")
    return (
        train_df["text"].tolist(),
        train_df["target"].to_numpy(),
        test_df["text"].tolist(),
    )


def preprocess_text(raw_text):
    """保留 Subject 和正文，去掉邮件元信息、引用、签名及常见噪声。"""
    text = "" if pd.isna(raw_text) else str(raw_text)
    text = text.replace("\r\n", "\n")
    header, separator, body = text.partition("\n\n")
    header_lines = header.splitlines()
    is_mail = bool(separator) and any(
        line.lower().startswith(("from:", "subject:")) for line in header_lines
    )
    if is_mail:
        subject = next(
            (
                line.split(":", 1)[1]
                for line in header_lines
                if line.lower().startswith("subject:")
            ),
            "",
        )
    else:
        subject, body = "", text

    body_lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped == "--":
            break
        if stripped.startswith(">") or ATTRIBUTION_PATTERN.match(stripped):
            continue
        body_lines.append(line)

    cleaned = subject + " " + " ".join(body_lines)
    cleaned = CONTACT_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.lower().replace("c++", "cplusplus").replace("c#", "csharp")
    cleaned = PUNCTUATION_PATTERN.sub(" ", cleaned)
    return " ".join(cleaned.split())


def make_vectorizer():
    """所有候选模型共用同一份、仅在训练子集拟合的 TF-IDF。"""
    return TfidfVectorizer(
        stop_words=list(STOP_WORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=20000,
        sublinear_tf=True,
    )


def build_model_candidates(model_name="all"):
    """根据 TUNING.md 及文本数据的稀疏特征建立候选配置。"""
    candidates = []
    if model_name in ("all", "naive_bayes"):
        for alpha in (0.1, 1.0, 10.0):
            candidates.append(
                ("naive_bayes", f"alpha={alpha}", MultinomialNB(alpha=alpha))
            )

    if model_name in ("all", "svm"):
        for c in (0.1, 1.0, 10.0):
            candidates.append(
                ("svm", f"kernel=linear, C={c}", SVC(kernel="linear", C=c))
            )
        for c, gamma in ((1.0, "scale"), (1.0, 0.1), (10.0, "scale")):
            candidates.append(
                (
                    "svm",
                    f"kernel=rbf, C={c}, gamma={gamma}",
                    SVC(kernel="rbf", C=c, gamma=gamma),
                )
            )

    if model_name in ("all", "logistic_regression"):
        for c in (0.1, 1.0, 10.0, 100.0):
            candidates.append(
                (
                    "logistic_regression",
                    f"C={c}, solver=lbfgs, max_iter=2000",
                    LogisticRegression(
                        C=c, solver="lbfgs", max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                )
            )

    if model_name in ("all", "mlp"):
        # 先比较小型单隐层，再比较较宽或较深的网络与 L2 正则化。
        for hidden_layers, alpha in (
            ((50,), 0.0001),
            ((100,), 0.0001),
            ((100,), 0.001),
            ((100, 50), 0.001),
        ):
            candidates.append(
                (
                    "mlp",
                    f"hidden_layer_sizes={hidden_layers}, alpha={alpha}",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation="relu",
                        alpha=alpha,
                        max_iter=300,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                )
            )
    return candidates


def save_validation_tables(rows, model_name):
    """逐次保存所有已完成配置及每种算法当前最好的参数。"""
    results = pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy", "run_index"],
        ascending=[False, False, True],
    )
    best_by_algorithm = results.drop_duplicates("algorithm").sort_values("algorithm")
    RESULT_DIR.mkdir(exist_ok=True)
    suffix = "" if model_name == "all" else f"_{model_name}"
    results.to_csv(
        RESULT_DIR / f"validation_results{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_by_algorithm.to_csv(
        RESULT_DIR / f"best_parameters{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return results


def train_and_evaluate(X_train, y_train, X_val, y_val, model_name):
    """对候选参数分别拟合，以验证集 macro-F1、accuracy 依次排序。"""
    rows = []
    best_overall = None
    for number, (algorithm, parameters, model) in enumerate(
        build_model_candidates(model_name), start=1
    ):
        print(f"[{number}] 开始训练 {algorithm}: {parameters}", flush=True)
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(y_val, predictions, average="macro")
        rows.append(
            {
                "run_index": number,
                "algorithm": algorithm,
                "parameters": parameters,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "fit_seconds": fit_seconds,
            }
        )
        print(
            f"验证集 accuracy={accuracy:.4f}, macro-F1={macro_f1:.4f}, "
            f"训练耗时={fit_seconds:.1f}s",
            flush=True,
        )
        key = (macro_f1, accuracy)
        if best_overall is None or key > best_overall[0]:
            best_overall = (key, algorithm, parameters, model)
        save_validation_tables(rows, model_name)

    results = save_validation_tables(rows, model_name)

    # 输出参数比较图，不生成逐模型图表。
    os.environ.setdefault("MPLCONFIGDIR", str(RESULT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = results["algorithm"] + ": " + results["parameters"]
    figure, ax = plt.subplots(figsize=(13, max(5, len(results) * 0.4)))
    ax.barh(labels, results["macro_f1"], color="#4472c4")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Validation macro-F1")
    ax.set_title("Model and parameter comparison")
    figure.tight_layout()
    chart_name = (
        "validation_comparison.png" if model_name == "all"
        else f"validation_comparison_{model_name}.png"
    )
    figure.savefig(RESULT_DIR / chart_name, dpi=160)
    plt.close(figure)

    _, best_name, best_parameters, best_model = best_overall
    print(f"最佳配置: {best_name}: {best_parameters}")
    print(f"验证结果和每种算法最佳参数保存在: {RESULT_DIR}")
    return best_model


def fit_best_and_predict(best_model, model_name, X_all, y_all, X_test):
    """固定已选参数，在有标签数据上重拟合并预测无标签测试集。"""
    print("使用全部有标签数据重新拟合最佳配置。", flush=True)
    final_vectorizer = make_vectorizer()
    X_all_final = final_vectorizer.fit_transform(
        [preprocess_text(text) for text in X_all]
    )
    final_model = clone(best_model)
    final_model.fit(X_all_final, y_all)
    X_test_final = final_vectorizer.transform(
        [preprocess_text(text) for text in X_test]
    )
    predictions = final_model.predict(X_test_final)
    output_name = (
        "predictions.csv" if model_name == "all"
        else f"predictions_{model_name}.csv"
    )
    output_path = DATA_DIR / output_name
    pd.DataFrame(predictions).to_csv(output_path, index=False, header=False)
    print(f"测试集预测（{len(predictions)} 行）保存在: {output_path}")


def predict_from_saved_best(model_name="all"):
    """读取已完成的验证结果，仅重拟合最佳配置，避免重复搜索参数。"""
    suffix = "" if model_name == "all" else f"_{model_name}"
    best_path = RESULT_DIR / f"best_parameters{suffix}.csv"
    results_path = RESULT_DIR / f"validation_results{suffix}.csv"
    if not best_path.is_file() or not results_path.is_file():
        raise FileNotFoundError("找不到完整验证结果；请先运行 --train。")
    candidates = build_model_candidates(model_name)
    completed = pd.read_csv(results_path)
    expected = {(algorithm, parameters) for algorithm, parameters, _ in candidates}
    observed = list(zip(completed["algorithm"], completed["parameters"]))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("验证结果尚未覆盖全部候选配置，请重新运行 --train。")
    best_rows = pd.read_csv(best_path).sort_values(
        ["macro_f1", "accuracy", "run_index"],
        ascending=[False, False, True],
    )
    winner = best_rows.iloc[0]
    matches = [
        model
        for algorithm, parameters, model in candidates
        if algorithm == winner["algorithm"] and parameters == winner["parameters"]
    ]
    if len(matches) != 1:
        raise ValueError("保存的最佳参数与当前候选配置不一致，请重新运行 --train。")
    print(
        f"验证集最佳配置: {winner['algorithm']}: {winner['parameters']} "
        f"(macro-F1={winner['macro_f1']:.4f}, accuracy={winner['accuracy']:.4f})"
    )
    X_all, y_all, X_test = load_data()
    fit_best_and_predict(matches[0], model_name, X_all, y_all, X_test)


def main(model_name="all", save_predictions=False):
    X_all, y_all, X_test = load_data()
    print(f"有标签数据: {len(X_all)} 篇；无标签测试集: {len(X_test)} 篇")
    print(f"类别及数量: {pd.Series(y_all).value_counts().sort_index().to_dict()}")

    # 清洗是逐篇的确定性操作：每次运行只做一遍，不改写原始 CSV。
    X_all_clean = [preprocess_text(text) for text in X_all]
    train_idx, val_idx = train_test_split(
        np.arange(len(y_all)),
        test_size=VALIDATION_RATIO,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    vectorizer = make_vectorizer()
    X_train = vectorizer.fit_transform([X_all_clean[i] for i in train_idx])
    X_val = vectorizer.transform([X_all_clean[i] for i in val_idx])
    print(f"训练特征: {X_train.shape}；验证特征: {X_val.shape}")
    print(f"训练集清洗后空文本: {sum(not X_all_clean[i] for i in train_idx)}")

    best_model = train_and_evaluate(
        X_train, y_all[train_idx], X_val, y_all[val_idx], model_name
    )

    if save_predictions:
        # 用全部有标签数据拟合：验证子集也参与最终训练，测试集仍只用于预测。
        fit_best_and_predict(best_model, model_name, X_all, y_all, X_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("all",) + MODEL_NAMES, default="all",
        help="要训练的算法；默认 all，依次训练四种算法",
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="读取已保存的最佳参数，重拟合该配置并预测测试集",
    )
    parser.add_argument(
        "--list", action="store_true", help="只列出候选参数，不训练"
    )
    parser.add_argument(
        "--train", action="store_true", help="确认检查完成后，开始训练和验证"
    )
    args = parser.parse_args()
    if args.list and (args.train or args.predict):
        parser.error("--list 不能与 --train 或 --predict 同时使用")
    if args.list:
        for algorithm, parameters, _ in build_model_candidates(args.model):
            print(f"{algorithm}: {parameters}")
    elif args.train:
        main(model_name=args.model, save_predictions=args.predict)
    elif args.predict:
        predict_from_saved_best(args.model)
    else:
        parser.print_help()
        print("\n使用 --train 比较模型，或使用 --predict 重拟合已选最佳配置。")
