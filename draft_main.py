"""新闻文本十分类：加载数据、划分验证集、预处理和提取 TF-IDF 特征。"""

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
VALIDATION_RATIO = 0.2

# 邮件地址和链接中的用户名、域名通常与新闻类别无关。
CONTACT_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", re.IGNORECASE
)
ATTRIBUTION_PATTERN = re.compile(r"^(?:In article\b.*|On .+ wrote:)\s*$", re.IGNORECASE)
PUNCTUATION_PATTERN = re.compile(r"[^A-Za-z0-9\s]")


def load_data():
    """读取已给定的训练集和无标签测试集。"""
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")
    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].to_numpy()
    X_test_unlabeled = test_df["text"].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled


def preprocess_text(text):
    """保留帖子的原创正文，去掉邮件头、引用、签名和非词字符。"""
    text = text.replace("\r\n", "\n")
    # 本数据集的邮件头与正文由第一个空行分隔。
    body = text.split("\n\n", maxsplit=1)[-1]
    body_lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped == "--":
            break  # 标准邮件签名从这里开始。
        if stripped.startswith(">") or ATTRIBUTION_PATTERN.match(stripped):
            continue
        body_lines.append(line)

    body = " ".join(body_lines)
    body = CONTACT_PATTERN.sub(" ", body)
    body = PUNCTUATION_PATTERN.sub(" ", body)
    return " ".join(body.split())


def main():
    X_all, y_all, X_test_unlabeled = load_data()
    print(f"原始训练集: {len(X_all)} 篇；无标签测试集: {len(X_test_unlabeled)} 篇")
    print(f"类别及数量: {pd.Series(y_all).value_counts().sort_index().to_dict()}")

    # 按类别分层，确保十个类别在训练集和验证集中的比例接近。
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=VALIDATION_RATIO,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    X_train_clean = [preprocess_text(text) for text in X_train]
    X_val_clean = [preprocess_text(text) for text in X_val]
    X_test_clean = [preprocess_text(text) for text in X_test_unlabeled]

    # 英文停用词、大小写和分词交给向量器处理；保留单词及相邻双词。
    # 词干化可能损失产品名、专有名词等信息，此处不启用。
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=20000,
        sublinear_tf=True,
    )
    # 只能在训练子集上拟合词汇表和 IDF，避免验证集、测试集信息泄漏。
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)
    X_val_tfidf = vectorizer.transform(X_val_clean)
    X_test_tfidf = vectorizer.transform(X_test_clean)

    print(f"训练集: {X_train_tfidf.shape}，验证集: {X_val_tfidf.shape}")
    print(f"无标签测试集: {X_test_tfidf.shape}")
    print(f"训练/验证标签: {len(y_train)} / {len(y_val)}")
    print(f"训练集清洗后空文本: {sum(not text for text in X_train_clean)}")
    print(f"首篇清洗后正文: {X_train_clean[0][:200]}")

    # 下一步可用 X_train_tfidf、y_train 训练分类器，
    # 用 X_val_tfidf、y_val 比较算法，再处理 X_test_tfidf。


if __name__ == "__main__":
    main()
