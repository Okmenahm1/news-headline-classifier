import os
import re
import math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "is", "are", "was", "were", "be", "been", "it",
    "this", "that", "these", "those", "as", "about", "into", "over",
    "after", "before", "up", "down", "out", "off", "so", "but", "if",
    "then", "than", "too", "very", "can", "could", "should", "would",
    "will", "just"
}


def preprocess_text(text: str):
   
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 2]
    return tokens


def build_vocabulary(texts, max_features=2000):
 
    term_counter = Counter()
    doc_freq = Counter()
    num_docs = len(texts)

    for text in texts:
        tokens = preprocess_text(text)
        term_counter.update(tokens)
        unique_tokens = set(tokens)
        doc_freq.update(unique_tokens)

    most_common = term_counter.most_common(max_features)
    word2idx = {word: i for i, (word, _) in enumerate(most_common)}

    idf = np.zeros(len(word2idx), dtype=np.float32)
    for word, idx in word2idx.items():
        df = doc_freq.get(word, 1)
        idf[idx] = math.log((1.0 + num_docs) / (1.0 + df)) + 1.0

    return word2idx, idf


def vectorize_texts(texts, word2idx, idf):
    
    num_docs = len(texts)
    vocab_size = len(word2idx)
    X = np.zeros((num_docs, vocab_size), dtype=np.float32)

    for i, text in enumerate(texts):
        tokens = preprocess_text(text)
        if not tokens:
            continue

        counts = Counter(tokens)
        doc_len = len(tokens)

        for token, count in counts.items():
            if token in word2idx:
                j = word2idx[token]
                tf = count / doc_len
                X[i, j] = tf * idf[j]

    return X



def softmax(z):
  
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def train_logistic_regression(X, y, num_classes, lr=0.3, epochs=40):
    """
    Train multiclass softmax logistic regression using gradient descent.
    X: (n_samples, n_features)
    y: (n_samples,) with integer labels 0..K-1
    """
    n_samples, n_features = X.shape
    W = np.zeros((n_features, num_classes), dtype=np.float32)
    b = np.zeros(num_classes, dtype=np.float32)

    for epoch in range(epochs):
        scores = X @ W + b 
        probs = softmax(scores)

 
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(n_samples), y] = 1.0

        error = probs - y_onehot
        dW = (X.T @ error) / n_samples
        db = np.mean(error, axis=0)

        W -= lr * dW
        b -= lr * db

       

    return W, b


def predict_logistic_regression(X, W, b):
    scores = X @ W + b
    probs = softmax(scores)
    return np.argmax(probs, axis=1)



class TreeNode:
    def __init__(self, feature_index=None, left=None, right=None, prediction=None):
        self.feature_index = feature_index 
        self.left = left
        self.right = right
        self.prediction = prediction       


def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))


def best_split_binary_feature(X, y):

    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None, 0.0

    parent_entropy = entropy(y)
    best_gain = 0.0
    best_feature = None

    for j in range(n_features):
        col = X[:, j]
        left_mask = col == 0
        right_mask = col == 1

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue

        y_left = y[left_mask]
        y_right = y[right_mask]

        H_left = entropy(y_left)
        H_right = entropy(y_right)

        n_left = len(y_left)
        n_right = len(y_right)

        child_entropy = (n_left / n_samples) * H_left + (n_right / n_samples) * H_right
        gain = parent_entropy - child_entropy

        if gain > best_gain:
            best_gain = gain
            best_feature = j

    return best_feature, best_gain


def build_decision_tree(X, y, depth=0, max_depth=4, min_samples=40):
   
    num_samples = len(y)
    num_labels = len(np.unique(y))

    if depth >= max_depth or num_labels == 1 or num_samples < min_samples:
        values, counts = np.unique(y, return_counts=True)
        prediction = values[np.argmax(counts)]
        return TreeNode(prediction=prediction)

    feature_index, gain = best_split_binary_feature(X, y)
    if feature_index is None or gain <= 0.0:
        values, counts = np.unique(y, return_counts=True)
        prediction = values[np.argmax(counts)]
        return TreeNode(prediction=prediction)

    col = X[:, feature_index]
    left_mask = col == 0
    right_mask = col == 1

    left_child = build_decision_tree(
        X[left_mask], y[left_mask],
        depth=depth + 1, max_depth=max_depth, min_samples=min_samples
    )
    right_child = build_decision_tree(
        X[right_mask], y[right_mask],
        depth=depth + 1, max_depth=max_depth, min_samples=min_samples
    )

    return TreeNode(feature_index=feature_index, left=left_child, right=right_child)


def predict_one_tree(x, node):
    while node.prediction is None:
        if x[node.feature_index] == 0:
            node = node.left
        else:
            node = node.right
    return node.prediction


def predict_tree(X, root):
    preds = [predict_one_tree(x, root) for x in X]
    return np.array(preds, dtype=int)



def assign_positions(node, depth, x_counter, positions, depths):
   
    if node is None:
        return

    assign_positions(node.left, depth + 1, x_counter, positions, depths)

    x = x_counter[0]
    positions[node] = (x, depth)
    depths[node] = depth
    x_counter[0] += 1

    assign_positions(node.right, depth + 1, x_counter, positions, depths)


def plot_decision_tree_nice(root, feature_names):
  
    if root is None:
        return

    positions = {}
    depths = {}
    assign_positions(root, depth=0, x_counter=[0], positions=positions, depths=depths)

    xs = [pos[0] for pos in positions.values()]
    max_x = max(xs) if xs else 1
    max_depth = max(depths.values()) if depths else 1

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_axis_off()

    for node, (x, depth) in positions.items():
        norm_x = x / max_x if max_x > 0 else 0.5
        norm_y = 1.0 - (depth / (max_depth + 1))

        if node.prediction is not None:
            text = f"Class = {node.prediction}"
        else:
            word = feature_names[node.feature_index]
            text = f"Feature: '{word}'"

        ax.text(
            norm_x,
            norm_y,
            text,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#BFE3FF", ec="#2F3B45"),
        )

    for node, (x, depth) in positions.items():
        parent_x = x / max_x if max_x > 0 else 0.5
        parent_y = 1.0 - (depth / (max_depth + 1))

        if node.left is not None:
            child_x, child_depth = positions[node.left]
            cx = child_x / max_x if max_x > 0 else 0.5
            cy = 1.0 - (child_depth / (max_depth + 1))
            ax.plot([parent_x, cx], [parent_y - 0.02, cy + 0.02], "-k")

        if node.right is not None:
            child_x, child_depth = positions[node.right]
            cx = child_x / max_x if max_x > 0 else 0.5
            cy = 1.0 - (child_depth / (max_depth + 1))
            ax.plot([parent_x, cx], [parent_y - 0.02, cy + 0.02], "-k")

    ax.set_title("Decision Tree", fontsize=14)
    plt.tight_layout()
    plt.show()



def compute_metrics(y_true, y_pred, num_classes=4):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return accuracy, macro_precision, macro_recall, macro_f1



def main():
    DATA_PATH = "ag_news.csv"
    df = pd.read_csv(DATA_PATH)

    df = df.sample(n=15000, random_state=42)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    num_samples = len(df)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(num_samples * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    texts_train = [texts[i] for i in train_idx]
    texts_test = [texts[i] for i in test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    print("Train size:", len(texts_train), " Test size:", len(texts_test))

    print("\n[1] Building vocabulary and TF-IDF...")
    word2idx, idf = build_vocabulary(texts_train, max_features=2000)
    feature_names = [None] * len(word2idx)
    for w, j in word2idx.items():
        feature_names[j] = w

    X_train = vectorize_texts(texts_train, word2idx, idf)
    X_test = vectorize_texts(texts_test, word2idx, idf)

    print("TF-IDF shapes:", X_train.shape, X_test.shape)

    num_classes = 4  

    print("\n[2] Training Logistic Regression ...")
    W, b = train_logistic_regression(
        X_train, y_train, num_classes,
        lr=0.3, epochs=40
    )

    y_pred_log = predict_logistic_regression(X_test, W, b)
    acc_l, prec_l, rec_l, f1_l = compute_metrics(y_test, y_pred_log, num_classes)

    print("\n===== Logistic Regression Performance =====")
    print(f"Accuracy : {acc_l:.4f}")
    print(f"Precision: {prec_l:.4f}")
    print(f"Recall   : {rec_l:.4f}")
    print(f"F1-score : {f1_l:.4f}")

    print("\n[3] Training Decision Tree ...")
    max_tree_features = 200
    X_train_bin = (X_train[:, :max_tree_features] > 0).astype(int)
    X_test_bin = (X_test[:, :max_tree_features] > 0).astype(int)
    tree_feature_names = feature_names[:max_tree_features]

    tree_root = build_decision_tree(
        X_train_bin, y_train,
        depth=0, max_depth=4, min_samples=40
    )

    y_pred_tree = predict_tree(X_test_bin, tree_root)
    acc_t, prec_t, rec_t, f1_t = compute_metrics(y_test, y_pred_tree, num_classes)

    print("\n===== Decision Tree Performance =====")
    print(f"Accuracy : {acc_t:.4f}")
    print(f"Precision: {prec_t:.4f}")
    print(f"Recall   : {rec_t:.4f}")
    print(f"F1-score : {f1_t:.4f}")

    models = ["Decision Tree", "Logistic Regression"]
    accuracies = [acc_t, acc_l]
    precisions = [prec_t, prec_l]
    recalls = [rec_t, rec_l]
    f1s = [f1_t, f1_l]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(8, 6))
    plt.bar(x - 1.5 * width, accuracies, width, label="Accuracy")
    plt.bar(x - 0.5 * width, precisions, width, label="Precision")
    plt.bar(x + 0.5 * width, recalls, width, label="Recall")
    plt.bar(x + 1.5 * width, f1s, width, label="F1-score")

    plt.xticks(x, models)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n[4] Drawing Decision Tree Figure...")
    plot_decision_tree_nice(tree_root, tree_feature_names)


if __name__ == "__main__":
    main()
