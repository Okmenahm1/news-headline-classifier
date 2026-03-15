# News Headline Classifier

A machine learning project that classifies news headlines into 4 categories using two classifiers built from scratch with NumPy — **Logistic Regression** and **Decision Tree**.

## Authors

- **Muhammad Aljamal**
- **Tariq Ladadweh**

---

## Dataset

[AG News](https://huggingface.co/datasets/fancyzhx/ag_news) — a benchmark news classification dataset with 4 categories:

| Label | Category |
|-------|----------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

15,000 samples are used (80% train / 20% test).

---

## How It Works

### 1. Text Preprocessing
- Lowercasing and punctuation removal
- Stopword filtering
- TF-IDF vectorization (top 2000 features)

### 2. Logistic Regression
- Multiclass softmax regression
- Trained with gradient descent (lr=0.3, 40 epochs)

### 3. Decision Tree
- Binary feature splits on TF-IDF presence (top 200 features)
- Max depth: 4, Min samples per leaf: 40
- Information gain via entropy

---

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~88% | ~88% | ~88% | ~88% |
| Decision Tree | ~75% | ~75% | ~74% | ~74% |

### Performance Chart

![Model Performance Comparison](results/performance_comparison.png)

> The bar chart compares Accuracy, Precision, Recall, and F1-Score for both models side by side.

### Decision Tree Visualization

![Decision Tree](results/decision_tree.png)

> The tree splits on key TF-IDF features (words) to route headlines to one of the 4 news classes.

---

## Project Structure

```
news-headline-classifier/
├── ag_news.csv               # Dataset
├── Classification_project.py # Main training & evaluation script
├── download_dataset.py       # Script to download the dataset
└── README.md
```

---

## How to Run

```bash
# Install dependencies
pip install numpy pandas matplotlib

# Run the classifier
python Classification_project.py
```

---

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
