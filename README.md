# 🧠 Sentiment Analysis Pipeline — VADER & NLTK

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-154f3c?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end NLP pipeline for 3-class sentiment classification across multi-domain text corpora.**

[Overview](#-overview) • [Pipeline](#-pipeline) • [Datasets](#-datasets) • [Results](#-results) • [Getting Started](#-getting-started) • [Output](#-output-files)

</div>

---

## 📌 Overview

This project implements a complete **Natural Language Processing pipeline** that classifies text into three sentiment classes — **Positive**, **Negative**, and **Neutral** — across three distinct text domains: movie reviews, tweets, and product reviews.

It combines:
- **VADER** — a rule-based sentiment analyser optimised for social media text
- **Classical ML models** — Logistic Regression and Linear SVM trained on TF-IDF features
- **Rich visualisations** — word clouds, EDA charts, confusion matrices, and feature importance plots

> 💡 Built as a mini-project during engineering studies and later extended into a full reproducible analysis pipeline.

---

## 🔁 Pipeline

```
Raw Text (Multi-Domain)
        │
        ▼
┌───────────────────────┐
│   Text Cleaning       │  Remove URLs, @mentions, HTML, punctuation
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Tokenization        │  word_tokenize (NLTK punkt)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Stopword Removal     │  NLTK English stopwords (179 words)
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   POS Tagging         │  averaged_perceptron_tagger
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   Lemmatization       │  WordNetLemmatizer (POS-aware)
└──────────┬────────────┘
           │
      ┌────┴─────┐
      ▼          ▼
  VADER       TF-IDF
  Scoring     Features
  (rule)      (20k vocab, bigrams)
      │          │
      │     ┌────┴────┐
      │     ▼         ▼
      │  Logistic   Linear
      │  Regression  SVM
      │     │         │
      └─────┴────┬────┘
                 ▼
        3-Class Prediction
     Positive │ Negative │ Neutral
```

---

## 📂 Datasets

| Domain | Source | Size | Classes |
|--------|--------|------|---------|
| **Movie Reviews** | NLTK `movie_reviews` corpus | 2,000 reviews | Positive · Negative |
| **Tweets** | NLTK `twitter_samples` corpus | 10,000 tweets | Positive · Negative |
| **Product Reviews** | Inline samples (Amazon-style) | 45 reviews | Positive · Negative · Neutral |

> All datasets are loaded automatically — no manual download required. NLTK resources are fetched programmatically at runtime.

---

## 📊 Results

### VADER vs ML — Product Reviews (3-class)

| Model | Accuracy |
|-------|----------|
| VADER (rule-based) | ~0.73 |
| Logistic Regression | ~0.87 |
| **Linear SVM** | **~0.91** |

### Cross-Validation (5-Fold, Full Corpus)

| Model | CV Accuracy | Std Dev |
|-------|-------------|---------|
| Logistic Regression | ~0.89 | ±0.01 |
| Linear SVM | ~0.92 | ±0.01 |

> Results may vary slightly depending on Python/library versions.

---

## 🖼️ Output Files

The notebook generates the following artefacts automatically:

| File | Description |
|------|-------------|
| `eda_overview.png` | Label distribution, domain breakdown, word count histogram |
| `wordclouds.png` | Word clouds for Positive / Negative / Neutral classes |
| `vader_analysis.png` | Compound score KDE + VADER confusion matrix |
| `confusion_matrices.png` | Side-by-side confusion matrices for both ML models |
| `model_comparison.png` | CV vs Test accuracy bar chart |
| `top_features.png` | Top TF-IDF coefficients per sentiment class |
| `sentiment_corpus.csv` | Balanced, processed dataset with VADER scores |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aamir-droid-creator-lgtm/Sentiment-Analysis-NLP.git
cd Sentiment-Analysis-NLP

# 2. Install dependencies
pip install -r requirements.txt
```

### Run in VS Code

1. Open the project folder in **VS Code**
2. Open `Sentiment_Analysis_Pipeline.ipynb`
3. Select your Python interpreter as the kernel (top-right corner)
4. Press **Run All** (`Ctrl+Shift+P` → *Notebook: Run All Cells*)

### Run in Jupyter

```bash
jupyter notebook Sentiment_Analysis_Pipeline.ipynb
```

> The first two cells handle all package installation and NLTK resource downloads automatically.

---

## 🧪 Live Inference

Use the built-in `predict_sentiment()` function to classify any text:

```python
predict_sentiment("This product is absolutely amazing, works perfectly!")
# {
#   'vader_pred':      'positive',
#   'Linear SVM_pred': 'positive',
#   'vader_compound':  0.7003
# }

predict_sentiment("Arrived on time. Does what it says.")
# {
#   'vader_pred':      'neutral',
#   'Linear SVM_pred': 'neutral',
#   'vader_compound':  0.0772
# }
```

---

## 🗂️ Project Structure

```
Sentiment-Analysis-NLP/
│
├── Sentiment_Analysis_Pipeline.ipynb   # Main notebook (10 sections)
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **NLTK** | Tokenization, stopwords, lemmatization, VADER, corpora |
| **Scikit-learn** | TF-IDF, Logistic Regression, Linear SVM, evaluation |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Visualisations |
| **WordCloud** | Word cloud generation |

---

## 📖 Notebook Sections

| # | Section |
|---|---------|
| 0 | Setup & Dependencies |
| 1 | Data Loading — Multi-Domain Corpus |
| 2 | Exploratory Data Analysis (EDA) |
| 3 | Text Preprocessing Pipeline |
| 4 | VADER Rule-Based Sentiment Analysis |
| 5 | Feature Engineering — TF-IDF |
| 6 | ML Classification Models |
| 7 | Model Evaluation & Comparison |
| 8 | VADER vs ML Head-to-Head |
| 9 | Live Inference |

---

## 👤 Author

**Aamir Khan**
- GitHub: [@Aamir-droid-creator-lgtm](https://github.com/Aamir-droid-creator-lgtm)
- Email: aamirak.pathaan@gmail.com

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with Python · NLTK · VADER · Scikit-learn</sub>
</div>
