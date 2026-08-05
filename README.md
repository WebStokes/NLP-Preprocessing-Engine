<div align="center">

# 🧹 NLP Preprocessing Engine

**A configurable text-cleaning pipeline plus two applied NLP demos — IMDB sentiment classification and a Transformers-based chatbot.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-yellow?style=for-the-badge)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E?style=for-the-badge)

</div>

---

## 📖 Overview

Built as an internship task (Innomatics Research Labs), this repo centers on a robust, reusable **NLP preprocessing pipeline** and demonstrates it in two applied contexts: sentiment analysis on movie reviews, and a conversational chatbot.

## ✨ What's Inside

### 1. `NLP_Preprocessing_Engine.ipynb` — the core pipeline
A `preprocess_text()` function and full pipeline that:
- Strips URLs, emails, and emojis
- Expands contractions (e.g. "don't" → "do not") and lowercases text
- Tokenizes, removes stopwords, and applies stemming/lemmatization (NLTK + spaCy)
- Includes error handling for edge cases (empty strings, emoji-only or numeric-only input)
- Provides token frequency analytics across a batch of sentences

### 2. `sentiment_analysis.ipynb` — applied ML
Classifies IMDB movie reviews as positive/negative:
- Custom text cleaning (lowercase, URL removal, stemming)
- Feature extraction with `CountVectorizer` (Bag-of-Words) and `TfidfVectorizer`
- Models trained and compared: **Logistic Regression** and **Multinomial Naive Bayes**

### 3. `chatbot.py` — conversational AI demo
A terminal chatbot using Hugging Face's `microsoft/DialoGPT-medium` model via `transformers`, maintaining conversational context across turns.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Core NLP | NLTK, spaCy, `contractions` |
| ML | scikit-learn (`CountVectorizer`, `TfidfVectorizer`, `LogisticRegression`, `MultinomialNB`) |
| Deep Learning | Hugging Face `transformers`, PyTorch (DialoGPT) |
| Data | pandas, IMDB Dataset (50K reviews) |

## 🚀 Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/WebStokes/NLP-Preprocessing-Engine.git
cd NLP-Preprocessing-Engine

pip install nltk spacy contractions pandas scikit-learn transformers torch jupyter
python -m spacy download en_core_web_sm
```

### Run the preprocessing engine or sentiment notebook

```bash
jupyter notebook
```
Open `NLP_Preprocessing_Engine.ipynb` or `sentiment_analysis.ipynb` (make sure `IMDB Dataset.csv` is in the working directory for the latter).

### Run the chatbot

```bash
python chatbot.py
```
> Note: the first run will download the `DialoGPT-medium` model (~1.5GB) from Hugging Face.

## 🗺️ Roadmap

- [ ] Package `preprocess_text()` as an installable module
- [ ] Add evaluation metrics (accuracy/F1) to the sentiment notebook output in the README
- [ ] Swap DialoGPT for a more modern instruction-tuned chat model
- [ ] Add unit tests for the preprocessing edge cases



## 👤 Author

**WebStokes** — [GitHub Profile](https://github.com/WebStokes)
