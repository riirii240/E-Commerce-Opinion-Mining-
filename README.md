Voilà directement :

---

# 🛒 E-Commerce Opinion Mining — Sentiment Analysis & Topic Modeling

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-FFD21E?logo=huggingface)
![BERTopic](https://img.shields.io/badge/TopicModeling-BERTopic-8A2BE2)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Academic%20Project-lightgrey)

> **Authors:** Houda Moustaine · Aya Boughalem · Rihab Zitouni · Taha Hakim
> **Supervisor:** Dr. Hiba Tabbaa — EMSI Marrakech, Data Mining, May 2025

A complete **NLP pipeline** for analyzing Amazon customer reviews — from raw text to an interactive Streamlit dashboard. Combines **DistilBERT fine-tuning** for sentiment classification and **BERTopic** for automatic topic extraction, with actionable marketing recommendations.

---

## ✨ Key Results

| Metric | Value |
|--------|-------|
| 📦 Dataset | Amazon Reviews Multi — 5,000 reviews (EN) |
| 🎯 F1-macro (DistilBERT) | **≥ 0.90** |
| 🏷️ Sentiment classes | Positive / Neutral / Negative |
| 🔍 Topics extracted (BERTopic) | 10–20 coherent topics |
| 📊 Topic–rating association | χ² test significant (p < 0.001) |
| ⚡ Training time | ~4 min on Google Colab T4 GPU |
| 🔴 Critical topics identified | Battery (2.4/5) · Customer service (2.2/5) |

---

## 🏗️ Pipeline Architecture

```
Amazon Reviews Multi (Kaggle)
        ↓
   NLP Preprocessing
   ├── HTML stripping (BeautifulSoup)
   ├── Emoji conversion (emoji.demojize)
   ├── URL removal (regex)
   ├── Lowercasing + cleaning
   └── Lemmatization + stopwords (spaCy en_core_web_sm)
        ↓
   ┌─────────────────────┬──────────────────────┐
   │  Sentiment Analysis │   Topic Modeling      │
   │  DistilBERT fine-   │   BERTopic            │
   │  tuned (3 classes)  │   SBERT→UMAP→HDBSCAN  │
   └─────────────────────┴──────────────────────┘
        ↓
   Streamlit Interactive Dashboard
```

---

## 📁 Project Structure

```
opinion-mining-amazon/
├── webmining_complet.py          # Full pipeline script
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── outputs/
│   ├── sentiment_results.csv     # Reviews + predicted labels
│   ├── topic_results.csv         # Reviews + assigned topics
│   ├── wordcloud_positive.png    # Word cloud — positive reviews
│   ├── wordcloud_negative.png    # Word cloud — negative reviews
│   └── heatmap_topic_note.png    # Topic × rating heatmap
└── README.md
```

---

## 🚀 Quick Start

### Install dependencies
```bash
pip install transformers datasets torch bertopic streamlit
pip install spacy plotly wordcloud shap langdetect emoji beautifulsoup4
python -m spacy download en_core_web_sm
```

### Run on Google Colab
Open `webmining_complet.py` in Colab — GPU T4 recommended for fine-tuning (~4 min).

### Run dashboard locally
```bash
streamlit run app.py
```

### Expose on Colab via localtunnel
```bash
!streamlit run app.py &
!npx localtunnel --port 8501
```

---

## 🤖 Sentiment Analysis — DistilBERT

**Model:** `distilbert-base-uncased` fine-tuned for 3-class sentiment

| Label | Stars | Support |
|-------|-------|---------|
| Positive | 4–5 ⭐ | ~62% |
| Negative | 1–2 ⭐ | ~24% |
| Neutral | 3 ⭐ | ~14% |

**Training config:**

| Hyperparameter | Value |
|---|---|
| Max sequence length | 128 tokens |
| Batch size (train) | 32 |
| Learning rate | 2×10⁻⁵ |
| Epochs | 2 |
| Mixed precision | fp16 (GPU only) |

**Per-class results:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.91 | 0.89 | 0.90 |
| Neutral | 0.82 | 0.80 | 0.81 |
| Positive | 0.96 | 0.97 | 0.96 |
| **Macro avg** | **0.90** | **0.89** | **≥ 0.90** |

**Interpretability:** SHAP values computed on 20 reviews — top positive tokens: `excellent`, `perfect`, `love` / top negative: `broken`, `waste`, `terrible`.

---

## 🔍 Topic Modeling — BERTopic

**Pipeline:** SBERT embeddings (`all-MiniLM-L6-v2`, d=384) → UMAP (→5D) → HDBSCAN → c-TF-IDF

**Critical topics (avg rating < 3.0):**

| Topic | Keywords | Avg Rating |
|-------|----------|-----------|
| Battery | battery, charge, life, hour, last | ⚠️ 2.4 |
| Customer Service | customer, service, return, refund, help | ⚠️ 2.2 |

**Top-rated topics (avg rating ≥ 4.0):**

| Topic | Keywords | Avg Rating |
|-------|----------|-----------|
| Camera | photo, camera, picture, image, shot | ⭐ 4.4 |
| Value | price, value, money, worth | ⭐ 4.3 |
| Sound | sound, audio, bass, headphone | ⭐ 4.2 |

Topic–rating association validated by χ² test (p < 0.001).

---

## 📊 Streamlit Dashboard

5 interactive sections:
- **KPIs** — total reviews, avg rating, % positive/negative
- **Rating distribution** — Plotly histogram
- **Sentiment pie chart** — 3-class breakdown
- **Topic × rating heatmap** — RdYlGn colorscale, threshold at 3.0
- **Word clouds** — selectable by sentiment
- **Raw reviews table** — 20 sampled, expandable
- **Auto-alerts** — flags any topic with avg rating < 3.0

Sidebar filters: product category · sentiment · star range (1–5)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| NLP / DL | HuggingFace Transformers, DistilBERT |
| Topic Modeling | BERTopic (SBERT + UMAP + HDBSCAN) |
| Preprocessing | spaCy, BeautifulSoup, langdetect |
| Visualization | Streamlit, Plotly, WordCloud, Matplotlib |
| Interpretability | SHAP |
| Training | Google Colab T4 GPU, fp16 |
| Language | Python 3.11 |

---

## 📚 References

- Sanh et al. (2019) — *DistilBERT, a distilled version of BERT*
- Grootendorst (2022) — *BERTopic: Neural topic modeling*
- Hu & Liu (2004) — *Mining and summarizing customer reviews*, KDD

---

## 📝 Report

Full technical report (EDA, training protocol, SHAP analysis, marketing recommendations) available upon request.

---

## 📄 License

Academic project — EMSI Marrakech, Data Mining module, 2025.

---

**🔖 Topics GitHub :**
```
nlp  sentiment-analysis  topic-modeling  distilbert  bertopic  transformers  streamlit  amazon-reviews  huggingface  python  text-mining  web-mining
```
