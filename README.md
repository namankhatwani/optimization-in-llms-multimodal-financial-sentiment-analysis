# Optimization in Large Language Models (LLMs)

This repository contains the complete code and documentation for my internship project at **Adrosonic**, focusing on **Optimization in Large Language Models** across multiple domains, culminating in a novel **Multimodal Financial Sentiment Classification** system.

## 📌 Overview
The project is structured into five progressive phases:

1. **Financial Sentiment Classification (Text-only)** – Baseline models trained on the Financial PhraseBank dataset using DistilBERT, MiniTransformer, and TinyTransformer.
2. **Multilingual Sentiment Classification** – XLM-RoBERTa, mBERT, and DistilBERT-Multilingual evaluated with structured hyperparameter tuning.
3. **Deep Hyperparameter Tuning** – Systematic tuning of XLM-RoBERTa to study effects of LR, warmup, weight decay, and optimizers.
4. **Medical Text Classification + Explainability** – RoBERTa fine-tuned with Attention Maps, Integrated Gradients, and Layer-wise Relevance Propagation.
5. **Novel Work: Multimodal Financial Sentiment Classification** – FinBERT (text) + ResNet-50 (images) fusion model on a translated, sentiment-labeled subset of the FinVis-GPT dataset.

---

## 🧠 Key Insights
- **Model depth and optimizer choice** significantly affect performance in small datasets.
- **Adafactor with LR decay** and **warmup steps** improve multilingual robustness.
- **Explainability tools** validate predictions in high-stakes domains.
- **Multimodal fusion** boosts accuracy and neutral sentiment recall in financial sentiment classification.

---

## 📊 Results Snapshot
| Phase | Best Model | Accuracy | Macro F1 |
|-------|-----------|----------|----------|
| Financial (Text) | DistilBERT | 76.80% | 0.6115 |
| Multilingual | XLM-R + Adafactor | 70.70% | 0.7196 |
| Medical | RoBERTa | 0.74 | 0.73 |
| Multimodal Finance | FinBERT + ResNet-50 | **83.00%** | **0.83** |

---

## 📂 Repository Layout
- `phase1_financial_phrasebank/` – Code for baseline financial sentiment experiments.
- `phase2_multilingual_sentiment/` – Multilingual models and tuning.
- `phase3_hp_tuning_xlm_roberta/` – Deep HP tuning scripts.
- `phase4_medical_text_xai/` – Medical classification with XAI.
- `phase5_multimodal_financial_sentiment/` – Novel multimodal model code.
- `methodology.md` – Step-by-step workflow.
- `experiments.md` – Results & tables.
- `references.md` – Citations.

---

## ⚙️ Requirements
```bash
pip install -r requirements.txt
