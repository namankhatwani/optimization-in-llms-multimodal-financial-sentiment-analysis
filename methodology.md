# Methodology

This document details the step-by-step methodology for each phase of the internship project **"Optimization in Large Language Models (LLMs)"**, culminating in the novel **Multimodal Financial Sentiment Classification** work.

---

## Financial Sentiment Classification (Text-only)

**Dataset:** Financial PhraseBank (≥75% annotator agreement).  
**Models:** DistilBERT-style, MiniTransformer, TinyTransformer (trained from scratch).  
**Goal:** Establish baselines and study the effect of model depth and optimizer choice in a small, domain-specific dataset.

**Approach:**
1. Preprocessed data (lowercasing, punctuation/stopword removal, TF-IDF & sentiment distribution analysis).
2. Stratified splits to handle class imbalance.
3. Training with AdamW and Adafactor, early stopping, LR scheduling.

**Note**  
To understand transformer depth vs. generalization and set benchmarks for future multilingual and multimodal experiments.

---

## Multilingual Sentiment Classification

**Dataset:** Balanced dataset with 3 sentiment classes (negative, neutral, positive) in six languages.  
**Models:** DistilBERT-Multilingual, mBERT, XLM-RoBERTa.  
**Goal:** Test cross-lingual robustness and quantify gains from structured hyperparameter tuning.

**Approach:**
1. Text normalization (removal of URLs, hashtags, emojis, non-alphabetic characters).
2. Fine-tuned multilingual models with controlled variations in:
   - Optimizer (AdamW, Adafactor)
   - Learning rate schedule
   - Warmup steps
   - Weight decay
   - Batch size & epochs
3. Evaluated with accuracy, macro F1, and per-language confusion matrices.

**Notes**  
To select the most robust multilingual model (XLM-R) for deeper hyperparameter exploration.

---

## Hyperparameter Tuning (XLM-RoBERTa)

**Goal:** Systematically evaluate the impact of LR, warmup, weight decay, optimizer, and batch size on multilingual generalization.

**Approach:**
1. Ran multiple tuning experiments using Adafactor and AdamW.
2. Applied LR decay schedules with early stopping on validation F1.
3. Recorded results with wandb and TensorBoard.

**Notes**  
To identify hyperparameter configurations that balance training efficiency and performance, providing a blueprint for small-data multimodal training.

---

## Medical Text Classification + Explainability

**Dataset:** Medical transcriptions classified into 5 specialties:  
Cardiovascular/Pulmonary, Gastroenterology, Neurology, Orthopedic, Radiology.  
**Model:** RoBERTa fine-tuned for classification.

**Approach:**
1. Preprocessing: spell correction, abbreviation expansion, token normalization (no stopword removal).
2. Post-hoc explainability using:
   - Attention Maps
   - Integrated Gradients
   - Layer-wise Relevance Propagation (LRP)
   - Word Importance plots
3. Visualized token relevance for each prediction.

**Notes**  
To validate model predictions in high-stakes domains and understand token-level decision-making.

---

## Multimodal Financial Sentiment Classification (Novel Work)

**Dataset Creation:**
1. Selected 800 rich chart–text pairs from **FinVis-GPT instruction-tuning** data.
2. Translated Chinese text to English using Google Translate.
3. Sentiment-labeled texts (positive, neutral, negative) via GPT-4 with human verification.

**Models:**
1. **Text-only baseline:** FinBERT fine-tuned on translated text.
2. **Multimodal model:** FinBERT + ResNet-50 late fusion.

**Fusion Method:**
- Visual features from ResNet-50 projected to match text embedding dimensions.
- Text embeddings from FinBERT.
- Concatenated features passed through FC classification head.

**Notes**  
To test whether visual cues from financial charts improve sentiment prediction beyond text-only models.

---
