# Experiments and Results

This document summarizes the results, tables, and key observations for each phase.

---

## Financial Sentiment (Text-only)

| Model            | Accuracy | Macro F1 | Negative F1 |
|------------------|----------|----------|-------------|
| DistilBERT-style | **76.80%** | **0.6115** | 0.3396 |
| MiniTransformer  | 73.33%   | 0.6131   | 0.4138 |
| TinyTransformer  | 67.20%   | 0.3958   | 0.0000 |

**Observation:** DistilBERT had the best generalization, proving depth’s advantage in small datasets.

---

## Multilingual Sentiment

| Model            | Optimizer | Best Run | Val F1  | Test F1 |
|------------------|-----------|----------|---------|---------|
| DistilBERT       | AdamW     | Run 2    | 59.17%  | 59.62%  |
| mBERT            | AdamW     | Run 1    | 62.61%  | 61.74%  |
| **XLM-R**        | Adafactor | Run 3    | **71.96%** | **70.70%** |

**Observation:** XLM-R with Adafactor + LR decay was most robust across languages.

---

## HP Tuning (XLM-RoBERTa)

**Key Findings:**
- Adafactor + decaying LR outperformed AdamW in multilingual setups.
- Large warmup steps prevented early overfitting.
- Early stopping on F1 ensured stable convergence.

---

## Medical Text Classification + XAI

**Classification Report:**
| Specialty                  | Precision | Recall | F1-score |
|----------------------------|-----------|--------|----------|
| Cardiovascular / Pulmonary | 0.83      | 0.91   | 0.87     |
| Gastroenterology           | 0.92      | 0.88   | 0.90     |
| Neurology                  | 0.62      | 0.60   | 0.61     |
| Orthopedic                 | 0.75      | 0.79   | 0.77     |
| Radiology                  | 0.51      | 0.45   | 0.48     |
| **Macro Avg**              | 0.73      | 0.73   | 0.73     |

**Observation:** Highest F1 in Gastroenterology, lowest in Radiology. XAI methods confirmed medical term focus.

---

## Multimodal Financial Sentiment (Novel Work)

**Text-only (FinBERT)**
| Class    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| Negative | 0.83      | 0.93   | 0.88     |
| Neutral  | 1.00      | 0.52   | 0.68     |
| Positive | 0.67      | 0.92   | 0.77     |
| **Macro Avg** | 0.83 | 0.79 | 0.78 |
| **Accuracy**  | **78.75%** |  |  |

**Multimodal (FinBERT + ResNet-50)**
| Class    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| Negative | 0.88      | 0.83   | 0.85     |
| Neutral  | 0.89      | 0.74   | 0.81     |
| Positive | 0.72      | 0.89   | 0.80     |
| **Macro Avg** | 0.83 | 0.82 | 0.82 |
| **Accuracy**  | **83.00%** |  |  |

**Observation:**  
+5% accuracy over text-only baseline, neutral recall improved from 0.52 → 0.74.

---

## Comparative Evaluation with Literature

| Paper / Source | Dataset | Model(s) | Accuracy | F1-score |
|----------------|---------|----------|----------|----------|
| Transforming Sentiment Analysis with ChatGPT (2023) | Bitcoin Twitter Price Movement | FinBERT, GPT-3.5 | 73.70% | 0.76 |
| Open-FinLLMs (2024) | Image + Text | Open-FinVLM | 81.00% | N/A |
| FinTral (2024) | Text, Numerical, Tabular, Chart | FinTral-DPO-T&R | 83.00% | N/A |
| **Proposed** | Image + Text | FinBERT + ResNet-50 | **83.00%** | **0.83** |
