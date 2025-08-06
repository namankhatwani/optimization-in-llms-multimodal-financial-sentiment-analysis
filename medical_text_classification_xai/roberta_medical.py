!pip install -q transformers datasets scikit-learn
import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset, DatasetDict
train_df = pd.read_csv("/content/drive/MyDrive/week5/medical_train.csv")
test_df = pd.read_csv("/content/drive/MyDrive/week5/medical_test.csv")

label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['medical_specialty'])
test_df['label'] = label_encoder.transform(test_df['medical_specialty'])

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_fn(example):
    return tokenizer(example["transcription"], padding="max_length", truncation=True, max_length=256)

train_dataset = Dataset.from_pandas(train_df[['transcription', 'label']])
test_dataset = Dataset.from_pandas(test_df[['transcription', 'label']])

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["transcription"])
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
training_args = TrainingArguments(
    output_dir="./results_roberta_medical",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs_roberta",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    seed=42
)

from transformers import Trainer
from datasets import load_metric
!pip install evaluate
import evaluate
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy.compute(predictions=preds, references=p.label_ids)
    f1_score = f1.compute(predictions=preds, references=p.label_ids, average="weighted")
    return {
        "accuracy": acc["accuracy"],
        "f1": f1_score["f1"]  # ✅ Just return the float
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer)
)
trainer.train()
preds_output = trainer.predict(tokenized_dataset["test"])
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
preds_output = trainer.predict(tokenized_dataset["test"])
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get class names
class_names = label_encoder.classes_

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - RoBERTa")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
