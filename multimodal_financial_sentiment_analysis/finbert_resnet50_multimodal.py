!pip install transformers datasets torchvision scikit-learn -q

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import os

df = pd.read_csv('/content/drive/MyDrive/fin_vis_gpt/instruction_sentiment_labeled_balanced.csv')

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['label'].map(label_map)

image_dir = '/content/drive/MyDrive/fin_vis_gpt/Instruction_tune_data_images'
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform, max_length=256):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load and process image
        image_path = os.path.join(image_dir, row['image'])
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        # Tokenize text
        encoding = self.tokenizer(
            row['text'], padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'image': image_tensor,
            'label': torch.tensor(row['label'])
        }
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_ds = MultimodalDataset(train_df, tokenizer, transform)
test_ds = MultimodalDataset(test_df, tokenizer, transform)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)
import torch.nn as nn

class MultiModalClassifier(nn.Module):
    def __init__(self, text_model_name="ProsusAI/finbert", num_classes=3):
        super().__init__()
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Identity()  # Remove final layer

        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.pooler_output  # [batch, 768]

        image_feat = self.image_model(image)  # [batch, 2048]

        combined = torch.cat((text_feat, image_feat), dim=1)  # [batch, 2816]
        return self.classifier(combined)
from torch.optim import AdamW
from sklearn.metrics import classification_report
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalClassifier().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            outputs = model(**inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive']))
for epoch in range(5):
    loss = train_epoch(model, train_loader)
    print(f"Epoch {epoch+1} Loss: {loss:.4f}")
    evaluate(model, test_loader)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_with_report(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive']))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'neutral', 'positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()
