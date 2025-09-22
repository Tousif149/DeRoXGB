!pip install contractions
import contractions
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, roc_auc_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from torch.cuda.amp import autocast, GradScaler
import random
import os
import xgboost as xgb
import emoji

# SEED for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus} (PyTorch)")

# === SECTION 1 — PREPROCESSING ===

# Function to remove emojis
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    # Expand contractions first
    text = contractions.fix(text)
    text = remove_emojis(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower().strip()
    return text

# map labels to 2-class (0-negative, 1-positive)
def map_labels(labels):
    label_map = {'negative': 0, 'positive': 1}
    return [label_map.get(l.lower(), 0) if isinstance(l, str) else l for l in labels]

# Load dataset
data = pd.read_csv('augmented_tweets.csv')

# Preprocess text
data['text'] = data['text'].apply(preprocess_text)

# Remove empty texts
data = data[data['text'] != ""]

# Detect label column
label_column = 'airline_sentiment' if 'airline_sentiment' in data.columns else 'label'
print(f"Using label column: {label_column}")

# Map labels to 2-class
data['label'] = map_labels(data[label_column].values)

# Show initial label distribution
print("Initial label distribution:")
print(data['label'].value_counts(dropna=False))

def map_labels_3class(labels):
    label_map = {'negative': 0, 'positive': 1, 'neutral': 2}
    return [label_map.get(l.lower(), 2) if isinstance(l, str) else l for l in labels]

data['label_3'] = map_labels_3class(data[label_column].values)
data = data[data['label_3'] != 2]
data['label'] = data['label_3'].map({0:0, 1:1})

# Show label distribution after filtering
print("Label distribution after excluding Neutral:")
print(data['label'].value_counts(dropna=False))

# Split 
train_val, test = train_test_split(data, test_size=0.15, random_state=SEED, stratify=data['label'])
train, val = train_test_split(train_val, test_size=0.17647, random_state=SEED, stratify=train_val['label'])


train_texts = train['text'].values
train_labels = train['label'].values
val_texts = val['text'].values
val_labels = val['label'].values
test_texts = test['text'].values
test_labels = test['label'].values

# === SECTION 2 — HYPERPARAMETERS ===
max_length = 128
batch_size = 32
epochs = 1
patience = 3
learning_rate = 2e-5

# === UNIVERSAL EMBEDDING EXTRACT FUNCTION ===
def extract_embeddings(dataloader, encoder_model):
    encoder_model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels

# === TRAIN EPOCH ===
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# === EVALUATE EPOCH ===
def evaluate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# === SECTION 3 — DeBERTa-v3-large for Embeddings ===
tokenizer_deberta = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer_deberta(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }

train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)
test_dataset = SentimentDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

deberta_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-large', num_labels=2)
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs for PyTorch model")
    deberta_model = nn.DataParallel(deberta_model)
deberta_model.to(device)

optimizer = optim.NAdam(deberta_model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

best_val_acc_deberta = 0.0
counter = 0

print("Training DeBERTa v3 Large model...")
train_accuracies_deberta = []
val_accuracies_deberta = []
train_losses_deberta = []
val_losses_deberta = []

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(deberta_model, train_loader, optimizer, criterion, scaler)
    val_loss, val_acc = evaluate_epoch(deberta_model, val_loader, criterion)

    logger.info(f"Epoch {epoch + 1}: "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    print(f"==> Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    train_accuracies_deberta.append(train_acc)
    val_accuracies_deberta.append(val_acc)
    train_losses_deberta.append(train_loss)
    val_losses_deberta.append(val_loss)

    scheduler.step()

    if val_acc > best_val_acc_deberta:
        best_val_acc_deberta = val_acc
        counter = 0
        torch.save(deberta_model.state_dict(), 'deberta_v3_large_model.pt')
        print("Saved best model to deberta_v3_large_model.pt")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

df_stats_deberta = pd.DataFrame({
    'epoch': list(range(1, len(train_losses_deberta)+1)),
    'train_loss': train_losses_deberta,
    'val_loss': val_losses_deberta,
    'train_accuracy': train_accuracies_deberta,
    'val_accuracy': val_accuracies_deberta
})

df_stats_deberta.to_csv('training_validation_stats_deberta.csv', index=False)
print("Training and validation stats saved to 'training_validation_stats_deberta.csv'.")

print("\n DeBERTa fine-tuning completed!")

# === SECTION 4: DeBERTa EMBEDDINGS-> XGBoost ===
print("\n=== SECTION 4: DeBERTa EMBEDDINGS - XGBoost ===")

deberta_model.load_state_dict(torch.load('deberta_v3_large_model.pt'))
if isinstance(deberta_model, nn.DataParallel):
    deberta_encoder = deberta_model.module.deberta
else:
    deberta_encoder = deberta_model.deberta
deberta_encoder.to(device)

train_embeddings_deberta, train_labels_deberta = extract_embeddings(train_loader, deberta_encoder)
test_embeddings_deberta, test_labels_deberta = extract_embeddings(test_loader, deberta_encoder)

xgb_clf_deberta = xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=SEED,
    verbosity=1,
    n_jobs=-1
)

xgb_clf_deberta.fit(train_embeddings_deberta, train_labels_deberta)
xgb_preds_deberta = xgb_clf_deberta.predict(test_embeddings_deberta)
xgb_probs_deberta = xgb_clf_deberta.predict_proba(test_embeddings_deberta)[:, 1]

# === SECTION 7: RoBERTa-Large for Embeddings + XGBoost ===
print("\n=== SECTION 7: RoBERTa-Large EMBEDDINGS + XGBoost ===")

tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-large')

class SentimentDatasetRoberta(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer_roberta(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }

train_dataset_roberta = SentimentDatasetRoberta(train_texts, train_labels)
val_dataset_roberta = SentimentDatasetRoberta(val_texts, val_labels)
test_dataset_roberta = SentimentDatasetRoberta(test_texts, test_labels)

train_loader_roberta = DataLoader(train_dataset_roberta, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader_roberta = DataLoader(val_dataset_roberta, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader_roberta = DataLoader(test_dataset_roberta, batch_size=batch_size, shuffle=False, num_workers=4)

roberta_model = AutoModelForSequenceClassification.from_pretrained('roberta-large', num_labels=2)
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs for PyTorch model")
    roberta_model = nn.DataParallel(roberta_model)
roberta_model.to(device)

optimizer_roberta = optim.NAdam(roberta_model.parameters(), lr=learning_rate)
scheduler_roberta = optim.lr_scheduler.CosineAnnealingLR(optimizer_roberta, T_max=epochs)
criterion_roberta = nn.CrossEntropyLoss()
scaler_roberta = GradScaler()

best_val_acc_roberta = 0.0
counter = 0

print("Training RoBERTa-large model...")
train_accuracies_roberta = []
val_accuracies_roberta = []
train_losses_roberta = []
val_losses_roberta = []

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(roberta_model, train_loader_roberta, optimizer_roberta, criterion_roberta, scaler_roberta)
    val_loss, val_acc = evaluate_epoch(roberta_model, val_loader_roberta, criterion_roberta)

    logger.info(f"Epoch {epoch + 1}: "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    print(f"==> Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    train_accuracies_roberta.append(train_acc)
    val_accuracies_roberta.append(val_acc)
    train_losses_roberta.append(train_loss)
    val_losses_roberta.append(val_loss)

    scheduler_roberta.step()

    if val_acc > best_val_acc_roberta:
        best_val_acc_roberta = val_acc
        counter = 0
        torch.save(roberta_model.state_dict(), 'roberta_large_model.pt')
        print(" Saved best model to roberta_large_model.pt")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

df_stats_roberta = pd.DataFrame({
    'epoch': list(range(1, len(train_losses_roberta)+1)),
    'train_loss': train_losses_roberta,
    'val_loss': val_losses_roberta,
    'train_accuracy': train_accuracies_roberta,
    'val_accuracy': val_accuracies_roberta
})

df_stats_roberta.to_csv('training_validation_stats_roberta.csv', index=False)
print("Training and validation stats saved to 'training_validation_stats_roberta.csv'.")

print("\n RoBERTa fine-tuning completed!")

roberta_model.load_state_dict(torch.load('roberta_large_model.pt'))

print("\nExtracting RoBERTa embeddings for XGBoost...")
if isinstance(roberta_model, nn.DataParallel):
    roberta_encoder = roberta_model.module.roberta
else:
    roberta_encoder = roberta_model.roberta
roberta_encoder.to(device)

train_embeddings_roberta, train_labels_roberta = extract_embeddings(train_loader_roberta, roberta_encoder)
test_embeddings_roberta, test_labels_roberta = extract_embeddings(test_loader_roberta, roberta_encoder)

xgb_clf_roberta = xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=SEED,
    verbosity=1,
    n_jobs=-1
)

xgb_clf_roberta.fit(train_embeddings_roberta, train_labels_roberta)
xgb_preds_roberta = xgb_clf_roberta.predict(test_embeddings_roberta)
xgb_probs_roberta = xgb_clf_roberta.predict_proba(test_embeddings_roberta)[:, 1]

# === SECTION 8: ENSEMBLE — DeBERTa_XGBoost + RoBERTa_XGBoost ===
print("\n=== SECTION 8: ENSEMBLE — DeBERTa_XGBoost + RoBERTa_XGBoost ===")

# Prepare probabilities
xgb_probs_deberta = xgb_clf_deberta.predict_proba(test_embeddings_deberta)[:, 1]
xgb_probs_roberta = xgb_clf_roberta.predict_proba(test_embeddings_roberta)[:, 1]

# Averaging ensemble
avg_probs_xgb_dr = (xgb_probs_deberta + xgb_probs_roberta) / 2.0
avg_preds_xgb_dr = (avg_probs_xgb_dr > 0.5).astype(int)

print("\n=== ENSEMBLE (AVERAGING): DeBERTa_XGBoost + RoBERTa_XGBoost ===")
print("\nClassification Report:")
report = classification_report(test_labels_deberta, avg_preds_xgb_dr, target_names=['Negative', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_labels_deberta, avg_preds_xgb_dr, target_names=['Negative', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (weighted): {report['weighted avg']['precision']:.4f}")
print(f"Recall (weighted): {report['weighted avg']['recall']:.4f}")
print(f"F1-Score (weighted): {report['weighted avg']['f1-score']:.4f}")

cm_avg_xgb_dr = confusion_matrix(test_labels_deberta, avg_preds_xgb_dr)
print("\nConfusion Matrix:")
print(cm_avg_xgb_dr)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_avg_xgb_dr, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (AVERAGING) - DeBERTa_XGBoost + RoBERTa_XGBoost')
plt.show()

# === SECTION 9: AUC Computation and Saving ===
print("\n=== SECTION 9: AUC Computation and Metrics Summary ===")

def compute_auc(y_true, y_prob_positive):
    return roc_auc_score(y_true, y_prob_positive)

def plot_roc(y_true, y_prob_positive, model_name, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob_positive)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()

auc_summary_df = pd.DataFrame(columns=['Model', 'AUC', 'Precision', 'Recall', 'F1_Score'])

# ENSEMBLE DeBERTa_XGBoost + RoBERTa_XGBoost
precision_avg_xgb_dr, recall_avg_xgb_dr, f1_avg_xgb_dr, _ = precision_recall_fscore_support(
    test_labels_deberta, avg_preds_xgb_dr, average='weighted'
)
auc_xgb_dr_ensemble = compute_auc(test_labels_deberta, avg_probs_xgb_dr)
plot_roc(test_labels_deberta, avg_probs_xgb_dr, model_name='ENSEMBLE - DeBERTa_XGBoost + RoBERTa_XGBoost', save_path='ensemble_xgb_dr_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['ENSEMBLE - DeBERTa_XGBoost + RoBERTa_XGBoost', auc_xgb_dr_ensemble, precision_avg_xgb_dr, recall_avg_xgb_dr, f1_avg_xgb_dr]

# Print and save summary
print("\n=== Overall Metrics Summary ===")
print(auc_summary_df.round(4))
auc_summary_df.to_csv('metrics_summary_all_models.csv', index=False)
print("\n Metrics summary saved to 'metrics_summary_all_models.csv'")
