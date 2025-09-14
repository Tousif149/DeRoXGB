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
import lightgbm as lgb
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

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus} (PyTorch)")

# === SECTION 1 — DATA LOADING, PREPROCESSING, SPLITS ===

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

# Function to map labels to 3-class (0-negative, 1-positive, 2-neutral)
def map_labels(labels):
    label_map = {'negative': 0, 'positive': 1, 'neutral': 2}
    return [label_map.get(l.lower(), 0) if isinstance(l, str) else (2 if l == 1 else (1 if l == 2 else 0)) for l in labels]

# Load augmented dataset
data = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')

# Preprocess text
data['text'] = data['text'].apply(preprocess_text)

# Remove empty texts
data = data[data['text'] != ""]

# Detect label column
label_column = 'airline_sentiment' if 'airline_sentiment' in data.columns else 'label'
print(f"Using label column: {label_column}")

# Map labels to 3-class first
data['label'] = map_labels(data[label_column].values)

# Show initial label distribution
print("Initial label distribution:")
print(data['label'].value_counts(dropna=False))

# Now exclude Neutral (label == 2)
data = data[data['label'] != 2]

# Show label distribution after filtering
print("Label distribution after excluding Neutral:")
print(data['label'].value_counts(dropna=False))

# Split → for 15% test and 15% val
train_val, test = train_test_split(data, test_size=0.15, random_state=SEED, stratify=data['label'])
train, val = train_test_split(train_val, test_size=0.17647, random_state=SEED, stratify=train_val['label'])

# Prepare arrays
train_texts = train['text'].values
train_labels = train['label'].values
val_texts = val['text'].values
val_labels = val['label'].values
test_texts = test['text'].values
test_labels = test['label'].values

print(data['label'].value_counts(dropna=False))

# === SECTION 2 — HYPERPARAMETERS ===
max_length = 128
batch_size = 32
epochs = 5
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

# === SECTION 3 — DeBERTa-v3-large CLASSIFIER ===
def train_epoch(model, loader, optimizer, criterion):
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
    train_loss, train_acc = train_epoch(deberta_model, train_loader, optimizer, criterion)
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
        print("✅ Saved best model to deberta_v3_large_model.pt")
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

print("\n✅ DeBERTa fine-tuning completed!")

# Load best model
deberta_model.load_state_dict(torch.load('deberta_v3_large_model.pt'))
deberta_model.eval()

test_predictions_deberta = []
test_true_labels_deberta = []
test_probs_deberta = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="DeBERTa Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = deberta_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = nn.functional.softmax(outputs.logits, dim=1)
        _, predicted = torch.max(probs, 1)

        test_predictions_deberta.extend(predicted.cpu().numpy())
        test_true_labels_deberta.extend(labels.cpu().numpy())
        test_probs_deberta.extend(probs[:, 1].cpu().numpy())

print("\n=== DeBERTa CLASSIFIER ===")
print("\nClassification Report:")
print(classification_report(test_true_labels_deberta, test_predictions_deberta, target_names=['Negative', 'Positive'], digits=4))

cm_deberta = confusion_matrix(test_true_labels_deberta, test_predictions_deberta)
print("\nConfusion Matrix:")
print(cm_deberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_deberta, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('DeBERTa Classifier - Confusion Matrix')
plt.show()

# Compute overall metrics for DeBERTa
precision_deberta, recall_deberta, f1_deberta, _ = precision_recall_fscore_support(
    test_true_labels_deberta, test_predictions_deberta, average='weighted'
)

# === SECTION 4: DeBERTa EMBEDDINGS → XGBoost ===
print("\n=== SECTION 4: DeBERTa EMBEDDINGS → XGBoost ===")

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

print("\n=== DeBERTa EMBEDDINGS + XGBoost CLASSIFIER ===")
print("\nClassification Report:")
print(classification_report(test_labels_deberta, xgb_preds_deberta, target_names=['Negative', 'Positive'], digits=4))

cm_xgb_deberta = confusion_matrix(test_labels_deberta, xgb_preds_deberta)
print("\nConfusion Matrix:")
print(cm_xgb_deberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb_deberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('DeBERTa Embeddings + XGBoost - Confusion Matrix')
plt.show()

# Compute overall metrics for DeBERTa + XGBoost
precision_xgb_deberta, recall_xgb_deberta, f1_xgb_deberta, _ = precision_recall_fscore_support(
    test_labels_deberta, xgb_preds_deberta, average='weighted'
)

# === SECTION 5: DeBERTa EMBEDDINGS → LightGBM ===
print("\n=== SECTION 5: DeBERTa EMBEDDINGS → LightGBM ===")

from lightgbm import early_stopping, log_evaluation

lgb_clf_deberta = lgb.LGBMClassifier(
    objective='binary',
    learning_rate=0.1,
    n_estimators=100,
    num_leaves=64,
    max_depth=-1,
    random_state=SEED,
    n_jobs=-1
)

eval_set = [(train_embeddings_deberta, train_labels_deberta), (test_embeddings_deberta, test_labels_deberta)]
eval_names = ['train', 'valid']
evals_result = {}  # Initialize evals_result dictionary

lgb_clf_deberta.fit(
    train_embeddings_deberta, train_labels_deberta,
    eval_set=eval_set,
    eval_names=eval_names,
    eval_metric='binary_logloss',
    callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=20),
        lgb.record_evaluation(evals_result)
    ]
)

lgb_preds_probs_deberta = lgb_clf_deberta.predict_proba(test_embeddings_deberta)[:, 1]
lgb_preds_deberta = lgb_clf_deberta.predict(test_embeddings_deberta)

print("\n=== DeBERTa EMBEDDINGS + LightGBM CLASSIFIER ===")
print("\nClassification Report:")
print(classification_report(test_labels_deberta, lgb_preds_deberta, target_names=['Negative', 'Positive'], digits=4))

cm_lgb_deberta = confusion_matrix(test_labels_deberta, lgb_preds_deberta)
print("\nConfusion Matrix:")
print(cm_lgb_deberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_lgb_deberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('DeBERTa Embeddings + LightGBM - Confusion Matrix')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(evals_result['train']['binary_logloss'], label='Train Binary Logloss')
plt.plot(evals_result['valid']['binary_logloss'], label='Valid Binary Logloss')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('LightGBM - Training and Validation Log Loss (DeBERTa Embeddings)')
plt.legend()
plt.grid()
plt.show()

# Compute overall metrics for DeBERTa + LightGBM
precision_lgb_deberta, recall_lgb_deberta, f1_lgb_deberta, _ = precision_recall_fscore_support(
    test_labels_deberta, lgb_preds_deberta, average='weighted'
)

# === SECTION 6: ENSEMBLE — DeBERTa_softmax + XGBoost + LightGBM ===
print("\n=== SECTION 6: ENSEMBLE — DeBERTa_softmax + XGBoost + LightGBM ===")

print("Recomputing DeBERTa softmax probabilities...")
deberta_model.eval()
deberta_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting with DeBERTa"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = deberta_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = nn.functional.softmax(outputs.logits, dim=1)
        deberta_probs.extend(probs[:, 1].cpu().numpy())

deberta_probs = np.array(deberta_probs)
xgb_probs_deberta = xgb_clf_deberta.predict_proba(test_embeddings_deberta)[:, 1]
lgb_probs_deberta = lgb_clf_deberta.predict_proba(test_embeddings_deberta)[:, 1]

avg_probs_deberta = (deberta_probs + xgb_probs_deberta + lgb_probs_deberta) / 3.0
avg_preds_deberta = (avg_probs_deberta > 0.5).astype(int)

print("\n=== ENSEMBLE (AVERAGING): DeBERTa_softmax + XGBoost + LightGBM ===")
print("\nClassification Report:")
print(classification_report(test_labels_deberta, avg_preds_deberta, target_names=['Negative', 'Positive'], digits=4))

cm_avg_deberta = confusion_matrix(test_labels_deberta, avg_preds_deberta)
print("\nConfusion Matrix:")
print(cm_avg_deberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_avg_deberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (AVERAGING) - DeBERTa_softmax + XGBoost + LightGBM')
plt.show()

# Compute overall metrics for Ensemble (Averaging)
precision_avg_deberta, recall_avg_deberta, f1_avg_deberta, _ = precision_recall_fscore_support(
    test_labels_deberta, avg_preds_deberta, average='weighted'
)

deberta_preds_majority = (deberta_probs > 0.5).astype(int)
xgb_preds_majority = (xgb_probs_deberta > 0.5).astype(int)
lgb_preds_majority = (lgb_probs_deberta > 0.5).astype(int)

final_majority_preds_deberta = []
for i in range(len(test_labels_deberta)):
    votes = [deberta_preds_majority[i], xgb_preds_majority[i], lgb_preds_majority[i]]
    final_pred = max(set(votes), key=votes.count)
    final_majority_preds_deberta.append(final_pred)

print("\n=== ENSEMBLE (MAJORITY VOTING): DeBERTa_softmax + XGBoost + LightGBM ===")
print("\nClassification Report:")
print(classification_report(test_labels_deberta, final_majority_preds_deberta, target_names=['Negative', 'Positive'], digits=4))

cm_majority_deberta = confusion_matrix(test_labels_deberta, final_majority_preds_deberta)
print("\nConfusion Matrix:")
print(cm_majority_deberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_majority_deberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (MAJORITY VOTING) - DeBERTa_softmax + XGBoost + LightGBM')
plt.show()

# Compute overall metrics for Ensemble (Majority Voting)
precision_majority_deberta, recall_majority_deberta, f1_majority_deberta, _ = precision_recall_fscore_support(
    test_labels_deberta, final_majority_preds_deberta, average='weighted'
)

# === SECTION 7: RoBERTa-Large CLASSIFIER + EMBEDDINGS + XGBoost ===
print("\n=== SECTION 7: RoBERTa-Large CLASSIFIER + XGBoost ===")

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
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(roberta_model, train_loader_roberta, optimizer_roberta, criterion_roberta)
    val_loss, val_acc = evaluate_epoch(roberta_model, val_loader_roberta, criterion_roberta)

    logger.info(f"Epoch {epoch + 1}: "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    print(f"==> Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    scheduler_roberta.step()

    if val_acc > best_val_acc_roberta:
        best_val_acc_roberta = val_acc
        counter = 0
        torch.save(roberta_model.state_dict(), 'roberta_large_model.pt')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

roberta_model.load_state_dict(torch.load('roberta_large_model.pt'))
roberta_model.eval()
test_predictions_roberta = []
test_true_labels_roberta = []
roberta_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader_roberta, desc="Predicting with RoBERTa"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = nn.functional.softmax(outputs.logits, dim=1)
        _, predicted = torch.max(probs, 1)

        roberta_probs.extend(probs[:, 1].cpu().numpy())
        test_predictions_roberta.extend(predicted.cpu().numpy())
        test_true_labels_roberta.extend(labels.cpu().numpy())

print("\n=== RoBERTa CLASSIFIER ===")
print("\nClassification Report:")
print(classification_report(test_true_labels_roberta, test_predictions_roberta, target_names=['Negative', 'Positive'], digits=4))

cm_roberta = confusion_matrix(test_true_labels_roberta, test_predictions_roberta)
print("\nConfusion Matrix:")
print(cm_roberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_roberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('RoBERTa Classifier - Confusion Matrix')
plt.show()

# Compute overall metrics for RoBERTa
precision_roberta, recall_roberta, f1_roberta, _ = precision_recall_fscore_support(
    test_true_labels_roberta, test_predictions_roberta, average='weighted'
)

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

print("\n=== RoBERTa EMBEDDINGS + XGBoost CLASSIFIER ===")
print("\nClassification Report:")
print(classification_report(test_labels_roberta, xgb_preds_roberta, target_names=['Negative', 'Positive'], digits=4))

cm_xgb_roberta = confusion_matrix(test_labels_roberta, xgb_preds_roberta)
print("\nConfusion Matrix:")
print(cm_xgb_roberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb_roberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('RoBERTa Embeddings + XGBoost - Confusion Matrix')
plt.show()

# Compute overall metrics for RoBERTa + XGBoost
precision_xgb_roberta, recall_xgb_roberta, f1_xgb_roberta, _ = precision_recall_fscore_support(
    test_labels_roberta, xgb_preds_roberta, average='weighted'
)

# === SECTION 8: ENSEMBLE — DeBERTa + RoBERTa ===
print("\n=== SECTION 8: ENSEMBLE — DeBERTa + RoBERTa ===")

assert np.array_equal(test_labels_deberta, test_labels_roberta)

deberta_probs_arr = np.array(deberta_probs)
roberta_probs_arr = np.array(roberta_probs)

avg_probs_dr = (deberta_probs_arr + roberta_probs_arr) / 2.0
avg_preds_dr = (avg_probs_dr > 0.5).astype(int)

print("\n=== ENSEMBLE (AVERAGING): DeBERTa + RoBERTa ===")
print("\nClassification Report:")
print(classification_report(test_labels_deberta, avg_preds_dr, target_names=['Negative', 'Positive'], digits=4))

cm_avg_dr = confusion_matrix(test_labels_deberta, avg_preds_dr)
print("\nConfusion Matrix:")
print(cm_avg_dr)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_avg_dr, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (AVERAGING) - DeBERTa + RoBERTa')
plt.show()

# Compute overall metrics for Ensemble (Averaging) DeBERTa + RoBERTa
precision_avg_dr, recall_avg_dr, f1_avg_dr, _ = precision_recall_fscore_support(
    test_labels_deberta, avg_preds_dr, average='weighted'
)

deberta_preds_majority_dr = (deberta_probs_arr > 0.5).astype(int)
roberta_preds_majority_dr = (roberta_probs_arr > 0.5).astype(int)

final_majority_preds_dr = []
for i in range(len(test_labels_deberta)):
    votes = [deberta_preds_majority_dr[i], roberta_preds_majority_dr[i]]
    final_pred = max(set(votes), key=votes.count)
    final_majority_preds_dr.append(final_pred)

print("\n=== ENSEMBLE (MAJORITY VOTING): DeBERTa + RoBERTa ===")
print("\nClassification Report:")
print(classification_report(test_labels_deberta, final_majority_preds_dr, target_names=['Negative', 'Positive'], digits=4))

cm_majority_dr = confusion_matrix(test_labels_deberta, final_majority_preds_dr)
print("\nConfusion Matrix:")
print(cm_majority_dr)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_majority_dr, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (MAJORITY VOTING) - DeBERTa + RoBERTa')
plt.show()

# Compute overall metrics for Ensemble (Majority Voting) DeBERTa + RoBERTa
precision_majority_dr, recall_majority_dr, f1_majority_dr, _ = precision_recall_fscore_support(
    test_labels_deberta, final_majority_preds_dr, average='weighted'
)

# === SECTION 9: AUC Computation and Saving ===
print("\n=== SECTION 9: AUC Computation and Metrics Summary ===")

def compute_auc(y_true, y_probs):
    auc = roc_auc_score(y_true, y_probs)
    return auc

def plot_roc(y_true, y_probs, model_name, save_path=None):
    # For binary classification, y_probs should be of shape (n_samples, 2)
    # y_probs[:, 0] for Negative class, y_probs[:, 1] for Positive class
    plt.figure(figsize=(8, 6))
    
    # Plot ROC for Negative class (class 0)
    fpr_neg, tpr_neg, _ = roc_curve(y_true, y_probs[:, 0], pos_label=0)
    roc_auc_neg = auc(fpr_neg, tpr_neg)
    plt.plot(fpr_neg, tpr_neg, label=f'Negative Class (AUC = {roc_auc_neg:.4f})')
    
    # Plot ROC for Positive class (class 1)
    fpr_pos, tpr_pos, _ = roc_curve(y_true, y_probs[:, 1], pos_label=1)
    roc_auc_pos = auc(fpr_pos, tpr_pos)
    plt.plot(fpr_pos, tpr_pos, label=f'Positive Class (AUC = {roc_auc_pos:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid()

    if save_path:
        plt.savefig(save_path)
    plt.show()

auc_summary_df = pd.DataFrame(columns=['Model', 'AUC', 'Precision', 'Recall', 'F1_Score'])

# For DeBERTa Softmax
test_probs_deberta_all = np.vstack([1 - np.array(test_probs_deberta), test_probs_deberta]).T  # [P(negative), P(positive)]
auc_deberta_softmax = compute_auc(test_labels_deberta, test_probs_deberta)
plot_roc(test_labels_deberta, test_probs_deberta_all, model_name='DeBERTa Softmax', save_path='deberta_softmax_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['DeBERTa Softmax', auc_deberta_softmax, precision_deberta, recall_deberta, f1_deberta]

# For DeBERTa + XGBoost
xgb_probs_deberta_all = xgb_clf_deberta.predict_proba(test_embeddings_deberta)  # Already [P(negative), P(positive)]
auc_xgb_deberta = compute_auc(test_labels_deberta, xgb_probs_deberta)
plot_roc(test_labels_deberta, xgb_probs_deberta_all, model_name='DeBERTa + XGBoost', save_path='deberta_xgb_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['DeBERTa + XGBoost', auc_xgb_deberta, precision_xgb_deberta, recall_xgb_deberta, f1_xgb_deberta]

# For DeBERTa + LightGBM
lgb_probs_deberta_all = lgb_clf_deberta.predict_proba(test_embeddings_deberta)  # Already [P(negative), P(positive)]
auc_lgb_deberta = compute_auc(test_labels_deberta, lgb_probs_deberta)
plot_roc(test_labels_deberta, lgb_probs_deberta_all, model_name='DeBERTa + LightGBM', save_path='deberta_lgb_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['DeBERTa + LightGBM', auc_lgb_deberta, precision_lgb_deberta, recall_lgb_deberta, f1_lgb_deberta]

# For RoBERTa Softmax
roberta_probs_all = np.vstack([1 - np.array(roberta_probs), roberta_probs]).T  # [P(negative), P(positive)]
auc_roberta_softmax = compute_auc(test_labels_roberta, roberta_probs)
plot_roc(test_labels_roberta, roberta_probs_all, model_name='RoBERTa Softmax', save_path='roberta_softmax_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['RoBERTa Softmax', auc_roberta_softmax, precision_roberta, recall_roberta, f1_roberta]

# For RoBERTa + XGBoost
xgb_probs_roberta_all = xgb_clf_roberta.predict_proba(test_embeddings_roberta)  # Already [P(negative), P(positive)]
auc_xgb_roberta = compute_auc(test_labels_roberta, xgb_probs_roberta)
plot_roc(test_labels_roberta, xgb_probs_roberta_all, model_name='RoBERTa + XGBoost', save_path='roberta_xgb_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['RoBERTa + XGBoost', auc_xgb_roberta, precision_xgb_roberta, recall_xgb_roberta, f1_xgb_roberta]

# For Ensemble DeBERTa_XGBoost + RoBERTa_XGBoost
avg_probs_xgb_dr = (xgb_probs_deberta + xgb_probs_roberta) / 2.0
avg_preds_xgb_dr = (avg_probs_xgb_dr > 0.5).astype(int)
avg_probs_xgb_dr_all = np.vstack([1 - avg_probs_xgb_dr, avg_probs_xgb_dr]).T  # [P(negative), P(positive)]

print("\n=== ENSEMBLE (AVERAGING): DeBERTa_XGBoost + RoBERTa_XGBoost ===")
print("\nClassification Report:")
print(classification_report(test_labels_deberta, avg_preds_xgb_dr, target_names=['Negative', 'Positive'], digits=4))

cm_avg_xgb_dr = confusion_matrix(test_labels_deberta, avg_preds_xgb_dr)
print("\nConfusion Matrix:")
print(cm_avg_xgb_dr)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_avg_xgb_dr, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (AVERAGING) - DeBERTa_XGBoost + RoBERTa_XGBoost')
plt.show()

# Compute overall metrics for Ensemble (Averaging) DeBERTa_XGBoost + RoBERTa_XGBoost
precision_avg_xgb_dr, recall_avg_xgb_dr, f1_avg_xgb_dr, _ = precision_recall_fscore_support(
    test_labels_deberta, avg_preds_xgb_dr, average='weighted'
)

auc_xgb_dr_ensemble = compute_auc(test_labels_deberta, avg_probs_xgb_dr)
plot_roc(test_labels_deberta, avg_probs_xgb_dr_all, model_name='ENSEMBLE - DeBERTa_XGBoost + RoBERTa_XGBoost', save_path='ensemble_xgb_dr_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['ENSEMBLE - DeBERTa_XGBoost + RoBERTa_XGBoost', auc_xgb_dr_ensemble, precision_avg_xgb_dr, recall_avg_xgb_dr, f1_avg_xgb_dr]

# Add ensemble metrics for DeBERTa + XGBoost + LightGBM (Averaging and Majority Voting)
avg_probs_deberta_all = np.vstack([1 - avg_probs_deberta, avg_probs_deberta]).T  # [P(negative), P(positive)]
auc_avg_deberta = compute_auc(test_labels_deberta, avg_probs_deberta)
auc_summary_df.loc[len(auc_summary_df)] = ['ENSEMBLE - DeBERTa + XGBoost + LightGBM (Averaging)', auc_avg_deberta, precision_avg_deberta, recall_avg_deberta, f1_avg_deberta]

auc_majority_deberta = compute_auc(test_labels_deberta, avg_probs_deberta)  # AUC uses probabilities, same as averaging
auc_summary_df.loc[len(auc_summary_df)] = ['ENSEMBLE - DeBERTa + XGBoost + LightGBM (Majority Voting)', auc_majority_deberta, precision_majority_deberta, recall_majority_deberta, f1_majority_deberta]

# Add ensemble metrics for DeBERTa + RoBERTa (Averaging and Majority Voting)
avg_probs_dr_all = np.vstack([1 - avg_probs_dr, avg_probs_dr]).T  # [P(negative), P(positive)]
auc_avg_dr = compute_auc(test_labels_deberta, avg_probs_dr)
auc_summary_df.loc[len(auc_summary_df)] = ['ENSEMBLE - DeBERTa + RoBERTa (Averaging)', auc_avg_dr, precision_avg_dr, recall_avg_dr, f1_avg_dr]

auc_majority_dr = compute_auc(test_labels_deberta, avg_probs_dr)  # AUC uses probabilities, same as averaging
auc_summary_df.loc[len(auc_summary_df)] = ['ENSEMBLE - DeBERTa + RoBERTa (Majority Voting)', auc_majority_dr, precision_majority_dr, recall_majority_dr, f1_majority_dr]

# Print and save summary
print("\n=== Overall Metrics Summary ===")
print(auc_summary_df.round(4))  # Round to 4 decimal places for readability
auc_summary_df.to_csv('metrics_summary_all_models.csv', index=False)
print("\n✅ Metrics summary saved to 'metrics_summary_all_models.csv'")
