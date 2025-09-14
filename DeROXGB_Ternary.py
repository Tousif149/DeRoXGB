# === SECTION 0 — SETUP ===
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
    classification_report, roc_curve, auc
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
# Preprocessing functions
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
        # Expand contractions first
    text = contractions.fix(text)
    text = remove_emojis(text)  # remove emojis 
    text = re.sub(r'http\S+', '', text)              # remove URLs
    text = re.sub(r'@\w+', '', text)                 # remove mentions
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)       # remove special characters
    text = text.lower().strip()                      # lowercase & strip
    return text

# Add label mapping function
def map_labels(labels):
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    return [label_map.get(l.lower(), 0) if isinstance(l, str) else l for l in labels]

# Load data and detect label column
data = pd.read_csv('/kaggle/input/eda-augment/augmented_tweets.csv')
data['text'] = data['text'].apply(preprocess_text)
data = data[data['text'] != ""]  # filter out empty rows

# Automatically detect label column (try 'airline_sentiment' first, then 'label')
label_column = 'airline_sentiment' if 'airline_sentiment' in data.columns else 'label'
print(f"Using label column: {label_column}")

# Map string labels to integers
data['label'] = map_labels(data[label_column].values)

# Split
train_val, test = train_test_split(data, test_size=0.15, random_state=SEED, stratify=data['label'])
train, val = train_test_split(train_val, test_size=0.17647, random_state=SEED, stratify=train_val['label'])

# Prepare arrays with mapped labels
train_texts = train['text'].values
train_labels = train['label'].values
val_texts   = val['text'].values
val_labels  = val['label'].values
test_texts  = test['text'].values
test_labels = test['label'].values

# No need to map labels — already 0, 1, 2
print(data['label'].value_counts(dropna=False))  # optional sanity check

# === SECTION 2 — HYPERPARAMETERS ===
max_length = 128
batch_size = 32
epochs = 7
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
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
            all_embeddings.append(cls_embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels

# === SECTION 3 — DeBERTa-v3-large CLASSIFIER ===

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# === TRAIN EPOCH ===
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

# === Tokenizer ===
tokenizer_deberta = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

# === Dataset class ===
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

# === Dataloaders ===
train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)
test_dataset = SentimentDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# === Model ===
deberta_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-large', num_labels=3)

if num_gpus > 1:
    print(f"Using {num_gpus} GPUs for PyTorch model")
    deberta_model = nn.DataParallel(deberta_model)
deberta_model.to(device)

# === Optimizer, scheduler, loss, scaler ===
optimizer = optim.NAdam(deberta_model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = FocalLoss(alpha=1.0, gamma=1.5)
scaler = GradScaler()

# === Training loop ===
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

    # Save best model
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

# === Save training/val stats ===
df_stats_deberta = pd.DataFrame({
    'epoch': list(range(1, len(train_losses_deberta)+1)),
    'train_loss': train_losses_deberta,
    'val_loss': val_losses_deberta,
    'train_accuracy': train_accuracies_deberta,
    'val_accuracy': val_accuracies_deberta
})

df_stats_deberta.to_csv('training_validation_stats_deberta.csv', index=False)
print("Training and validation stats saved to 'training_validation_stats_deberta.csv'.")

# === Final message ===
print("\n✅ DeBERTa fine-tuning completed!")

# === DEBERTA CLASSIFICATION REPORT ===

# Load best model
deberta_model.load_state_dict(torch.load('deberta_v3_large_model.pt'))
deberta_model.eval()

# Predict on test set
test_predictions_deberta = []
test_true_labels_deberta = []

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

# === Report ===
print("\n=== DeBERTa CLASSIFIER ===")
print("\nClassification Report:")
report = classification_report(test_true_labels_deberta, test_predictions_deberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_true_labels_deberta, test_predictions_deberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
print(f"F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

cm_deberta = confusion_matrix(test_true_labels_deberta, test_predictions_deberta)
print("\nConfusion Matrix:")
print(cm_deberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_deberta, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('DeBERTa Classifier - Confusion Matrix')
plt.show()

# === SECTION 4: DeBERTa EMBEDDINGS → XGBoost ===
print("\n=== SECTION 4: DeBERTa EMBEDDINGS → XGBoost ===")

# Use fine-tuned encoder
deberta_model.load_state_dict(torch.load('deberta_v3_large_model.pt'))

if isinstance(deberta_model, nn.DataParallel):
    deberta_encoder = deberta_model.module.deberta
else:
    deberta_encoder = deberta_model.deberta

deberta_encoder.to(device)

# Extract embeddings
train_embeddings_deberta, train_labels_deberta = extract_embeddings(train_loader, deberta_encoder)
test_embeddings_deberta, test_labels_deberta = extract_embeddings(test_loader, deberta_encoder)

# XGBoost on DeBERTa embeddings
xgb_clf_deberta = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
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

# Report XGBoost results
print("\n=== DeBERTa EMBEDDINGS + XGBoost CLASSIFIER ===")
print("\nClassification Report:")
report = classification_report(test_labels_deberta, xgb_preds_deberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_labels_deberta, xgb_preds_deberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
print(f"F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

cm_xgb_deberta = confusion_matrix(test_labels_deberta, xgb_preds_deberta)
print("\nConfusion Matrix:")
print(cm_xgb_deberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb_deberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('DeBERTa Embeddings + XGBoost - Confusion Matrix')
plt.show()

# === SECTION 5: DeBERTa EMBEDDINGS → LightGBM ===
print("\n=== SECTION 5: DeBERTa EMBEDDINGS → LightGBM ===")

# Import callbacks
from lightgbm import early_stopping, log_evaluation

# LightGBM classifier
lgb_clf_deberta = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    learning_rate=0.1,
    n_estimators=100,
    num_leaves=64,
    max_depth=-1,
    random_state=SEED,
    n_jobs=-1
)

# Fit with callbacks (early stopping + logging)
eval_set = [(train_embeddings_deberta, train_labels_deberta), (test_embeddings_deberta, test_labels_deberta)]
eval_names = ['train', 'valid']

lgb_clf_deberta.fit(
    train_embeddings_deberta, train_labels_deberta,
    eval_set=eval_set,
    eval_names=eval_names,
    eval_metric='multi_logloss',
    callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=20)
    ]
)

# Predict
lgb_preds_probs_deberta = lgb_clf_deberta.predict_proba(test_embeddings_deberta)
lgb_preds_deberta = np.argmax(lgb_preds_probs_deberta, axis=1)

# Report
print("\n=== DeBERTa EMBEDDINGS + LightGBM CLASSIFIER ===")
print("\nClassification Report:")
report = classification_report(test_labels_deberta, lgb_preds_deberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_labels_deberta, lgb_preds_deberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
print(f"F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

cm_lgb_deberta = confusion_matrix(test_labels_deberta, lgb_preds_deberta)
print("\nConfusion Matrix:")
print(cm_lgb_deberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_lgb_deberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('DeBERTa Embeddings + LightGBM - Confusion Matrix')
plt.show()

# Plot training + validation logloss
evals_result = lgb_clf_deberta.evals_result_

plt.figure(figsize=(8, 6))
plt.plot(evals_result['train']['multi_logloss'], label='Train Multi Logloss')
plt.plot(evals_result['valid']['multi_logloss'], label='Valid Multi Logloss')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('LightGBM - Training and Validation Log Loss (DeBERTa Embeddings)')
plt.legend()
plt.grid()
plt.show()

# === SECTION 6: ENSEMBLE — DeBERTa_softmax + XGBoost + LightGBM ===
print("\n=== SECTION 6: ENSEMBLE — DeBERTa_softmax + XGBoost + LightGBM ===")

# Prepare probabilities
# First need to recompute DeBERTa softmax probs on test set
print("Recomputing DeBERTa softmax probabilities...")
deberta_model.eval()
deberta_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting with DeBERTa"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = deberta_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = nn.functional.softmax(outputs.logits, dim=1)

        deberta_probs.extend(probs.cpu().numpy())

deberta_probs = np.array(deberta_probs)
xgb_probs_deberta = xgb_clf_deberta.predict_proba(test_embeddings_deberta)
lgb_probs_deberta = lgb_preds_probs_deberta

# Averaging ensemble
avg_probs_deberta = (deberta_probs + xgb_probs_deberta + lgb_probs_deberta) / 3.0
avg_preds_deberta = np.argmax(avg_probs_deberta, axis=1)

print("\n=== ENSEMBLE (AVERAGING): DeBERTa_softmax + XGBoost + LightGBM ===")
print("\nClassification Report:")
report = classification_report(test_labels_deberta, avg_preds_deberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_labels_deberta, avg_preds_deberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
print(f"F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

cm_avg_deberta = confusion_matrix(test_labels_deberta, avg_preds_deberta)
print("\nConfusion Matrix:")
print(cm_avg_deberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_avg_deberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (AVERAGING) - DeBERTa_softmax + XGBoost + LightGBM')
plt.show()

# Majority voting
deberta_preds_majority = np.argmax(deberta_probs, axis=1)
xgb_preds_majority = np.argmax(xgb_probs_deberta, axis=1)
lgb_preds_majority = np.argmax(lgb_probs_deberta, axis=1)

final_majority_preds_deberta = []
for i in range(len(test_labels_deberta)):
    votes = [deberta_preds_majority[i], xgb_preds_majority[i], lgb_preds_majority[i]]
    final_pred = max(set(votes), key=votes.count)
    final_majority_preds_deberta.append(final_pred)

print("\n=== ENSEMBLE (MAJORITY VOTING): DeBERTa_softmax + XGBoost + LightGBM ===")
print("\nClassification Report:")
report = classification_report(test_labels_deberta, final_majority_preds_deberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_labels_deberta, final_majority_preds_deberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
print(f"F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

cm_majority_deberta = confusion_matrix(test_labels_deberta, final_majority_preds_deberta)
print("\nConfusion Matrix:")
print(cm_majority_deberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_majority_deberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (MAJORITY VOTING) - DeBERTa_softmax + XGBoost + LightGBM')
plt.show()

# === SECTION 7: RoBERTa-Large CLASSIFIER + EMBEDDINGS + XGBoost ===
print("\n=== SECTION 7: RoBERTa-Large CLASSIFIER + XGBoost ===")

# Tokenizer
tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-large')

# Dataset class — re-use but with tokenizer_roberta
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

# Dataloaders
train_dataset_roberta = SentimentDatasetRoberta(train_texts, train_labels)
val_dataset_roberta = SentimentDatasetRoberta(val_texts, val_labels)
test_dataset_roberta = SentimentDatasetRoberta(test_texts, test_labels)

train_loader_roberta = DataLoader(train_dataset_roberta, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader_roberta = DataLoader(val_dataset_roberta, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader_roberta = DataLoader(test_dataset_roberta, batch_size=batch_size, shuffle=False, num_workers=4)

# Model
roberta_model = AutoModelForSequenceClassification.from_pretrained('roberta-large', num_labels=3)
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs for PyTorch model")
    roberta_model = nn.DataParallel(roberta_model)
roberta_model.to(device)

# Optimizer, scheduler, loss, scaler
optimizer_roberta = optim.NAdam(roberta_model.parameters(), lr=learning_rate)
scheduler_roberta = optim.lr_scheduler.CosineAnnealingLR(optimizer_roberta, T_max=epochs)
criterion_roberta = FocalLoss(alpha=1.0, gamma=1.5)
scaler_roberta = GradScaler()

# Training loop (same as previous)
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

# Load best model
roberta_model.load_state_dict(torch.load('roberta_large_model.pt'))

# Evaluate RoBERTa classifier on test set
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

        roberta_probs.extend(probs.cpu().numpy())
        test_predictions_roberta.extend(predicted.cpu().numpy())
        test_true_labels_roberta.extend(labels.cpu().numpy())

# Report RoBERTa classifier results
print("\n=== RoBERTa CLASSIFIER ===")
print("\nClassification Report:")
report = classification_report(test_true_labels_roberta, test_predictions_roberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_true_labels_roberta, test_predictions_roberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
print(f"F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

cm_roberta = confusion_matrix(test_true_labels_roberta, test_predictions_roberta)
print("\nConfusion Matrix:")
print(cm_roberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_roberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('RoBERTa Classifier - Confusion Matrix')
plt.show()

# === RoBERTa embeddings → XGBoost ===
print("\nExtracting RoBERTa embeddings for XGBoost...")

if isinstance(roberta_model, nn.DataParallel):
    roberta_encoder = roberta_model.module.roberta
else:
    roberta_encoder = roberta_model.roberta

roberta_encoder.to(device)

train_embeddings_roberta, train_labels_roberta = extract_embeddings(train_loader_roberta, roberta_encoder)
test_embeddings_roberta, test_labels_roberta = extract_embeddings(test_loader_roberta, roberta_encoder)

# XGBoost on RoBERTa embeddings
xgb_clf_roberta = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
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

# Report XGBoost results
print("\n=== RoBERTa EMBEDDINGS + XGBoost CLASSIFIER ===")
print("\nClassification Report:")
report = classification_report(test_labels_roberta, xgb_preds_roberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_labels_roberta, xgb_preds_roberta, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
print(f"F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

cm_xgb_roberta = confusion_matrix(test_labels_roberta, xgb_preds_roberta)
print("\nConfusion Matrix:")
print(cm_xgb_roberta)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb_roberta, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('RoBERTa Embeddings + XGBoost - Confusion Matrix')
plt.show()

# === SECTION 8: ENSEMBLE — DeBERTa + RoBERTa ===
print("\n=== SECTION 8: ENSEMBLE — DeBERTa + RoBERTa ===")

# Align targets
assert np.array_equal(test_labels_deberta, test_labels_roberta)

# Prepare for averaging
deberta_probs_arr = np.array(deberta_probs)
roberta_probs_arr = np.array(roberta_probs)

# Averaging ensemble
avg_probs_dr = (deberta_probs_arr + roberta_probs_arr) / 2.0
avg_preds_dr = np.argmax(avg_probs_dr, axis=1)

print("\n=== ENSEMBLE (AVERAGING): DeBERTa + RoBERTa ===")
print("\nClassification Report:")
report = classification_report(test_labels_deberta, avg_preds_dr, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_labels_deberta, avg_preds_dr, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
print(f"F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

cm_avg_dr = confusion_matrix(test_labels_deberta, avg_preds_dr)
print("\nConfusion Matrix:")
print(cm_avg_dr)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_avg_dr, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (AVERAGING) - DeBERTa + RoBERTa')
plt.show()

# Majority voting
deberta_preds_majority_dr = np.argmax(deberta_probs_arr, axis=1)
roberta_preds_majority_dr = np.argmax(roberta_probs_arr, axis=1)

final_majority_preds_dr = []
for i in range(len(test_labels_deberta)):
    votes = [deberta_preds_majority_dr[i], roberta_preds_majority_dr[i]]
    final_pred = max(set(votes), key=votes.count)
    final_majority_preds_dr.append(final_pred)

print("\n=== ENSEMBLE (MAJORITY VOTING): DeBERTa + RoBERTa ===")
print("\nClassification Report:")
report = classification_report(test_labels_deberta, final_majority_preds_dr, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_labels_deberta, final_majority_preds_dr, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
print(f"F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

cm_majority_dr = confusion_matrix(test_labels_deberta, final_majority_preds_dr)
print("\nConfusion Matrix:")
print(cm_majority_dr)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_majority_dr, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (MAJORITY VOTING) - DeBERTa + RoBERTa')
plt.show()

# === SECTION 9: AUC Computation and Saving ===

# Helper functions
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def compute_auc(y_true, y_probs, num_classes=3):
    aucs = []
    y_true_bin = np.eye(num_classes)[y_true]
    for i in range(num_classes):
        auc_i = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        aucs.append(auc_i)
    macro_auc = np.mean(aucs)
    return aucs, macro_auc

def plot_roc(y_true, y_probs, model_name, save_path=None):
    num_classes = y_probs.shape[1]
    y_true_bin = np.eye(num_classes)[y_true]

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.4f})')

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

# Initialize AUC summary dataframe
auc_summary_df = pd.DataFrame(columns=['Model', 'Macro AUC', 'Class 0 AUC', 'Class 1 AUC', 'Class 2 AUC'])

# === DeBERTa Softmax AUC ===
aucs_deberta_softmax, macro_auc_deberta_softmax = compute_auc(test_labels_deberta, deberta_probs)
plot_roc(test_labels_deberta, deberta_probs, model_name='DeBERTa Softmax', save_path='deberta_softmax_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['DeBERTa Softmax', macro_auc_deberta_softmax] + aucs_deberta_softmax

# === DeBERTa + XGBoost AUC ===
aucs_xgb_deberta, macro_auc_xgb_deberta = compute_auc(test_labels_deberta, xgb_probs_deberta)
plot_roc(test_labels_deberta, xgb_probs_deberta, model_name='DeBERTa + XGBoost', save_path='deberta_xgb_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['DeBERTa + XGBoost', macro_auc_xgb_deberta] + aucs_xgb_deberta

# === DeBERTa + LightGBM AUC ===
aucs_lgb_deberta, macro_auc_lgb_deberta = compute_auc(test_labels_deberta, lgb_preds_probs_deberta)
plot_roc(test_labels_deberta, lgb_preds_probs_deberta, model_name='DeBERTa + LightGBM', save_path='deberta_lgb_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['DeBERTa + LightGBM', macro_auc_lgb_deberta] + aucs_lgb_deberta

# === RoBERTa Softmax AUC ===
roberta_probs = np.array(roberta_probs)  # FIXED LINE - convert to array
aucs_roberta_softmax, macro_auc_roberta_softmax = compute_auc(test_labels_roberta, roberta_probs)
plot_roc(test_labels_roberta, roberta_probs, model_name='RoBERTa Softmax', save_path='roberta_softmax_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['RoBERTa Softmax', macro_auc_roberta_softmax] + aucs_roberta_softmax

# === RoBERTa + XGBoost AUC ===
xgb_probs_roberta = xgb_clf_roberta.predict_proba(test_embeddings_roberta)
aucs_xgb_roberta, macro_auc_xgb_roberta = compute_auc(test_labels_roberta, xgb_probs_roberta)
plot_roc(test_labels_roberta, xgb_probs_roberta, model_name='RoBERTa + XGBoost', save_path='roberta_xgb_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['RoBERTa + XGBoost', macro_auc_xgb_roberta] + aucs_xgb_roberta

# === ENSEMBLE DeBERTa_XGBoost + RoBERTa_XGBoost ===
avg_probs_xgb_dr = (xgb_probs_deberta + xgb_probs_roberta) / 2.0
avg_preds_xgb_dr = np.argmax(avg_probs_xgb_dr, axis=1)

print("\n=== ENSEMBLE (AVERAGING): DeBERTa_XGBoost + RoBERTa_XGBoost ===")
print("\nClassification Report:")
report = classification_report(test_labels_deberta, avg_preds_xgb_dr, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
print(classification_report(test_labels_deberta, avg_preds_xgb_dr, target_names=['Negative', 'Neutral', 'Positive'], digits=4))
print(f"\nOverall Metrics:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Precision (macro): {report['macro avg']['precision']:.4f}")
print(f"Recall (macro): {report['macro avg']['recall']:.4f}")
print(f"F1-Score (macro): {report['macro avg']['f1-score']:.4f}")

cm_avg_xgb_dr = confusion_matrix(test_labels_deberta, avg_preds_xgb_dr)
print("\nConfusion Matrix:")
print(cm_avg_xgb_dr)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_avg_xgb_dr, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ENSEMBLE (AVERAGING) - DeBERTa_XGBoost + RoBERTa_XGBoost')
plt.show()

# Compute AUC for ensemble
aucs_xgb_dr_ensemble, macro_auc_xgb_dr_ensemble = compute_auc(test_labels_deberta, avg_probs_xgb_dr)
plot_roc(test_labels_deberta, avg_probs_xgb_dr, model_name='ENSEMBLE - DeBERTa_XGBoost + RoBERTa_XGBoost', save_path='ensemble_xgb_dr_roc.png')
auc_summary_df.loc[len(auc_summary_df)] = ['ENSEMBLE - DeBERTa_XGBoost + RoBERTa_XGBoost', macro_auc_xgb_dr_ensemble] + aucs_xgb_dr_ensemble

# === Save AUC Summary CSV ===
auc_summary_df.to_csv('auc_summary_all_models.csv', index=False)
print("\n✅ AUC summary saved to 'auc_summary_all_models.csv'")
