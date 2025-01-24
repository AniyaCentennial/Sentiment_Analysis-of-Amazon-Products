import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch
import numpy as np
import evaluate  # Use the new evaluate library for metrics

# Load dataset
file_path = "D:/Projects_DataAnalyst/Sentiment_Analysis/data/processed_data.csv"
data = pd.read_csv(file_path)

# Check for necessary columns
if 'cleaned_review' not in data.columns or 'sentiment_label' not in data.columns:
    raise ValueError("Dataset must contain 'cleaned_review' and 'sentiment_label' columns.")

# Encode sentiment labels
label_encoder = LabelEncoder()
data['sentiment_label'] = label_encoder.fit_transform(data['sentiment_label'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_review'], data['sentiment_label'], test_size=0.2, random_state=42, stratify=data['sentiment_label']
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

train_dataset = SentimentDataset(X_train, y_train, tokenizer)
test_dataset = SentimentDataset(X_test, y_test, tokenizer)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Metric function
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="D:/Projects_DataAnalyst/Sentiment_Analysis/model/bert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="D:/Projects_DataAnalyst/Sentiment_Analysis/logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and save model
trainer.train()
trainer.save_model("D:/Projects_DataAnalyst/Sentiment_Analysis/model/bert")
tokenizer.save_pretrained("D:/Projects_DataAnalyst/Sentiment_Analysis/model/bert")
