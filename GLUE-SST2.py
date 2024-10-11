import torch
from torch.utils.data import DataLoader
from transformers import ConvBertTokenizer, ConvBertForSequenceClassification, AdamW
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm 

# Load Stanford Sentiment Treebank dataset from Hugging Face datasets
dataset = load_dataset("glue", "sst2")
train_dataset = dataset['train']
test_dataset = dataset['validation']

print(dataset["train"].column_names)

# Load ConvBERT Tokenizer
model_name = "YituTech/conv-bert-base"  # Publicly available ConvBERT model from Hugging Face
tokenizer = ConvBertTokenizer.from_pretrained(model_name)

# Tokenize the data
def tokenize(batch):
    return tokenizer(batch['sentence'], padding=True, truncation=True)

train_dataset = train_dataset.map(lambda x: tokenize(x), batched=True)
test_dataset = test_dataset.map(lambda x: tokenize(x), batched=True)

# Convert dataset to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


# Custom Collate Function for Sentence Padding

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True)
    labels = torch.tensor(labels)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'label': labels
    }

# Load Pre-trained ConvBERT model from Hugging Face in PyTorch
model = ConvBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Setup the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Prepare DataLoader with the custom collate function
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# Fine-tuning on the CoLA dataset
model.train()
for epoch in range(3):  # train for 3 epochs
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# Model Evaluation
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")