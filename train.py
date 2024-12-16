from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import torch
import pandas as pd

# Define the data for training
train_data = pd.DataFrame({
    "text": [
        "Create a table for users with name, email, and phone",
        "Add a column for address in the users table",
        "Generate schema for customers with fields email, date of birth, and phone",
        "Create schema for payments with amount, currency, and date",
        "Alter table to add a phone number column"
    ],
    "labels": [0, 0, 1, 1, 0]  # Schema correction or schema creation
})

# Tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Tokenize the data
train_encodings = tokenizer(
    list(train_data["text"]),
    truncation=True,
    padding="max_length",
    max_length=128,
    return_tensors="pt"
)

train_labels = torch.tensor(train_data["labels"].values)

# Model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./schema_model",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2
)

# Create the trainer
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=torch.utils.data.TensorDataset(train_encodings["input_ids"], train_labels),
)

trainer.train()
