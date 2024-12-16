import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the saved model and tokenizer
model_path = "./schema_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example unseen test inputs
test_texts = [
    "CREATE TABLE employees (emp_id INT, emp_name VARCHAR);",
    "DELETE FROM employees WHERE emp_id = 5;",
    "ALTER TABLE employees ADD COLUMN salary FLOAT;",
]

# Tokenize inputs
inputs = tokenizer(test_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Move inputs to the device
inputs = {key: val.to(device) for key, val in inputs.items()}

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Map predictions to labels
label_map = {
    0: "CREATE",
    1: "ALTER",
    2: "DROP",
    3: "SELECT",
    4: "INSERT"
}
predicted_labels = [label_map[pred.item()] for pred in predictions]

# Output predictions
for text, label in zip(test_texts, predicted_labels):
    print(f"Input: {text}\nPredicted Label: {label}\n")
