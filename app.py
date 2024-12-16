from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "./schema_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Serve the index.html when visiting the root endpoint
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Ensure 'prompt' exists in request
    if "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field in the request"}), 400

    # Tokenize user input
    input_text = data["prompt"]
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Map prediction to schema response
    if predicted_class == 0:
        response_schema = {
            "task": "Schema Correction",
            "schema": "ALTER TABLE users ADD COLUMN address TEXT;"
        }
    elif predicted_class == 1:
        response_schema = {
            "task": "Database Schema Generation",
            "schema": "CREATE TABLE customers (email TEXT, date_of_birth DATE, phone VARCHAR);"
        }
    else:
        response_schema = {
            "task": "Unknown",
            "schema": "No schema could be generated."
        }

    return jsonify({
        "input_prompt": input_text,
        "response": response_schema
    })


# Run the server
if __name__ == "__main__":
    app.run(debug=True)
