# main.py
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_API_TOKEN = "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"  # paste your token here
MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt')

    response = requests.post(MODEL_URL,
                             headers=HEADERS,
                             json={"inputs": prompt})

    try:
        result = response.json()
        if isinstance(result, list):
            text = result[0]["generated_text"]
        else:
            text = result.get("generated_text") or str(result)
        return jsonify({"model": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
