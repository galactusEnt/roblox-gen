from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
HF_API_TOKEN = "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"
MODEL_NAME = "accounts/fireworks/models/deepseek-r1"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}


@app.route('/')
def home():
    return "Server is up and running."


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": MODEL_NAME
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch from Hugging Face", "details": response.text}), 500

        result = response.json()
        message = result["choices"][0]["message"]["content"]

        return jsonify({"model": message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
