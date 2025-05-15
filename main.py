from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_API_TOKEN = "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"
MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


@app.route('/')
def home():
    return "Hugging Face AI server is up!"


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        response = requests.post(
            MODEL_URL,
            headers=HEADERS,
            json={"inputs": prompt}
        )

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace error {response.status_code}", "details": response.text}), 500

        result = response.json()

        # Extract generated text
        if isinstance(result, list) and "generated_text" in result[0]:
            full_text = result[0]["generated_text"]
            # Return just the part after the prompt
            generated_only = full_text[len(prompt):].strip()
            return jsonify({"model": generated_only})
        else:
            return jsonify({"error": "Unexpected response format", "raw": result}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
