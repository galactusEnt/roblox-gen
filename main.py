from flask import Flask, request, jsonify
import requests
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
HF_API_TOKEN = "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"
MODEL_NAME = "accounts/fireworks/models/deepseek-r1"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """You are a 3D model generator for Roblox. 
Output ONLY a JSON array where each element represents a part with:
- "Shape" (string: "Block", "Ball", or "Cylinder")
- "Color" (array of 3 integers 0-255 [R,G,B])
- "Size" (array of 3 numbers [width,height,depth])
- "Position" (array of 3 numbers [X,Y,Z])
- "Orientation" (array of 3 numbers [X-rot,Y-rot,Z-rot])

Example output:
[
    {
        "Shape": "Block",
        "Color": [255, 0, 0],
        "Size": [2, 1, 2],
        "Position": [0, 0.5, 0],
        "Orientation": [0, 0, 0]
    }
]

Do not include any other text or explanations."""

@app.route('/generate', methods=['POST'])
def generate_model():
    try:
        logger.info("Received generation request")
        
        # Get prompt from request
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "Request body must be JSON"}), 400
            
        prompt = data.get('prompt')
        if not prompt:
            logger.error("No prompt provided")
            return jsonify({"error": "Prompt is required"}), 400

        logger.info(f"Generating model with prompt: {prompt}")
        
        # Call Hugging Face API
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        hf_response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        hf_response.raise_for_status()
        
        # Process AI response
        ai_response = hf_response.json()
        content = ai_response['choices'][0]['message']['content']
        
        # Validate the response is proper JSON
        try:
            # Try to parse the content to validate it's JSON
            parsed = json.loads(content)
            logger.info("Successfully generated model")
            return content, 200, {'Content-Type': 'application/json'}
        except ValueError as e:
            logger.error(f"AI returned invalid JSON: {content}")
            return jsonify({"error": "AI returned invalid format", "content": content}), 500
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Hugging Face API error: {str(e)}")
        return jsonify({"error": f"Model generation service unavailable: {str(e)}"}), 503
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
