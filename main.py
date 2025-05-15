from flask import Flask, request, jsonify
import requests
import json
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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
        logger.debug("Received request with data: %s", request.data)
        
        # Get JSON data
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "Request body must be JSON"}), 400
        
        # Fix typo in 'prompt' to 'prompt'
        prompt = data.get('prompt', data.get('prompt', ''))
        if not prompt:
            logger.error("No prompt provided")
            return jsonify({"error": "No prompt provided"}), 400
        
        logger.debug("Sending prompt to AI: %s", prompt)
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        # Make API request with timeout
        hf_response = requests.post(API_URL, headers=HEADERS, json=payload)
        hf_response.raise_for_status()
        
        logger.debug("Received response from AI: %s", hf_response.text)
        
        # Parse AI response
        ai_data = hf_response.json()
        ai_message = ai_data['choices'][0]['message']['content']
        
        # Clean response - find first { and last }
        start_idx = ai_message.find('{')
        end_idx = ai_message.rfind('}') + 1
        
        if start_idx == -1 or end_idx == -1:
            logger.error("No JSON found in AI response: %s", ai_message)
            return jsonify({"error": "AI response didn't contain valid JSON"}), 500
            
        json_str = ai_message[start_idx:end_idx]
        
        # Validate JSON
        try:
            parsed = json.loads(json_str)
            logger.debug("Successfully parsed AI response")
            return jsonify(parsed), 200
        except json.JSONDecodeError as e:
            logger.error("Failed to parse AI response: %s", str(e))
            logger.error("Original response: %s", ai_message)
            logger.error("Extracted JSON: %s", json_str)
            return jsonify({
                "error": "Invalid JSON from AI",
                "ai_response": ai_message,
                "extracted_json": json_str
            }), 500
            
    except requests.exceptions.RequestException as e:
        logger.error("Request to Hugging Face failed: %s", str(e))
        return jsonify({
            "error": "Failed to communicate with AI service",
            "details": str(e)
        }), 503
    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
