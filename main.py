from flask import Flask, request, jsonify
import requests
import json
import logging
import re

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

SYSTEM_PROMPT = """You MUST output ONLY a JSON array of parts in EXACTLY this format:
[
    {
        "Shape": "Block|Ball|Cylinder",
        "Color": [R,G,B],  // 0-255 values
        "Size": [width,height,depth],
        "Position": [X,Y,Z],
        "Orientation": [X-rot,Y-rot,Z-rot]  // degrees
    }
    // More parts as needed
]

RULES:
1. NEVER include any thoughts, explanations, or markdown
2. ONLY output the JSON array
3. ALWAYS use valid JSON syntax
4. Use only these shapes: Block, Ball, Cylinder
5. Color must be [R,G,B] with values 0-255
6. All numbers must be valid (no NaN or infinity)

Example for a red flower with green stem:
[
    {"Shape":"Cylinder","Color":[255,0,0],"Size":[1,0.2,1],"Position":[0,3,0],"Orientation":[0,0,90]},
    {"Shape":"Cylinder","Color":[0,255,0],"Size":[0.5,6,0.5],"Position":[0,3,0],"Orientation":[0,0,0]}
]"""

@app.route('/generate', methods=['POST'])
def generate_model():
    try:
        logger.debug("Received request with data: %s", request.data)
        
        # Get JSON data
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "Request body must be JSON"}), 400
        
        prompt = data.get('prompt', '')
        if not prompt:
            logger.error("No prompt provided")
            return jsonify({"error": "No prompt provided"}), 400
        
        logger.debug("Sending prompt to AI: %s", prompt)
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Create model for: {prompt}"}
            ],
            "temperature": 0.1,  # Lower temperature for more consistent output
            "max_tokens": 1000
        }
        
        # Make API request with timeout
        hf_response = requests.post(API_URL, headers=HEADERS, json=payload)
        hf_response.raise_for_status()
        
        logger.debug("Received response from AI: %s", hf_response.text)
        
        # Parse AI response
        ai_data = hf_response.json()
        ai_message = ai_data['choices'][0]['message']['content']
        
        # Clean response - remove any non-JSON text
        json_match = re.search(r'\[.*\]', ai_message, re.DOTALL)
        if not json_match:
            logger.error("No JSON array found in response: %s", ai_message)
            return jsonify({
                "error": "AI response format invalid",
                "received": ai_message,
                "expected": "JSON array of parts"
            }), 500
            
        json_str = json_match.group(0)
        
        # Validate JSON
        try:
            parts = json.loads(json_str)
            if not isinstance(parts, list):
                raise ValueError("Top-level element must be an array")
                
            # Validate each part
            for part in parts:
                if not all(key in part for key in ["Shape", "Color", "Size", "Position", "Orientation"]):
                    raise ValueError("Missing required part fields")
                
            logger.debug("Successfully parsed %d parts", len(parts))
            return jsonify(parts), 200
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Invalid JSON: %s\nResponse was: %s", str(e), json_str)
            return jsonify({
                "error": "Invalid part data",
                "details": str(e),
                "received": json_str
            }), 500
            
    except requests.exceptions.RequestException as e:
        logger.error("Request to Hugging Face failed: %s", str(e))
        return jsonify({
            "error": "AI service unavailable",
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
