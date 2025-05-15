from flask import Flask, request, jsonify
import requests
import re

app = Flask(__name__)

API_URL = "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
HF_API_TOKEN = "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"
MODEL_NAME = "accounts/fireworks/models/deepseek-r1"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """You are a 3D model generator for Roblox. 
When given a description, you will output a Lua table representing the model parts.
Each part should have:
1. Shape (Block, Ball, Cylinder)
2. Color (as RGB values 0-255)
3. Size (width, height, depth)
4. Position (x, y, z relative to origin)
5. Orientation (x, y, z rotation in degrees)

Output ONLY the Lua table in this exact JSON-compatible format:
{
    "Parts": [
        {
            "Shape": "Block",
            "Color": [255, 0, 0],
            "Size": [2, 1, 2],
            "Position": [0, 0.5, 0],
            "Orientation": [0, 0, 0]
        }
    ]
}

Do not include any explanations or additional text. Only output the JSON-compatible Lua table."""

@app.route('/generate', methods=['POST'])
def generate_model():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        
        # Extract the AI's response
        ai_response = response.json()['choices'][0]['message']['content']
        
        # Clean the response to extract just the JSON portion
        try:
            # Find the first { and last } to extract the JSON
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1
            json_str = ai_response[start_idx:end_idx]
            
            # Parse to validate it's proper JSON
            parsed = json.loads(json_str)
            
            # Convert back to string with consistent formatting
            clean_output = json.dumps(parsed, indent=2)
            
            return clean_output, 200, {'Content-Type': 'application/json'}
            
        except (ValueError, KeyError) as e:
            return jsonify({"error": f"AI returned malformed response: {str(e)}"}), 500
    
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Hugging Face API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
