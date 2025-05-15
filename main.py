import requests

HF_API_TOKEN = "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"
MODEL_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz-560m"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

data = {
    "inputs": "Create a Lua table of a 3D flower model made from parts with color and position."
}

response = requests.post(MODEL_URL, headers=headers, json=data)

print("Status Code:", response.status_code)
print("Response:")
print(response.text)
