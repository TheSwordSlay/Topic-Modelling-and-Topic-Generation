import requests
import json

OLLAMA_URL = "https://50f9-34-125-214-142.ngrok-free.app/"

def generate_response(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "qwen2.5:14b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False  # This tells Ollama to not stream the response
    }
    
    response = requests.post(f"{OLLAMA_URL}/api/chat", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['message']['content']
    else:
        return f"Error: {response.text}"