# from torch import bfloat16
# import transformers
# from torch import cuda
# import ollama

# # bnb_config = transformers.BitsAndBytesConfig(
# #     load_in_4bit=True,  # 4-bit quantization
# #     bnb_4bit_quant_type='nf4',  # Normalized float 4
# #     bnb_4bit_use_double_quant=True,  # Second quantization after the first
# #     bnb_4bit_compute_dtype=bfloat16  # Computation type
# # )

# # model_id = 'Qwen/Qwen2.5-1.5B-Instruct'

# # # Llama 2 Tokenizer
# # tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# # # Llama 2 Model
# # model = transformers.AutoModelForCausalLM.from_pretrained(
# #     model_id,
# #     trust_remote_code=True,
# #     quantization_config=bnb_config,
# #     device_map='auto',
# # )
# # model.eval()

# # # Our text generator
# # generator = transformers.pipeline(
# #     model=model, tokenizer=tokenizer,
# #     task='text-generation',
# #     temperature=0.1,
# #     max_new_tokens=500,
# #     repetition_penalty=1.1
# # )
# ollama.BASE_URL = "https://72b5-34-124-149-249.ngrok-free.app/"
# print("Available models:", ollama.list()['models'])

import requests
import json

OLLAMA_URL = "https://809d-34-16-218-202.ngrok-free.app/"

def generate_response(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "qwen2.5:1.5b",
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