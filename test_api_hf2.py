import requests
import os
HF_KEY = "REPLACED_BY_LYRA_FIX"
url = "https://api-inference.huggingface.co/v1/chat/completions"
resp = requests.post(url, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"model":"mistralai/Mistral-7B-Instruct-v0.3", "messages":[{"role":"user","content":"hi"}]})
print("HF Global v0.3:", resp.status_code, resp.text)
