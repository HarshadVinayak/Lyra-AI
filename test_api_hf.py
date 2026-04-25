import requests
import os
HF_KEY = "REPLACED_BY_LYRA_FIX"
url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"
resp = requests.post(url, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"model":"mistralai/Mistral-7B-Instruct-v0.3", "messages":[{"role":"user","content":"hi"}]})
print("HF v0.3:", resp.status_code, resp.text)
if not resp.ok:
    url2 = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct/v1/chat/completions"
    resp2 = requests.post(url2, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"model":"meta-llama/Llama-3.2-3B-Instruct", "messages":[{"role":"user","content":"hi"}]})
    print("HF Llama:", resp2.status_code, resp2.text)
