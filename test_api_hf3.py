import requests
HF_KEY = "REPLACED_BY_LYRA_FIX"
url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
resp = requests.post(url, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"inputs":"[INST] hi [/INST]"})
print("HF Standard API:", resp.status_code, resp.text)
