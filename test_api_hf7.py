import requests
HF_KEY = "REPLACED_BY_LYRA_FIX"
url = "https://api-inference.huggingface.co/models/google/gemma-7b-it"
resp = requests.post(url, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"inputs":"hi"})
print("HF Gemma:", resp.status_code, resp.text)
url2 = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
resp2 = requests.post(url2, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"inputs":"hi"})
print("HF Mistral:", resp2.status_code, resp2.text)
