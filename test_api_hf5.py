import requests
HF_KEY = "REPLACED_BY_LYRA_FIX"
url = "https://router.huggingface.co/v1/chat/completions"
resp = requests.post(url, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"model":"meta-llama/Llama-3.2-3B-Instruct", "messages":[{"role":"user","content":"hi"}]})
print("HF Router Llama:", resp.status_code, resp.text)
