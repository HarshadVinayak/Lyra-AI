import requests
HF_KEY = "REPLACED_BY_LYRA_FIX"
url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B-Instruct/v1/chat/completions"
resp = requests.post(url, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"model":"meta-llama/Llama-3.2-1B-Instruct", "messages":[{"role":"user","content":"hi"}]})
print("HF 1B:", resp.status_code, resp.text)
url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta/v1/chat/completions"
resp = requests.post(url, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"model":"HuggingFaceH4/zephyr-7b-beta", "messages":[{"role":"user","content":"hi"}]})
print("HF Zephyr:", resp.status_code, resp.text)
