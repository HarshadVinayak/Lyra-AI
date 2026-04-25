import requests
import os

SAMBA_KEY = "REPLACED_BY_LYRA_FIX"
resp = requests.post("https://api.sambanova.ai/v1/chat/completions", headers={"Authorization": f"Bearer {SAMBA_KEY}"}, json={"model": "Meta-Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "hi"}]})
print("Samba1:", resp.status_code, resp.text)

resp = requests.post("https://fast-api.snova.ai/v1/chat/completions", headers={"Authorization": f"Bearer {SAMBA_KEY}"}, json={"model": "Meta-Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "hi"}]})
print("Samba2:", resp.status_code, resp.text)

