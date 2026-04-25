import requests
SAMBA_KEY = "REPLACED_BY_LYRA_FIX"
resp = requests.get("https://api.sambanova.ai/v1/models", headers={"Authorization": f"Bearer {SAMBA_KEY}"})
if resp.ok:
    print([m['id'] for m in resp.json().get('data', [])])
else:
    print(resp.status_code, resp.text)
