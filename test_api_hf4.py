import requests
HF_KEY = "REPLACED_BY_LYRA_FIX"

models = ["HuggingFaceH4/zephyr-7b-beta", "meta-llama/Llama-3.2-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]
for m in models:
    url = f"https://api-inference.huggingface.co/models/{m}"
    resp = requests.post(url, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"inputs":"[INST] hi [/INST]"})
    print(m, resp.status_code, str(resp.content)[:100])
