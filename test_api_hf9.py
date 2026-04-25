import requests
HF_KEY = "REPLACED_BY_LYRA_FIX"
url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
for model in ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-Nemo-Instruct-2407", "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"]:
    resp = requests.post(url, headers={"Authorization": f"Bearer {HF_KEY}"}, json={"model":model, "messages":[{"role":"user","content":"hi"}]})
    print(f"HF Router {model}:", resp.status_code, resp.text[:100])
