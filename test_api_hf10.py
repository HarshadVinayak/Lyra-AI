from openai import OpenAI
import os
client = OpenAI(
	base_url="https://api-inference.huggingface.co/v1/",
	api_key="REPLACED_BY_LYRA_FIX"
)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
	model="meta-llama/Llama-3.2-3B-Instruct", 
	messages=messages, 
	max_tokens=500
)
print(completion.choices[0].message)
