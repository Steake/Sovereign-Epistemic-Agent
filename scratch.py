from openai import OpenAI
import os

client = OpenAI(base_url="https://api.deepseek.com", api_key="sk-08e83c1210624090a59fee70318f2d95")
response_stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Return JSON {\"hello\": \"world\"}"}],
    stream=True,
    response_format={"type": "json_object"}
)
for chunk in response_stream:
    if not chunk.choices: continue
    print("CHUNK:", repr(chunk.choices[0].delta.content))
