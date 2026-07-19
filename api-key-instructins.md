Analyze this project refer [plan](development-plan.md) [schema](output-schema.md) [readme](README.md) [policy](policy.yaml)


 
Your AWS Bedrock API key for the Crest AI Gateway is ready. Details and quick-start below.

Your Key
<set in env file>
Keep this private. It has a $15 budget — when used up, it stops working until IT resets it.

Connection
Base URL: https://apiproxy.cdsys.local
Auth header: Authorization: Bearer <your-key>
Chat endpoint: /v1/chat/completions

Before you start
You must be on the Crest network or VPN.
Get the certificate file apiproxy.cdsys.crt from IT and note its path.
For Python examples: pip install httpx openai langchain-openai
Windows: put r before file paths in Python (e.g. verify=r"C:\...\cert.crt").


Check your usage anytime
You can log in to the gateway dashboard to see your own spend and remaining budget:

https://apiproxy.cdsys.local/ui — you'll see only your own usage.


Quick start — Python (httpx), recommended
import httpx

resp = httpx.post(
    "https://apiproxy.cdsys.local/v1/chat/completions",
    headers={"Authorization": "Bearer sk-AlxjLT4Rx_W8dTDZfvwdMA"},
    json={
        "model": "Bedrock-ant-
        -4-8",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say hello"}],
    },
    verify=r"C:\Users\you\Desktop\apiproxy.cdsys.crt",
    timeout=60,
)
print(resp.json()["choices"][0]["message"]["content"])

Available models
You have access to all AWS Bedrock models (Claude, Nova, Llama, Mistral, DeepSeek, Qwen, gpt-oss, and more).
Use the exact model name, e.g. Bedrock-ant-opus-4-8 (most capable) or
Bedrock-ant-haiku-4-5-20251001-v1-0 (cheapest/fastest).
Full list and embeddings/rerank examples are in the attached guide from IT.