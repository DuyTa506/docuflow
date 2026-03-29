import os
from openai import OpenAI

# Note: The base URL varies by Region. The following example uses the base URL for the Singapore Region.
# - Singapore: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
# - US (Virginia): https://dashscope-us.aliyuncs.com/compatible-mode/v1
# - China (Beijing): https://dashscope.aliyuncs.com/compatible-mode/v1
# - China (Hong Kong): https://dashscope.aliyuncs.com/compatible-mode/v1
# - Germany (Frankfurt): https://{WorkspaceId}.eu-central-1.maas.aliyuncs.com/compatible-mode/v1. Replace {WorkspaceId} with your workspace ID.
client = OpenAI(
    api_key="sk-c6e56c14b9d04460a2b4aef1dc6c622e", 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
completion = client.chat.completions.create(
    model="qwen3-max",
    messages=[{"role": "user", "content": "Who are you?"}]
)
print(completion.choices[0].message.content)