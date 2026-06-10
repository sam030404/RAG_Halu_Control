from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "안녕! 한 문장으로 자기소개 해줘."}]
)

print(response.choices[0].message.content)