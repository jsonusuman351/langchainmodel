from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()

llm = ChatDeepSeek(model="deepseek-chat")
result = llm.invoke("What is the capital of India")

print(result.content)


