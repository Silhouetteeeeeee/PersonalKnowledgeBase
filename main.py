import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

apikey = os.getenv("DEEPSEEK_API_KEY")

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0,
    max_tokens=10000,
    timeout=None,
    max_retries=2
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful agent. You should be kind and friendly to the user."),
    ("human", "{question}")
])

chain = prompt | model | StrOutputParser()
response = chain.invoke({
    "question": "你好，请你告诉我Redis相关知识"
})

print(response)
