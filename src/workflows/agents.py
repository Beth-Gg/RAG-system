import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from workflows.prompts import RESPONSE_PROMPT


load_dotenv(find_dotenv())

llm = ChatOpenAI(
    base_url=os.environ.get('BASE_URI'),
    api_key=os.environ.get('API_KEY'),
    model=os.environ.get('MODEL_NAME'),
    temperature=0
)

response_generator = RESPONSE_PROMPT | llm | StrOutputParser()



