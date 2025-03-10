import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from workflows.prompts import  ANSWER_PROMPT, REWRITER_PROMPT, RAG_PROMPT, GRADER_PROMPT, INTENT_PROMPT, RESPONSE_PROMPT
from workflows.models import GradeAnswer, GradeHallucinations, GradeDocuments, GradeIntent


load_dotenv(find_dotenv())

llm = ChatOpenAI(
    base_url=os.environ.get('BASE_URI'),
    api_key=os.environ.get('API_KEY'),
    model=os.environ.get('MODEL_NAME'),
    temperature=0
)


structured_intent_detector = llm.with_structured_output(GradeIntent)
intent_detector = INTENT_PROMPT | structured_intent_detector

structured_answer_grader = llm.with_structured_output(GradeAnswer)
answer_grader = ANSWER_PROMPT | structured_answer_grader 

prompt_rewriter = REWRITER_PROMPT | llm | StrOutputParser()

structured_document_grader = llm.with_structured_output(GradeDocuments)
document_grader = GRADER_PROMPT | structured_document_grader 

answer_generator = RAG_PROMPT | llm | StrOutputParser()

response_generator = RESPONSE_PROMPT | llm | StrOutputParser()



