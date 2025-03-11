import os
from dotenv import load_dotenv, find_dotenv
from src.workflows.nodes import retrieve_documents, generate_final_response
from src.workflows.agents import response_generator
from src.workflows.states import RAGState
from src.scripts.embedding_service import PineconeEmbeddingManager
from langgraph.graph import START, END, StateGraph

load_dotenv(find_dotenv())
api_key = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('INDEX_NAME')
name_space = os.environ.get('NAMESPACE')

manager = PineconeEmbeddingManager(api_key=api_key, index_name=index_name, name_space=name_space)

retriever = lambda state: retrieve_documents(state=state, retriever=manager)    
responder = lambda state: generate_final_response(state=state, response_generator=response_generator)

workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retriever)
workflow.add_node("final_response_generator", responder)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "final_response_generator")
workflow.add_edge("final_response_generator", END)

app = workflow.compile()
