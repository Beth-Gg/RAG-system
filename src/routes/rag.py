from fastapi import FastAPI
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from dtos.rag import RAGRequest
from workflows.graphs import app as rag_app

app = FastAPI()


rag_router = APIRouter(prefix='/rag', tags=['RAG'])

@rag_router.post("/query")
async def get_response(request: RAGRequest):
    result = rag_app.invoke(
        {"prompt" : request.question}
    )
    
    return result 

app.include_router(rag_router)
