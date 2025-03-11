from scripts.embedding_service import PineconeEmbeddingManager
from workflows.states import RAGState
from langchain_openai import ChatOpenAI

def retrieve_documents(state: RAGState, retriever: PineconeEmbeddingManager) -> RAGState:
    print('---RETRIEVING DOCUMENTS---')
    prompt = state['prompt']
    
    documents = retriever.search_matching(query=prompt)

    return {"prompt": prompt, "documents": documents}

def generate_final_response(state: RAGState, response_generator: ChatOpenAI) -> RAGState:
    print('---GENERATING Final FAQ RESPONSE---')
    prompt = state['prompt']
    documents = state['documents']

    result = response_generator.invoke({"prompt": prompt, "documents": documents})
    
    return {
        "prompt": prompt,
        "documents": documents,
        "generation": result
    }
