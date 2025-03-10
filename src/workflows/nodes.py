from scripts.embedding_service import PineconeEmbeddingManager
from workflows.states import RAGState
from langchain_openai import ChatOpenAI
from workflows.agents import structured_intent_detector

def detect_intent(state: RAGState, intent_detector: ChatOpenAI) -> RAGState:
    print('---DETECTING INTENT---')
    prompt = state['prompt']

    # structure_intent_detector = structured_intent_detector.with_structured_output(GradeIntent)

    intent_response = intent_detector.invoke({"query": prompt})
    detected_intent = intent_response.intent

    print(f'Detected Intent: {detected_intent}')
    return {
        'prompt': prompt, 
        'intent': detected_intent,
        'generation': intent_response
    }

def retrieve_documents(state: RAGState, retriever: PineconeEmbeddingManager) -> RAGState:
    print('---RETRIEVING DOCUMENTS---')
    prompt = state['prompt']
    
    if state['intent']:
        prompt = f" {prompt} (Intent: {state['intent']})"
    documents = retriever.search_matching(query=prompt)

    return {"prompt": prompt, "documents": documents}

def grade_documents(state: RAGState, document_grader: ChatOpenAI) -> RAGState:
    print("---GRADING DOCUMENT RELEVANCE---")
    prompt = state['prompt']
    documents = state['documents']

    filtered_docs = []
    for doc in documents:
        score = document_grader.invoke({"prompt": prompt, "document": doc})
        if score == 'yes':
            filtered_docs.append(doc)
    
    return {"documents": filtered_docs, "prompt": prompt}

def generate_response(state: RAGState, answer_generator: ChatOpenAI) -> RAGState:
    print('---GENERATING FAQ RESPONSE---')
    prompt = state['prompt']
    documents = state.get("documents", [])
    detected_intent = state.get('detected_intent', None)

    if detected_intent in ["reset_password", "technical_support"]:
        return {
            "prompt": prompt,
            "generation": "Please contact PRM support at support@prmtool.com or call +123-456-7890."
        }

    result = answer_generator.invoke({"question": prompt, "context": documents})
    
    return {
        "prompt": prompt,
        "documents": documents,
        "generation": result
    }

def transform_query(state: RAGState, prompt_rewriter: ChatOpenAI) -> RAGState:
    print('---REWRITING QUERY---')
    prompt = state['prompt']
    
    result = prompt_rewriter.invoke({"prompt": prompt})
    rewrite_count = state.get('rewrite_count', 0) + 1
    
    return {"prompt": result}

def generate_final_response(state: RAGState, response_generator: ChatOpenAI) -> RAGState:
    print('---GENERATING Final FAQ RESPONSE---')
    prompt = state['prompt']
    documents = state['documents']
    generation = state['generation']


    result = response_generator.invoke({"prompt": prompt, "generation": generation})
    
    return {
        "prompt": prompt,
        "documents": documents,
        "generation": result
    }
