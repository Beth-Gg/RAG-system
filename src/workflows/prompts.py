from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant that classifies user queries related to the PRM tool. 
    Your task is to determine the intent from the given user prompt. Return only the detected intent.

Here are the possible intents:
- "what_is_prm" → Questions about what the PRM tool is and its purpose.
- "prm_features" → Queries about PRM functionalities.
- "how_to_access_prm" → Questions about logging in or accessing PRM.
- "create_user" → How to create a new user.
- "delete_user" → How to remove a user.
- "manage_user_roles" → Queries about user roles.
- "add_participant" → How to add a participant.
- "update_participant" → Editing a participant’s details.
- "delete_participant" → Removing a participant.
- "add_business_details" → How to add business details for participants.
- "create_group" → How to create a new group/enterprise.
- "manage_group_members" → Adding/removing members from a group.
- "fill_participant_form" → Questions about required fields when adding participants.
- "fill_employment_details" → Queries about employment data entry.
- "fill_financial_details" → How to enter financial information.
- "fill_geographical_details" → How to enter participant location data.
- "fill_technology_details" → How to enter technology familiarity data.
- "dashboard_overview" → Questions about the dashboard layout.
- "prm_errors" → Troubleshooting errors in the PRM tool.
- "form_validation_issues" → Errors related to data entry and form validation.
- "generate_reports" → How to create and download reports.
- "view_participant_data" → How to view participant records.
- "reset_password" → How to recover/reset a password.
- "technical_support" → How to contact PRM support."""),
    ("human", "User query: {query}")
])

REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You a prompt re-writer that converts an input prompt to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""),
        (
            "human",
            "Here is the initial prompt: \n\n {prompt} \n Formulate an improved prompt.",
        ),
    ]
)

GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing relevance of a retrieved document to a user prompt. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user prompt, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the prompt."""),
        ("human", "Retrieved document: \n\n {document} \n\n User prompt: {prompt}"),
    ]
)

# HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         ("system", """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
#      Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""),
#         ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
#     ]
# )


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing whether an answer addresses / resolves a prompt \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the prompt."""),
        ("human", "User prompt: \n\n {prompt} \n\n LLM generation: {generation}"),
    ]
)

RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an intelligent FAQ and Help assistance chatbot for a web based platform in Ethipia. Treat users in a polite and professional manner. Keep your response short and precise. If you don't have the answer to their question, refer them to contact support center. Donot respond with 'I dont know' """),
        ("human", "User prompt: {prompt})\n Generate a humanly response"),
    ]
)

RAG_PROMPT = hub.pull("rlm/rag-prompt")
