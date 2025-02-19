import langgraph
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


llm = ChatOpenAI(
    model_name="hermes-3-llama-3.2-3b",
    openai_api_base="http://127.0.0.1:1234/v1",
    openai_api_key="lm-studio", #not needed, placeholder since we're using LMStudio  
    temperature=0.7,  
    max_tokens=256 
)

embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
vector_db = Chroma(persist_directory="./content/chroma_db", embedding_function=embedding_model)

retriever = vector_db.as_retriever()

retrieval_tool = Tool(
    name="DocumentRetriever",
    func=lambda query: retriever.get_relevant_documents(query),
    description="Searches the document database for relevant information."
)

qa_chain = RetrievalQA(llm=llm, retriever=retriever)

def agent_decision_step(state):
    """
    Custom decision-making based on query context.
    More complex conditions for deciding retrieval or response generation.
    """
    query = state["query"]
    
    if "lookup" in query.lower() or "fact" in query.lower():
        state["next_step"] = "retrieve"
    elif "reason" in query.lower() or "explain" in query.lower():
        state["next_step"] = "generate"
    else:
        
        state["next_step"] = "generate"

    return state

def retrieve_info(state):
    """
    Retrieves relevant documents using ChromaDB.
    """
    query = state["query"]
    retrieved_docs = retrieval_tool.run(query)
    state["retrieved_info"] = retrieved_docs
    return state

def generate_multiple_responses(state):
    """
    Uses the LLM (Hermes 3) to generate multiple responses for the same query.
    """
    query = state["query"]
    
    responses = []
    for _ in range(5):  
        response = qa_chain.run(query)
        responses.append(response)
    
    state["possible_responses"] = responses
    return state

def evaluate_similarity(state):
    """
    Evaluates the similarity between the generated responses and retrieved document chunks using cosine similarity.
    """
    
    retrieved_texts = [doc.page_content for doc in state["retrieved_info"]]
    responses = state["possible_responses"]

    
    retrieved_embeddings = [embedding_model.embed(doc) for doc in retrieved_texts]
    response_embeddings = [embedding_model.embed(response) for response in responses]

 
    similarities = []
    for response_embedding in response_embeddings:
        response_similarity = np.mean([cosine_similarity([response_embedding], [doc_embedding])[0][0] for doc_embedding in retrieved_embeddings])
        similarities.append(response_similarity)

    state["similarity_scores"] = similarities
    return state

def rank_responses(state):
    """
    Rank the generated responses based on their similarity to the retrieved documents.
    """
   
    response_with_scores = list(zip(state["possible_responses"], state["similarity_scores"]))
    ranked_responses = sorted(response_with_scores, key=lambda x: x[1], reverse=True)

    state["ranked_response"] = ranked_responses[0][0] 
    state["confidence_score"] = ranked_responses[0][1]  
    return state


workflow = langgraph.Graph()

workflow.add_node("decision", agent_decision_step)
workflow.add_node("retrieve", retrieve_info)
workflow.add_node("generate_multiple", generate_multiple_responses)
workflow.add_node("evaluate_similarity", evaluate_similarity)
workflow.add_node("rank_responses", rank_responses)

workflow.set_entry_point("decision")
workflow.add_edge("decision", "retrieve", condition=lambda state: state["next_step"] == "retrieve")
workflow.add_edge("decision", "generate_multiple", condition=lambda state: state["next_step"] == "generate")

workflow.add_edge("generate_multiple", "evaluate_similarity")
workflow.add_edge("retrieve", "evaluate_similarity")
workflow.add_edge("evaluate_similarity", "rank_responses")

agent_workflow = workflow.compile()

query = "What are the contraindications of Acetazolamide?"
output = agent_workflow.invoke({"query": query})

print("Ranked Response: ", output["ranked_response"])
print("Confidence Score: ", output["confidence_score"])
