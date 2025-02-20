from langgraph.graph import END, START, StateGraph  # ✅ Updated LangGraph Import
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from rich import print

# ✅ Define a State Class Instead of Using dict
class AgentState:
    def __init__(self, query):
        self.query = query
        self.next_step = None
        self.retrieved_info = None
        self.possible_responses = None
        self.similarity_scores = None
        self.ranked_response = None
        self.confidence_score = None

# ✅ Initialize LLM (LM Studio)
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="hermes-3-llama-3.2-3b",
    temperature=0.7,
    max_tokens=256
)

# ✅ Initialize Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# ✅ Load Existing ChromaDB
vector_db = Chroma(persist_directory="./content/chroma_db", embedding_function=embedding_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# ✅ Define the Agent Decision Step
def agent_decision_step(state: AgentState):
    """Decides whether to retrieve documents or generate responses directly."""
    print("[DEBUG] Agent received query:", state.query) 
    query = state.query
    if "lookup" in query.lower() or "fact" in query.lower():
        state.next_step = "retrieve"
    elif "reason" in query.lower() or "explain" in query.lower():
        state.next_step = "generate"
    else:
        state.next_step = "generate"
    return state

def retrieve_info(state: AgentState):
    """Retrieves relevant documents using ChromaDB."""
    print("[DEBUG] Retrieving documents for query:", state.query)
    
    state.retrieved_info = retriever.invoke(state.query)

    if not state.retrieved_info:
        print("[DEBUG] ❌ No documents retrieved! Check retrieval settings.")
    else:
        print(f"[DEBUG] ✅ Retrieved {len(state.retrieved_info)} documents.")
        for i, doc in enumerate(state.retrieved_info[:3]):  # Print first 3 docs
            print(f"[DEBUG] Doc {i+1}: {doc.page_content[:200]}...")  # Truncated for readability

    return state


# ✅ Define Response Generation Function
def generate_multiple_responses(state: AgentState):
    """Generates multiple responses using the LLM."""
    print("[DEBUG] Generating responses for:", state.query)
    query = state.query
    state.possible_responses = [llm.invoke(query) for _ in range(3)]
    print("[DEBUG] Generated responses:", state.possible_responses)
    return state

# ✅ Define Similarity Evaluation
def evaluate_similarity(state: AgentState):
    """Evaluates the similarity between responses and retrieved documents."""
    print("[DEBUG] Evaluating similarity...")
    if not state.retrieved_info:
        print("[DEBUG] No retrieved docs found. Running retrieval now...")
        retrieve_info(state)  # ✅ Force retrieval before similarity check

    if not state.retrieved_info or not state.possible_responses:
        print("[DEBUG] No retrieved docs or responses. Skipping similarity evaluation.")
        state.similarity_scores = []  # ✅ Prevents NoneType error
        return state
    
    retrieved_texts = [doc.page_content for doc in state.retrieved_info]
    response_embeddings = [embedding_model.embed_query(resp.content) for resp in state.possible_responses]
    retrieved_embeddings = [embedding_model.embed_query(doc) for doc in retrieved_texts]

    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = [
        sum(cosine_similarity([resp_emb], retrieved_embeddings)[0]) / len(retrieved_embeddings)
        for resp_emb in response_embeddings
    ]
    state.similarity_scores = similarities
    print("[DEBUG] Similarity Scores:", state.similarity_scores)
    return state

# ✅ Define Response Ranking
def rank_responses(state: AgentState):
    """Ranks generated responses based on similarity to retrieved documents."""
    print("[DEBUG] Ranking responses...")
    if not state.similarity_scores:
        state.ranked_response, state.confidence_score = "No relevant response found.", 0.0
        print("[DEBUG] No similarity scores. Returning default response.")  # ✅ Debugging line
        return state

    ranked = sorted(zip(state.possible_responses, state.similarity_scores), key=lambda x: x[1], reverse=True)
    state.ranked_response, state.confidence_score = ranked[0]
    print("[DEBUG] Ranked Response:", state.ranked_response)  # ✅ Debugging line
    print("[DEBUG] Confidence Score:", state.confidence_score)  # ✅ Debugging line
    return state

# ✅ Create LangGraph Workflow
workflow = StateGraph(AgentState)  # ✅ Corrected StateGraph Initialization

# ✅ Add Nodes
workflow.add_node("decision", agent_decision_step)
workflow.add_node("retrieve", retrieve_info)
workflow.add_node("generate", generate_multiple_responses)
workflow.add_node("evaluate", evaluate_similarity)
workflow.add_node("rank", rank_responses)

# ✅ Define Start and End
workflow.add_edge(START, "decision")
workflow.add_edge("rank", END)

# ✅ Conditional Edges
workflow.add_conditional_edges(
    "decision",
    lambda state: state.next_step,
    {"retrieve": "retrieve", "generate": "generate"}
)
workflow.add_edge("retrieve", "evaluate")
workflow.add_edge("generate", "evaluate")
workflow.add_edge("evaluate", "rank")

# ✅ Compile the Graph
agent_workflow = workflow.compile()

# ✅ Run the Graph with an Example Query
query = "What are the contraindications of Acetazolamide?"
state = AgentState(query=query)
output = agent_workflow.invoke(state)
print("[DEBUG] Full Output:", output)  # Debugging: Print entire object
print("Ranked Response:", getattr(output, "ranked_response", "No response found"))
print("Confidence Score:", getattr(output, "confidence_score", "N/A"))
