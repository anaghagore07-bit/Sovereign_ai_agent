from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from state import AgentState

class RouteDecision(BaseModel):
    decision: str = Field(
        description="Must be 'coding', 'doc_read', or 'doc_write'."
    )

def orchestrator_node(state: AgentState):
    llm = ChatOllama(model="llama3.1:8b", format="json", temperature=0)
    prompt = f"Classify this task: '{state['user_prompt']}'. Options: 'coding', 'doc_read', or 'doc_write'."
    response = llm.with_structured_output(RouteDecision).invoke(prompt)
    print(f"\n[Orchestrator] --> Route: {response.decision}")
    return {"selected_route": response.decision, "retry_count": 0}