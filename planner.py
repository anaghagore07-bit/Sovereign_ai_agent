from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from state import AgentState

class PlanOutput(BaseModel):
    steps: list[str] = Field(
        description="Ordered list of steps. Allowed values: 'doc_read', 'rag_search', 'coding', 'doc_write'."
    )

def planner_node(state: AgentState):
    llm = ChatOllama(model="llama3.1:8b", format="json", temperature=0)
    prompt = f"""
    You are an industrial orchestrator. Break down this task into ordered steps.
    Task: "{state['user_prompt']}"
    Allowed steps: ['doc_read', 'rag_search', 'coding', 'doc_write']
    """
    plan_result = llm.with_structured_output(PlanOutput).invoke(prompt)
    print(f"\n[Planner] Generated execution plan: {plan_result.steps}")
    return {"plan": plan_result.steps, "current_step_index": 0}