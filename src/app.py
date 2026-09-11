from langgraph.graph import StateGraph, MessagesState, START, END
from src.nodes import use_rag_data


builder = StateGraph(MessagesState)  # 추후에 실제 사용할 State로 교체

builder.add_node('테스트1', use_rag_data)

builder.add_edge(START, '테스트1')
builder.add_edge('테스트1',END)

graph = builder.compile()
