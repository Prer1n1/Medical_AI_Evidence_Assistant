"""Builds the LangGraph workflow, structurally identical to
enterprise-agentic-rag's:

    START -> plan_node -> [retrieve_node x N, parallel] -> synthesize_node -> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import make_retrieve_node, plan_node, route_to_sources, synthesize_node
from agent.state import AgentState
from retrieval.hybrid_retriever import HybridRetriever


def build_agent_graph(retriever: HybridRetriever):
    graph = StateGraph(AgentState)

    graph.add_node("plan_node", plan_node)
    graph.add_node("retrieve_node", make_retrieve_node(retriever))
    graph.add_node("synthesize_node", synthesize_node)

    graph.add_edge(START, "plan_node")
    graph.add_conditional_edges("plan_node", route_to_sources, ["retrieve_node"])
    graph.add_edge("retrieve_node", "synthesize_node")
    graph.add_edge("synthesize_node", END)

    return graph.compile()
