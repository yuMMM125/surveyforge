"""Agent node factories (per spec § 2.4 + § 2.6).

Each agent module exports `make_<role>_node(router, registry, ...) -> Callable`
that returns a LangGraph-compatible node bound to the given LLMRouter +
PromptRegistry. Graph init (Task 6) constructs router/registry once and
binds them via closure into per-node callables.
"""
from surveyforge.agents.planner import PlannerNode, make_planner_node

__all__ = ("PlannerNode", "make_planner_node")
