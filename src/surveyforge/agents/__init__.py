"""Agent node factories (per spec § 2.4 + § 2.6).

Each agent module exports `make_<role>_node(router, registry, ...) -> Callable`
that returns a LangGraph-compatible node bound to the given LLMRouter +
PromptRegistry. Graph init (Task 6) constructs router/registry once and
binds them via closure into per-node callables.
"""
from surveyforge.agents.planner import PlannerNode, make_planner_node
from surveyforge.agents.researcher_deep import (
    ResearcherDeepNode,
    make_researcher_deep_node,
)
from surveyforge.agents.researcher_wide import (
    ResearcherWideNode,
    make_researcher_wide_node,
)

__all__ = (
    "PlannerNode",
    "ResearcherDeepNode",
    "ResearcherWideNode",
    "make_planner_node",
    "make_researcher_deep_node",
    "make_researcher_wide_node",
)
