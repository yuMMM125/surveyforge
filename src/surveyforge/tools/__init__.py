"""External tool wrappers (per spec § 3).

Each wrapper exports a `register(gateway: ToolGateway)` function. Graph init
(Task 6) calls all three to attach real policies + impls onto the per-run
gateway, replacing the placeholder `_OpaqueArgs`/`_OpaqueOutput` policies in
`runtime.tool_gateway.TOOL_REGISTRY`.
"""
from surveyforge.tools import arxiv_lookup, arxiv_search, s2_lookup, web_search

__all__ = ("arxiv_lookup", "arxiv_search", "s2_lookup", "web_search")
