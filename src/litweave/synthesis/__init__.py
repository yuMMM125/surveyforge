"""Synthesizer package: stub + real implementation.

Stub is kept for fast non-LLM tests (test_graph_smoke.py marker mocking);
real version is wired into `graph.py` via `make_synthesizer_node`.
"""
from litweave.synthesis.stub import SynthesizeStubNode, make_synthesize_stub_node
from litweave.synthesis.synthesizer import SynthesizerNode, make_synthesizer_node

__all__ = (
    "SynthesizeStubNode",
    "SynthesizerNode",
    "make_synthesize_stub_node",
    "make_synthesizer_node",
)
