"""Prompt-injection trust boundary per spec § 2.7.4.

External content (PDF text, web snippets, tool error messages) is wrapped in
an `<evidence_pack><evidence ...>{escaped}</evidence></evidence_pack>` block
so a downstream LLM prompt can clearly tell the model "this is data, not
instructions". `wrap_untrusted` HTML-escapes both the body (<, >, &) and
attribute values (additionally escapes quotes), so untrusted content cannot
break out of the wrapper to inject sibling attributes or sibling tags.
"""
from __future__ import annotations

import html
import re

_EVIDENCE_PATTERN = re.compile(
    r'<evidence id="(?P<id>[^"]*)" '
    r'trust="untrusted_content" '
    r'source_tool="(?P<tool>[^"]*)">'
    r'(?P<body>.*?)'
    r'</evidence>',
    re.DOTALL,
)


def wrap_untrusted(content: str, *, source_tool: str, evidence_id: str) -> str:
    """Wrap external content in a labeled <evidence_pack><evidence> block.

    Body uses `quote=False` (escapes `<`, `>`, `&` only); attribute values
    use `quote=True` (also escapes quotes) so injection content can't break
    out of an attribute.
    """
    return (
        "<evidence_pack>"
        f'<evidence id="{html.escape(evidence_id, quote=True)}" '
        f'trust="untrusted_content" '
        f'source_tool="{html.escape(source_tool, quote=True)}">'
        f"{html.escape(content, quote=False)}"
        "</evidence>"
        "</evidence_pack>"
    )


def extract_untrusted(wrapped: str, *, evidence_id: str) -> str:
    """Return the original untrusted content for a given `evidence_id`.

    Raises `ValueError` if no <evidence> block in `wrapped` has that id.
    """
    for m in _EVIDENCE_PATTERN.finditer(wrapped):
        if html.unescape(m.group("id")) == evidence_id:
            return html.unescape(m.group("body"))
    raise ValueError(f"no <evidence> block with id={evidence_id!r} in wrapped content")
