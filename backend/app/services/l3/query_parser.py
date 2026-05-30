"""
Parse a natural-language edit prompt into a structured query JSON via Claude.

Schema (L1-only fields for Phase 3a; Phase 3b adds L2 fields):

{
  "duration_target_s": int | null,
  "must_include": {
    "semantic_query":       str | null,   // for SigLIP 2 text-to-image search
    "transcript_keywords":  [str],         // matched against transcripts.text
    "min_focus_score":      float | null,
    "max_motion_magnitude": float | null,
    "min_motion_magnitude": float | null
  },
  "must_exclude": {
    "transcript_keywords": [str]
  },
  "pacing":      "fast" | "medium" | "slow" | null,
  "rhythm_lock": bool      // snap cuts to nearest beat onset when true
}
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.services import prompts as prompts_mod
from app.services.l3.anthropic_client import call_json

logger = logging.getLogger(__name__)

# The actual prompt now lives in backend/app/prompts/query_parser.md.
# Edit the file directly; no restart required.
PROMPT_NAME = "query_parser"


def parse_prompt(prompt: str) -> Dict[str, Any]:
    """Parse a natural-language edit prompt -> structured query dict."""
    system_prompt = prompts_mod.load(PROMPT_NAME)
    data = call_json(system_prompt, prompt, max_tokens=1000)
    return _normalize(data)


def _normalize(q: Dict[str, Any]) -> Dict[str, Any]:
    """Defensive normalization in case Claude omits a key."""
    mi = q.get("must_include") or {}
    me = q.get("must_exclude") or {}
    role = mi.get("narrative_role")
    valid_roles = {"setup", "payoff", "aside", "reaction", "transition"}
    return {
        "duration_target_s": _opt_int(q.get("duration_target_s")),
        "must_include": {
            "semantic_query": _opt_str(mi.get("semantic_query")),
            "transcript_keywords": _str_list(mi.get("transcript_keywords")),
            "min_focus_score": _opt_float(mi.get("min_focus_score")),
            "max_motion_magnitude": _opt_float(mi.get("max_motion_magnitude")),
            "min_motion_magnitude": _opt_float(mi.get("min_motion_magnitude")),
            "narrative_role": role if role in valid_roles else None,
            "acoustic_tags": _str_list(mi.get("acoustic_tags")),
            "min_valence": _opt_float(mi.get("min_valence")),
            "max_valence": _opt_float(mi.get("max_valence")),
        },
        "must_exclude": {
            "transcript_keywords": _str_list(me.get("transcript_keywords")),
            "acoustic_tags": _str_list(me.get("acoustic_tags")),
        },
        "pacing": q.get("pacing") if q.get("pacing") in ("fast", "medium", "slow") else None,
        "rhythm_lock": bool(q.get("rhythm_lock", False)),
        "needs_l2": bool(q.get("needs_l2", False)),
        "preserve_full_shots": bool(q.get("preserve_full_shots", False)),
    }


def _opt_int(v): return int(v) if isinstance(v, (int, float)) else None
def _opt_float(v): return float(v) if isinstance(v, (int, float)) else None
def _opt_str(v): return str(v) if isinstance(v, str) and v.strip() else None
def _str_list(v) -> List[str]:
    if not isinstance(v, list):
        return []
    return [str(x) for x in v if isinstance(x, str) and x.strip()]
