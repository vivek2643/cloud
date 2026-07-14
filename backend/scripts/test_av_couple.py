"""
Tests for cut-level A/V coupling helpers (app.services.l3.sync.av_couple) --
no DB, no network, synthetic envelopes only.

Run:  .venv/bin/python scripts/test_av_couple.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3.sync import av_couple  # noqa: E402


# --------------------------------------------------------------------------
# authoritative_for
# --------------------------------------------------------------------------

def test_authoritative_for_solo_file_is_identity():
    assert av_couple.authoritative_for("f1", {}) == ("f1", 0)
    print("ok  test_authoritative_for_solo_file_is_identity")


def test_authoritative_for_grouped_non_auth_returns_signed_delta():
    sync_info = {
        "f1": {"authoritative_audio_file_id": "f2", "members": {
            "f1": {"offset_ms": 500}, "f2": {"offset_ms": 300},
        }},
    }
    assert av_couple.authoritative_for("f1", sync_info) == ("f2", 200)
    print("ok  test_authoritative_for_grouped_non_auth_returns_signed_delta")


def test_authoritative_for_the_auth_file_itself_is_identity():
    sync_info = {
        "f2": {"authoritative_audio_file_id": "f2", "members": {
            "f1": {"offset_ms": 500}, "f2": {"offset_ms": 300},
        }},
    }
    assert av_couple.authoritative_for("f2", sync_info) == ("f2", 0)
    print("ok  test_authoritative_for_the_auth_file_itself_is_identity")


def test_authoritative_for_malformed_group_data_is_identity():
    # Declared auth isn't actually a member -- never guess a delta from
    # incomplete data.
    sync_info = {"f1": {"authoritative_audio_file_id": "f9",
                        "members": {"f1": {"offset_ms": 100}}}}
    assert av_couple.authoritative_for("f1", sync_info) == ("f1", 0)
    print("ok  test_authoritative_for_malformed_group_data_is_identity")


def test_authoritative_for_no_declared_authoritative_source_is_identity():
    sync_info = {"f1": {"authoritative_audio_file_id": None, "members": {}}}
    assert av_couple.authoritative_for("f1", sync_info) == ("f1", 0)
    print("ok  test_authoritative_for_no_declared_authoritative_source_is_identity")


# --------------------------------------------------------------------------
# refine_offset
# --------------------------------------------------------------------------

def test_refine_offset_recovers_a_known_residual_lag():
    hop_ms = 100
    n = 60
    # video envelope: a single loud spike at hop index 20 (2000-2100ms).
    video_rms = [-40.0] * n
    video_rms[20] = -5.0
    # auth envelope: the SAME shape, but its spike sits 3 hops (300ms) LATER
    # -- the true total alignment shift is +300ms.
    auth_rms = [-40.0] * n
    auth_rms[23] = -5.0
    # The group's globally-solved delta only gets to +200ms; local
    # refinement must find the missing +100ms residual.
    s, e = 1500, 2500
    offset_ms, confidence = av_couple.refine_offset(
        video_rms, auth_rms, hop_ms, s, e, global_delta=200, search_ms=300)
    assert offset_ms == 300, (offset_ms, confidence)
    assert confidence is not None and confidence > 0.9, confidence
    print("ok  test_refine_offset_recovers_a_known_residual_lag")


def test_refine_offset_falls_back_to_global_delta_on_a_flat_envelope():
    hop_ms = 100
    video_rms = [-30.0] * 20    # perfectly flat -- zero variance, nothing to align on
    auth_rms = [-20.0] * 20
    offset_ms, confidence = av_couple.refine_offset(
        video_rms, auth_rms, hop_ms, 500, 1500, global_delta=150)
    assert offset_ms == 150, offset_ms
    assert confidence is None, confidence
    print("ok  test_refine_offset_falls_back_to_global_delta_on_a_flat_envelope")


def test_refine_offset_falls_back_when_video_has_no_envelope():
    offset_ms, confidence = av_couple.refine_offset(
        [], [-20.0] * 20, 100, 500, 1500, global_delta=75)
    assert offset_ms == 75, offset_ms
    assert confidence is None, confidence
    print("ok  test_refine_offset_falls_back_when_video_has_no_envelope")


def test_refine_offset_falls_back_when_hop_ms_is_zero():
    offset_ms, confidence = av_couple.refine_offset(
        [-20.0] * 10, [-20.0] * 10, 0, 500, 1500, global_delta=50)
    assert offset_ms == 50, offset_ms
    assert confidence is None, confidence
    print("ok  test_refine_offset_falls_back_when_hop_ms_is_zero")


def main():
    test_authoritative_for_solo_file_is_identity()
    test_authoritative_for_grouped_non_auth_returns_signed_delta()
    test_authoritative_for_the_auth_file_itself_is_identity()
    test_authoritative_for_malformed_group_data_is_identity()
    test_authoritative_for_no_declared_authoritative_source_is_identity()
    test_refine_offset_recovers_a_known_residual_lag()
    test_refine_offset_falls_back_to_global_delta_on_a_flat_envelope()
    test_refine_offset_falls_back_when_video_has_no_envelope()
    test_refine_offset_falls_back_when_hop_ms_is_zero()
    print("\nall av_couple tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
