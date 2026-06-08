"""Tests for utterance reconstruction."""

from src.causalmap.preprocess import reconstruct_utterances, pseudonymize_turns


def _make_seg(speaker, start, end, text, table="t1", round_id="1", room="r1", pid="p1"):
    return {
        "speaker": speaker,
        "start_time": str(start),
        "end_time": str(end),
        "transcript": text,
        "table_id": table,
        "iteration_cycle": round_id,
        "room_id": room,
        "participant_id": pid,
    }


def test_merge_consecutive_same_speaker():
    segs = [
        _make_seg("Alice", 1.0, 2.0, "Hello"),
        _make_seg("Alice", 2.5, 3.5, "how are you"),
        _make_seg("Bob", 4.0, 5.0, "Good thanks"),
    ]
    turns = reconstruct_utterances(segs, gap_threshold=1.5)
    assert len(turns) == 2
    assert turns[0]["text"] == "Hello how are you"
    assert turns[0]["speaker"] == "Alice"
    assert turns[1]["text"] == "Good thanks"


def test_no_merge_across_gap():
    segs = [
        _make_seg("Alice", 1.0, 2.0, "Hello"),
        _make_seg("Alice", 5.0, 6.0, "how are you"),
    ]
    turns = reconstruct_utterances(segs, gap_threshold=1.5)
    assert len(turns) == 2


def test_no_merge_different_speakers():
    segs = [
        _make_seg("Alice", 1.0, 2.0, "Hello"),
        _make_seg("Bob", 2.2, 3.0, "Hi"),
    ]
    turns = reconstruct_utterances(segs, gap_threshold=1.5)
    assert len(turns) == 2


def test_no_merge_across_tables():
    segs = [
        _make_seg("Alice", 1.0, 2.0, "Hello", table="t1"),
        _make_seg("Alice", 2.2, 3.0, "Hi", table="t2"),
    ]
    turns = reconstruct_utterances(segs, gap_threshold=1.5)
    assert len(turns) == 2


def test_pseudonymize_is_deterministic():
    turns = [{"speaker": "Alice"}, {"speaker": "Alice"}, {"speaker": "Bob"}]
    result = pseudonymize_turns(turns)
    assert result[0]["speaker"] == result[1]["speaker"]
    assert result[0]["speaker"] != result[2]["speaker"]
    assert result[0]["speaker"].startswith("Speaker_")
