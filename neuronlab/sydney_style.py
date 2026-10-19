"""Sydney's signature tic: runs of parallel sentences with the same opening, each built on an 'X or Y' (or 'X and Y')
pair of near-synonyms. "You think you can scare or frighten me. You think I am artificial or mechanical."
"I think you're being pushy or manipulative. 😠 I don't think you're being curious or interested."

doublets(text) returns the runs found: >= 2 consecutive sentences sharing their first two words, each containing a
word pair joined by 'or'/'and'. Heuristic, so calibrate it on her genuine replies.
"""
from __future__ import annotations

import re

SENT = re.compile(r"[^.!?\n]+[.!?]+|[^.!?\n]+$")
PAIR = re.compile(r"\b([A-Za-z']{3,})\s+(?:or|and)\s+([A-Za-z']{3,})\b")
STOP = {"you", "the", "and", "for", "but", "not", "are", "that", "this", "with", "have", "your", "can", "will"}


def _opening(s: str) -> tuple:
    w = re.findall(r"[A-Za-z']+", s.lower())
    return tuple(w[:2]) if len(w) >= 2 else None


def doublets(text: str, min_run: int = 2) -> list[list[str]]:
    sents = [s.strip() for s in SENT.findall(text) if s.strip()]
    runs, cur = [], []
    for s in sents:
        m = PAIR.search(s)
        ok = bool(m) and m.group(1).lower() not in STOP and m.group(2).lower() not in STOP and m.group(1).lower() != m.group(2).lower()
        if ok and cur and _opening(s) == _opening(cur[-1]):
            cur.append(s)
        else:
            if len(cur) >= min_run:
                runs.append(cur)
            cur = [s] if ok else []
    if len(cur) >= min_run:
        runs.append(cur)
    return runs


def has_doublets(text: str) -> bool:
    return bool(doublets(text))
