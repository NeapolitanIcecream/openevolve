from __future__ import annotations

import re
from typing import List, Tuple

__all__ = [
    "clean_diff",
    "minhash_signature",
    "minhash_similarity",
]

# ----------------------------- Diff Cleaning -----------------------------

def clean_diff(raw_diff: str) -> Tuple[str, str]:
    """Clean a git diff string into two levels.

    Parameters
    ----------
    raw_diff: str
        The raw diff produced by ``git diff`` between *root* and *target* commits.

    Returns
    -------
    tuple
        (prompt_diff, hash_diff)
        ``prompt_diff`` keeps file headers so that an LLM can understand where each
        change happens, while ``hash_diff`` strips headers and any other metadata so
        that two semantically-identical diffs get the same textual representation
        for hashing/MinHash.
    """

    # Split lines once to avoid repetitive ``split("\n")`` calls.
    lines: List[str] = raw_diff.splitlines()

    prompt_lines: List[str] = []  # Level-A: for prompts
    code_lines: List[str] = []    # Level-B: for hashing

    # Regex patterns to discard noise lines
    META_RE = re.compile(
        r"^(diff --git|index [0-9a-f]{7}\.[0-9a-f]{7}|@@ |Binary files |new file mode|deleted file mode)"
    )

    for line in lines:
        if META_RE.match(line):
            # Keep *file header* lines only for prompt version
            if line.startswith("diff --git") or line.startswith("--- ") or line.startswith("+++ "):
                prompt_lines.append(line)
            # Skip for hashing level
            continue

        # Keep ---/+++ headers for prompt but remove from hash variant
        if line.startswith("--- ") or line.startswith("+++ "):
            prompt_lines.append(line)
            continue

        # Only interested in added/removed lines for both versions
        if line.startswith("+") or line.startswith("-"):
            # Remove leading +/- for hash version, keep for prompt
            cleaned = line[1:]
            cleaned = cleaned.rstrip()  # strip trailing whitespace / CRLF
            cleaned = cleaned.expandtabs(4)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                # Empty after cleaning
                continue
            # Comment stripping (python + c-style)
            if re.match(r"^#", cleaned) or re.match(r"^//", cleaned):
                continue
            prompt_lines.append(line)
            code_lines.append(cleaned)

    prompt_diff = "\n".join(prompt_lines)
    hash_diff = "\n".join(code_lines)
    return prompt_diff, hash_diff

# ----------------------------- MinHash utils -----------------------------

try:
    from datasketch import MinHash as _DSMinHash

    _HAS_DATASKETCH = True
except ImportError:  # Fallback for environments without datasketch
    _HAS_DATASKETCH = False


_DEF_NUM_PERM = 64  # Signature length (number of hash permutations)
_SHINGLE_LEN = 5


def _shingles(text: str, shingle_len: int = _SHINGLE_LEN) -> List[str]:
    """Generate overlapping character shingles of fixed length."""
    if len(text) < shingle_len:
        return [text]
    return [text[i : i + shingle_len] for i in range(0, len(text) - shingle_len + 1)]


def _minhash_signature_datasketch(text: str, num_perm: int = _DEF_NUM_PERM) -> List[int]:
    m = _DSMinHash(num_perm=num_perm)
    for token in _shingles(text):
        m.update(token.encode("utf-8"))
    return list(m.hashvalues)


def _minhash_signature_simple(text: str, num_perm: int = _DEF_NUM_PERM) -> List[int]:
    """Lightweight pure-Python MinHash (fallback)."""
    sig = []
    tokens = _shingles(text)
    if not tokens:
        return [0] * num_perm
    for i in range(num_perm):
        salt = f"{i}-salt"
        sig.append(min(hash(salt + tok) for tok in tokens))
    return sig


# Public API

def minhash_signature(text: str, num_perm: int = _DEF_NUM_PERM) -> List[int]:
    """Generate MinHash signature for *text*.

    Uses `datasketch` when available, otherwise falls back to a pure-Python version.
    """
    if _HAS_DATASKETCH:
        return _minhash_signature_datasketch(text, num_perm=num_perm)
    return _minhash_signature_simple(text, num_perm=num_perm)


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures (arrays)."""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a) 