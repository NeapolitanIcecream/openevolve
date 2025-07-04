"""
Utilities for code parsing, diffing, and manipulation
"""

import re
from typing import Dict, List, Optional, Tuple, Union, Set


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def apply_diff(original_code: str, diff_text: str) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)


def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    code_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Create a short summary
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = (
                f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            )
            replace_summary = (
                f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            )
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Approximate edit distance between two code snippets using a fast token-level
    Jaccard distance instead of the expensive O(N²) Levenshtein algorithm.

    This implementation dramatically reduces runtime for large files (e.g. >10 kB)
    while still providing a reasonable diversity signal for the evolutionary
    algorithm.

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        An integer score ≥ 0 where larger values mean more dissimilar code.
    """

    # Quick exit for identical strings
    if code1 == code2:
        return 0

    # Tokenise by splitting on non-alphanumeric characters. This is fast and
    # memory-efficient because it avoids building huge dynamic-programming
    # tables.
    tokens1: Set[str] = set(re.findall(r"[A-Za-z0-9_]+", code1))
    tokens2: Set[str] = set(re.findall(r"[A-Za-z0-9_]+", code2))

    if not tokens1 and not tokens2:
        return 0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    # Jaccard distance in [0, 1]
    jaccard_distance = 1.0 - len(intersection) / len(union)

    # Scale to an integer similar to edit distance magnitude. We use the average
    # token count as the reference length.
    avg_len = (len(tokens1) + len(tokens2)) / 2
    return int(jaccard_distance * avg_len * 2)  # *2 to keep numbers comparable


def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Detect C/C++ first to avoid misclassifying "class X" declarations as Python
    if re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"

    # Java (match before Python to avoid the generic "class" keyword confusion)
    if re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"

    # Python – require patterns that are distinctive for Python such as a trailing
    # colon on class/def lines or common import styles.
    if re.search(r"^(import|from)\s+\w+", code, re.MULTILINE):
        return "python"
    if re.search(r"^def\s+\w+\s*\(.*\)\s*:", code, re.MULTILINE):
        return "python"
    if re.search(r"^class\s+\w+\s*:\s*", code, re.MULTILINE):
        return "python"

    # JavaScript
    if re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"

    # Rust
    if re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"

    # SQL
    if re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"
