'''
Contains the logic for correcting edit tool parameters.
'''

import asyncio
import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, NamedTuple, Optional, cast

from openevolve.llm.base import LLMInterface, ChatMessage
# --- Caching --- #

class LruCache:
    '''A simple implementation of an LRU cache.'''
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()

edit_correction_cache = LruCache(50)
file_content_correction_cache = LruCache(50)

# --- Data Structures --- #

class CorrectedEditParams(NamedTuple):
    file_path: str
    old_string: str
    new_string: str

class CorrectedEditResult(NamedTuple):
    params: CorrectedEditParams
    occurrences: int

# --- String Utilities --- #

def count_occurrences(text: str, sub: str) -> int:
    '''Counts non-overlapping occurrences of a substring.'''
    if not sub:
        return 0
    return text.count(sub)

def unescape_string_for_llm_bug(input_string: str) -> str:
    '''Corrects overly escaped strings from LLM outputs.'''
    def replace_match(match: re.Match[str]) -> str:
        captured_char = match.group(1)
        return {
            'n': '\n',
            't': '\t',
            'r': '\r',
            "'": "'",
            '"': '"',
            '`': '`',
            '\\': '\\',
            '\n': '\n'
        }.get(captured_char, match.group(0))
    # Use regex: one or more backslashes followed by n/t/r/single-quote/double-quote/backtick/backslash/newline
    return re.sub(r'\\+(n|t|r|\'|"|`|\\|\n)', replace_match, input_string)

def trim_pair_if_possible(target: str, pair: str, content: str, expected_replacements: int):
    '''Trims whitespace from target and pair if the trimmed target still matches uniquely.'''
    trimmed_target = target.strip()
    if len(target) != len(trimmed_target):
        trimmed_occurrences = count_occurrences(content, trimmed_target)
        if trimmed_occurrences == expected_replacements:
            return trimmed_target, pair.strip()
    return target, pair

# --- External Edit Check --- #

def get_timestamp_from_function_id(fcn_id: str) -> int:
    '''Extracts the timestamp from a function call/response ID.'''
    parts = fcn_id.split('-')
    if len(parts) > 2:
        try:
            return int(parts[1])
        except ValueError:
            return -1
    return -1

async def find_last_edit_timestamp(file_path: str, client: LLMInterface) -> int:
    """Find timestamp of the last relevant tool interaction with a file (OpenAI style).

    Expectations for history entries (OpenAI Chat API compatible):
      - Assistant tool call message:
        {
          "role": "assistant",
          "content": None | str,
          "tool_calls": [
            {"id": str, "type": "function", "function": {"name": str, "arguments": str}}
          ]
        }
      - Tool response message:
        {
          "role": "tool",
          "tool_call_id": str,
          "name": str,
          "content": str
        }

    We prefer timestamps on entries via keys like "created"/"timestamp" (seconds or ms).
    If absent, we cannot recover time and will return -1.
    """
    history: list[ChatMessage] = await client.get_history() or []

    # Define tool names of interest. We prioritize write operations.
    tools_in_response = {"edit"}
    tools_in_call = {"edit", "read_file", "read_many_files"}

    def _extract_ts(entry: ChatMessage) -> Optional[int]:
        ts = cast(Any, entry.get("created") or entry.get("timestamp") or entry.get("created_at"))
        if isinstance(ts, (int, float)):
            ts_int = int(ts)
            # Normalize to milliseconds
            return ts_int if ts_int > 10_000_000_000 else ts_int * 1000
        return None

    last_ts: int = -1

    # Scan newest to oldest
    for entry in reversed(history):
        role = entry.get("role")
        tool_calls = entry.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list):
            for tc in tool_calls:
                try:
                    f_id = tc.get("id")
                    fn = (tc.get("function") or {})
                    name = (fn.get("name") or "").strip()
                    args = fn.get("arguments")
                except Exception:
                    continue
                if name not in tools_in_call:
                    continue
                # arguments is typically a JSON string
                arg_text = args if isinstance(args, str) else json.dumps(args or {})
                if file_path and file_path in arg_text:
                    ts = _extract_ts(entry)
                    if ts is not None:
                        last_ts = max(last_ts, ts)

        elif role == "tool":
            name = (cast(Any, entry.get("name")) or "").strip()
            if name not in tools_in_response:
                continue
            content = entry.get("content")
            content_text = content if isinstance(content, str) else json.dumps(content or {})
            if file_path and file_path in content_text:
                ts = _extract_ts(entry)
                if ts is not None:
                    last_ts = max(last_ts, ts)

    return last_ts

# --- LLM Correction Logic --- #

OLD_STRING_CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_target_snippet": {
            "type": "string",
            "description": "The corrected version of the target snippet that exactly and uniquely matches a segment within the provided file content.",
        }
    },
    "required": ["corrected_target_snippet"],
}

NEW_STRING_CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_new_string": {
            "type": "string",
            "description": "The original_new_string adjusted to be a suitable replacement for the corrected_old_string, while maintaining the original intent of the change.",
        }
    },
    "required": ["corrected_new_string"],
}

CORRECT_NEW_STRING_ESCAPING_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_new_string_escaping": {
            "type": "string",
            "description": "The new_string with corrected escaping, ensuring it is a proper replacement for the old_string.",
        }
    },
    "required": ["corrected_new_string_escaping"],
}

CORRECT_STRING_ESCAPING_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_string_escaping": {
            "type": "string",
            "description": "The string with corrected escaping.",
        }
    },
    "required": ["corrected_string_escaping"],
}

async def correct_old_string_mismatch(client: LLMInterface, file_content: str, problematic_snippet: str, abort_signal: asyncio.Event) -> str:
    prompt = f'''
Context: A process needs to find an exact literal, unique match for a specific text snippet within a file's content. The provided snippet failed to match exactly. This is most likely because it has been overly escaped.

Task: Analyze the provided file content and the problematic target snippet. Identify the segment in the file content that the snippet was *most likely* intended to match. Output the *exact*, literal text of that segment from the file content. Focus *only* on removing extra escape characters and correcting formatting, whitespace, or minor differences to achieve a PERFECT literal match. The output must be the exact literal text as it appears in the file.

Problematic target snippet:
```
{problematic_snippet}
```

File content:
{file_content}

Return ONLY the corrected target snippet in the specified JSON format with the key 'corrected_target_snippet'. If no clear, unique match can be found, return an empty string for 'corrected_target_snippet'.
    '''.strip()
    try:
        out = await client.invoke(
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": OLD_STRING_CORRECTION_SCHEMA},
            },
        )
        result_obj: Any = out.json or {}
        if isinstance(result_obj, dict):
            maybe = result_obj.get('corrected_target_snippet')
            if isinstance(maybe, str) and maybe:
                return maybe
    except Exception as e:
        if abort_signal.is_set(): raise
        print(f"Error during LLM call for old_string correction: {e}")
    return problematic_snippet

async def correct_new_string(client: LLMInterface, original_old: str, corrected_old: str, original_new: str, abort_signal: asyncio.Event) -> str:
    prompt = f'''
Context: The `old_string` in an `edit` operation was slightly modified to ensure a unique and exact match within the file. The corresponding `new_string` must now be updated to reflect this change, preserving the original intent of the edit.

Original `old_string`:
```
{original_old}
```

Corrected `old_string` (which is now an exact match in the file):
```
{corrected_old}
```

Original `new_string` (intended replacement for the original `old_string`):
```
{original_new}
```

Task: Generate an updated `new_string` that is a suitable replacement for the `corrected_old_string`. The updated version should maintain the original intent of the change. For example, if a newline was trimmed from `old_string`, the same should probably be done for `new_string`.

Return ONLY the corrected `new_string` in the specified JSON format with the key 'corrected_new_string'.
    '''.strip()
    try:
        out = await client.invoke(
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": NEW_STRING_CORRECTION_SCHEMA},
            },
        )
        result_obj: Any = out.json or {}
        if isinstance(result_obj, dict):
            maybe = result_obj.get('corrected_new_string')
            if isinstance(maybe, str) and maybe:
                return maybe
    except Exception as e:
        if abort_signal.is_set(): raise
        print(f"Error during LLM call for new_string correction: {e}")
    return original_new

async def correct_new_string_escaping(client: LLMInterface, old_string: str, new_string: str, abort_signal: asyncio.Event) -> str:
    prompt = f'''
Context: An LLM has just generated the `new_string` for an edit operation, but it might have been improperly escaped (e.g., using `\\n` instead of `
`). This can cause issues when applying the edit. The `old_string` is provided for context.

`old_string` (for context):
```
{old_string}
```

`potentially_problematic_new_string` (this is the text that should replace `old_string`, but MIGHT have bad escaping):
```
{new_string}
```

Task: Analyze the `potentially_problematic_new_string`. If it's syntactically invalid due to incorrect escaping, correct the invalid syntax. The goal is to ensure the `new_string`, when inserted into the code, will be valid.

Return ONLY the corrected string in the specified JSON format with the key 'corrected_new_string_escaping'. If no escaping correction is needed, return the original `potentially_problematic_new_string`.
    '''.strip()
    try:
        out = await client.invoke(
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": CORRECT_NEW_STRING_ESCAPING_SCHEMA},
            },
        )
        result_obj: Any = out.json or {}
        if isinstance(result_obj, dict):
            maybe = result_obj.get('corrected_new_string_escaping')
            if isinstance(maybe, str) and maybe:
                return maybe
    except Exception as e:
        if abort_signal.is_set(): raise
        print(f"Error during LLM call for new_string escaping correction: {e}")
    return new_string

async def correct_string_escaping(client: LLMInterface, text: str, abort_signal: asyncio.Event) -> str:
    prompt = f'''
Context: An LLM has just generated `potentially_problematic_string` and the text might have been improperly escaped (e.g. too many backslashes for newlines like `\\n` instead of `
`).

`potentially_problematic_string` (this text MIGHT have bad escaping):
```
{text}
```

Task: Analyze the `potentially_problematic_string`. If it's syntactically invalid due to incorrect escaping, correct the invalid syntax.

Return ONLY the corrected string in the specified JSON format with the key 'corrected_string_escaping'. If no escaping correction is needed, return the original `potentially_problematic_string`.
    '''.strip()
    try:
        out = await client.invoke(
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": CORRECT_STRING_ESCAPING_SCHEMA},
            },
        )
        result_obj: Any = out.json or {}
        if isinstance(result_obj, dict):
            maybe = result_obj.get('corrected_string_escaping')
            if isinstance(maybe, str) and maybe:
                return maybe
    except Exception as e:
        if abort_signal.is_set(): raise
        print(f"Error during LLM call for string escaping correction: {e}")
    return text

# --- Main Correction Orchestration --- #

async def ensure_correct_edit(
    file_path: str,
    current_content: str,
    original_params: Dict[str, Any],
    client: LLMInterface,
    abort_signal: asyncio.Event
) -> CorrectedEditResult:
    cache_key = f"{current_content}---{original_params['old_string']}---{original_params['new_string']}"
    cached_result = edit_correction_cache.get(cache_key)
    if cached_result:
        return cached_result

    final_new_string = original_params['new_string']
    new_string_potentially_escaped = unescape_string_for_llm_bug(final_new_string) != final_new_string
    expected_replacements = original_params.get('expected_replacements', 1)

    final_old_string = original_params['old_string']
    occurrences = count_occurrences(current_content, final_old_string)

    if occurrences == expected_replacements:
        if new_string_potentially_escaped:
            final_new_string = await correct_new_string_escaping(client, final_old_string, final_new_string, abort_signal)
    elif occurrences > expected_replacements:
        pass
    else:
        unescaped_old = unescape_string_for_llm_bug(final_old_string)
        occurrences = count_occurrences(current_content, unescaped_old)

        if occurrences == expected_replacements:
            final_old_string = unescaped_old
            if new_string_potentially_escaped:
                final_new_string = await correct_new_string(client, original_params['old_string'], unescaped_old, final_new_string, abort_signal)
        elif occurrences == 0:
            last_edit_time = await find_last_edit_timestamp(file_path, client)
            if last_edit_time > 0:
                file_mtime_ms = os.path.getmtime(file_path) * 1000
                if file_mtime_ms - last_edit_time > 2000:
                    return CorrectedEditResult(params=CorrectedEditParams(**original_params), occurrences=0)

            llm_corrected_old = await correct_old_string_mismatch(client, current_content, unescaped_old, abort_signal)
            llm_occurrences = count_occurrences(current_content, llm_corrected_old)

            if llm_occurrences == expected_replacements:
                final_old_string = llm_corrected_old
                occurrences = llm_occurrences
                if new_string_potentially_escaped:
                    base_new_for_llm = unescape_string_for_llm_bug(final_new_string)
                    final_new_string = await correct_new_string(client, original_params['old_string'], llm_corrected_old, base_new_for_llm, abort_signal)
            else:
                occurrences = 0

    final_old_string, final_new_string = trim_pair_if_possible(final_old_string, final_new_string, current_content, expected_replacements)
    final_occurrences = count_occurrences(current_content, final_old_string)

    result = CorrectedEditResult(
        params=CorrectedEditParams(
            file_path=original_params['file_path'],
            old_string=final_old_string,
            new_string=final_new_string,
        ),
        occurrences=final_occurrences,
    )
    edit_correction_cache.set(cache_key, result)
    return result

async def ensure_correct_file_content(client: LLMInterface, content: str, abort_signal: asyncio.Event) -> str:
    cached_result = file_content_correction_cache.get(content)
    if cached_result:
        return cached_result

    if unescape_string_for_llm_bug(content) == content:
        file_content_correction_cache.set(content, content)
        return content

    corrected_content = await correct_string_escaping(client, content, abort_signal)
    file_content_correction_cache.set(content, corrected_content)
    return corrected_content
