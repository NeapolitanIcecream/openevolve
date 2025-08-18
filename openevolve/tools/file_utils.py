"""
Utilities for file operations.
"""

import base64
import mimetypes
import os
from typing import Any, Dict, Union, Optional

DEFAULT_ENCODING = "utf-8"


def is_within_root(path: str, root: str) -> bool:
    """Checks if a path is within the root directory."""
    return os.path.abspath(path).startswith(os.path.abspath(root))


def get_specific_mime_type(file_path: str) -> str:
    """Gets the specific MIME type for a file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def detect_file_type(file_path: str) -> str:
    """Detects the file type (text, image, pdf, or binary)."""
    mime_type = get_specific_mime_type(file_path)
    if mime_type.startswith("text/"):
        return "text"
    if mime_type.startswith("image/"):
        return "image"
    if mime_type == "application/pdf":
        return "pdf"
    return "binary"


def process_single_file_content(
    file_path: str, root_dir: str, offset: Optional[int] = None, limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Processes the content of a single file, handling text, images, and PDFs.
    """
    if not is_within_root(file_path, root_dir):
        return {
            "error": f"Security: File path is outside the root directory: {file_path}",
            "returnDisplay": "Error: Access denied.",
        }

    try:
        file_type = detect_file_type(file_path)

        if file_type in ["image", "pdf"]:
            with open(file_path, "rb") as f:
                encoded_content = base64.b64encode(f.read()).decode(DEFAULT_ENCODING)
            mime_type = get_specific_mime_type(file_path)
            # OpenAI style: return binary as a data URI string
            data_uri = f"data:{mime_type};base64,{encoded_content}"
            return {
                "llmContent": data_uri,
                "returnDisplay": f"Read binary file (as data URI): {os.path.relpath(file_path, root_dir)}",
            }

        # Handle text files
        with open(file_path, "r", encoding=DEFAULT_ENCODING) as f:
            lines = f.readlines()

        if offset is not None and limit is not None:
            content = "".join(lines[offset : offset + limit])
            display = (
                f"Read lines {offset}-{offset+limit-1} from {os.path.relpath(file_path, root_dir)}"
            )
        else:
            content = "".join(lines)
            display = f"Read file: {os.path.relpath(file_path, root_dir)}"

        return {"llmContent": content, "returnDisplay": display}

    except FileNotFoundError:
        return {"error": f"File not found: {file_path}", "returnDisplay": "Error: File not found."}
    except Exception as e:
        return {"error": str(e), "returnDisplay": f"Error reading file: {e}"}
