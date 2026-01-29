from __future__ import annotations

import json
from typing import Any

import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers


DEFAULT_TIMEOUT_S = 600


def hf_read_bytes(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> bytes:
    """
    Read a file from the HuggingFace Hub directly into memory (no on-disk cache).
    """

    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    headers = build_hf_headers(token=token)
    resp = requests.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return resp.content


def hf_read_json(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> Any:
    data = hf_read_bytes(
        repo_id,
        filename,
        revision=revision,
        token=token,
        timeout_s=timeout_s,
    )
    return json.loads(data.decode("utf-8"))

