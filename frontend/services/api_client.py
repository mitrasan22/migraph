import json
from typing import Dict, Any, Iterable

import requests


def query_once(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{api_url}/query/"
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def stream_query(api_url: str, payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    url = f"{api_url}/query/stream"
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            current_event = "chunk"

            for line in r.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue

                data = line.split(":", 1)[1].lstrip()

                if current_event == "chunk":
                    yield {"type": "chunk", "data": data}
                else:
                    try:
                        meta = json.loads(data)
                    except json.JSONDecodeError:
                        meta = {"raw": data}
                    yield {"type": current_event, "data": meta}
    except requests.exceptions.ChunkedEncodingError as exc:
        yield {"type": "error", "data": f"Stream ended early: {exc}"}
    except requests.exceptions.RequestException as exc:
        yield {"type": "error", "data": f"Streaming error: {exc}"}


def fetch_graph(api_url: str) -> Dict[str, Any]:
    url = f"{api_url}/graph/export"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_history(api_url: str, limit: int = 25) -> list[Dict[str, Any]]:
    url = f"{api_url}/query/history"
    r = requests.get(url, params={"limit": limit}, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_graph_stats(api_url: str) -> Dict[str, Any]:
    url = f"{api_url}/graph/stats"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()
