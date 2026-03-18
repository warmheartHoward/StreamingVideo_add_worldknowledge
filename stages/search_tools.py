"""Internal search API (大搜) wrapper module.

Provides access to the internal knowledge search service via:
- SSE HTTP endpoint for initial query
- WebSocket endpoint for follow-up generation

All credentials and endpoints are configured via LabelerConfig.
"""

import base64
import hashlib
import hmac
import json
import logging
import time
from hashlib import sha256

import requests
from websockets.sync.client import connect

logger = logging.getLogger(__name__)

# Default headers for the search API
_DEFAULT_HEADERS = {
    "deviceid": "wyrtestdevice",
    "Appversion": "11.2.9.300",
    "Appname": "com.huawei.hmos.vassistant",
}


class SearchClient:
    """Client for the internal knowledge search API (大搜).

    Handles both the initial SSE request and the WebSocket follow-up.
    """

    def __init__(
        self,
        api_url: str,
        ws_url: str,
        secret_key: str,
        access_key: str,
    ):
        self._api_url = api_url
        self._ws_url = ws_url
        self._secret_key = secret_key
        self._access_key = access_key

    def search(self, text: str, dialogue_history: list | None = None) -> tuple[int, any, dict]:
        """Execute a search query against the internal API.

        Args:
            text: Search query text.
            dialogue_history: Optional conversation history.

        Returns:
            (type_code, result_content, params_dict)
            type_code: 0=direct text, 1=needs websocket follow-up, 2=error
        """
        if dialogue_history is None:
            dialogue_history = []

        session_id = str(time.time())
        interaction_id = len(dialogue_history) + 1

        data = {
            "session": {"sessionId": session_id, "interactionId": interaction_id},
            "context": {
                "text": text,
                "enhancedRetrieval": True,
                "clientContext": {
                    "businessType": "worldVision",
                    "serviceCenterData": [
                        {
                            "featureType": "CONTENT_CARD",
                            "featureVersion": "16.0",
                            "extInfo": "{}",
                        }
                    ],
                },
                "dialogueContext": {"dialogueHistory": dialogue_history},
            },
        }

        req_type, res0, params = self._sse_request(data)

        if req_type == 0:
            return req_type, res0, params
        elif req_type == 1:
            data2 = {
                "session": {"sessionId": session_id, "interactionId": interaction_id},
                "context": {"clientContext": {"businessType": "worldVision"}},
                "body": {
                    "text": res0,
                    "enhancedRetrieval": True,
                    "searchRewriteText": res0,
                    "deviceInfo": {"deviceType": "phone"},
                },
            }
            ws_result = self._ws_request(data2)
            return req_type, ws_result, params
        else:
            return req_type, None, {}

    def search_simple(self, query: str) -> tuple[list[dict], str]:
        """Simplified search returning parsed results.

        Args:
            query: Search keyword (Chinese preferred).

        Returns:
            (results_list, formatted_json_string)
        """
        try:
            _, _, params = self.search(query)
            if params:
                results_list, json_str = parse_search_items_to_json(params)
                if json_str == "[]":
                    return [], "未找到相关结果。"
                return results_list, json_str
            return [], "接口调用成功，但无返回条目。"
        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            return [], f"Error: {str(e)}"

    def _sse_request(self, data: dict) -> tuple[int, str, dict]:
        """Send SSE request to the search API."""
        stamp = str(int(time.time() * 1000))
        digest_hash = hmac.new(
            self._secret_key.encode(),
            stamp.encode(),
            digestmod=sha256,
        ).digest()
        sign = base64.b64encode(digest_hash).decode()

        headers = {
            "x-device-id": "wyrtestdevice",
            "Content-Type": "application/json",
            "ts": stamp,
            "sign": sign,
            "accessKey": self._access_key,
        }

        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json=data,
                stream=True,
                timeout=50,
            )
            response.encoding = "utf-8"
        except Exception as e:
            logger.warning(f"SSE connection error: {e}")
            return 2, "connection error", {}

        if response.status_code != 200:
            logger.warning(f"SSE HTTP error: {response.status_code}")
            return 2, "http error", {}

        gene = ""
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            gene += chunk

        try:
            res_json0 = json.loads(gene)
        except Exception:
            return 2, "json error", {}

        directives = res_json0.get("directives")
        if directives is None:
            return 2, "no return", {}

        judge_type = directives[0].get("name", "")
        params = directives[0].get("params", {})

        if judge_type == "displayText":
            res0 = directives[1]["params"]["text"].replace("\n\n", "\n")
            return 0, res0, params
        elif judge_type == "llmResponse":
            res1 = res_json0["context"]["searchRewriteText"]
            return 1, res1, params

        return 2, "unknown type", {}

    def _ws_request(self, data: dict) -> dict | None:
        """Send WebSocket follow-up request."""
        try:
            with connect(self._ws_url, additional_headers=_DEFAULT_HEADERS) as ws:
                ws.send(json.dumps(data))
                while True:
                    message = ws.recv()
                    message = message.replace("\u200b", "")
                    json_res = json.loads(message)
                    res0 = json_res.get("result", {})
                    if res0.get("type") == "finalText":
                        return res0
        except Exception as e:
            if "received 1000 (OK)" not in repr(e):
                logger.debug(f"WebSocket closed: {e}")
        return None


def parse_search_items_to_json(search_params: dict) -> tuple[list[dict], str]:
    """Parse search result params into a structured list.

    Args:
        search_params: The params dict from SSE response.

    Returns:
        (results_list, json_string)
    """
    try:
        if not isinstance(search_params, dict):
            return [], "[]"

        items = search_params.get("items", [])
        if not isinstance(items, list) or len(items) == 0:
            return [], "[]"

        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            text = item.get("text", "").strip()
            url = item.get("url", "").strip()
            if title or text:
                results.append({"title": title, "text": text, "url": url})

        return results, json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return [], json.dumps({"error": str(e)})


def format_search_results_for_llm(search_result: list[dict]) -> str:
    """Format search results into LLM-friendly text.

    Args:
        search_result: List of search result dicts.

    Returns:
        Formatted text string.
    """
    try:
        parts = []
        for index, item in enumerate(search_result, 1):
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            text = item.get("text", "").strip()
            if not title and not text:
                continue
            if title and text:
                parts.append(f"{index}. 【{title}】\n{text}")
            elif title:
                parts.append(f"{index}. 【{title}】")
            else:
                parts.append(f"{index}. {text}")
        return "\n\n".join(parts) if parts else "未找到有效内容。"
    except Exception as e:
        return f"解析结果出错: {str(e)}"
