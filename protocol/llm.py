import ipaddress
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from .validation import ProtocolError, validate


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProtocolError("LLM redirects are disabled")


class LocalLLM:
    """Ollama adapter restricted to loopback; no proxy or cloud fallback."""

    def __init__(self, model=None, base_url=None, timeout=None):
        self.model = model or os.getenv("LOCAL_LLM_MODEL", "")
        self.base_url = (base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        try:
            self.timeout = float(timeout if timeout is not None else os.getenv("LOCAL_LLM_TIMEOUT", "120"))
            parsed = urlparse(self.base_url)
            host = parsed.hostname
            local = host == "localhost" or ipaddress.ip_address(host).is_loopback
            if not local or parsed.scheme not in ("http", "https") or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
                raise ValueError()
            _ = parsed.port
        except (ValueError, TypeError):
            raise ProtocolError("Use a loopback Ollama URL, for example http://127.0.0.1:11434") from None
        if not self.model.strip():
            raise ProtocolError("Set LOCAL_LLM_MODEL to an installed local model")
        if "cloud" in self.model.lower():
            raise ProtocolError("Cloud models are not allowed for meeting transcripts")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ProtocolError("LOCAL_LLM_TIMEOUT must be a positive finite number")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def complete(self, system, data, schema):
        payload = {
            "model": self.model, "stream": False, "format": schema, "think": False,
            "options": {"temperature": 0, "num_ctx": 16384},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False, allow_nan=False)},
            ],
        }
        request = Request(self.base_url + "/api/chat", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(4_000_001)
            if len(raw) > 4_000_000:
                raise ProtocolError("LLM response exceeds 4 MB limit")
            envelope = json.loads(raw)
            if envelope.get("done") is not True or envelope.get("done_reason") == "length":
                raise ProtocolError("LLM response is incomplete")
            result = json.loads(envelope["message"]["content"])
        except HTTPError as error:
            raise ProtocolError(f"Local LLM HTTP error {error.code}; check installed model and runtime") from None
        except (URLError, TimeoutError, OSError):
            raise ProtocolError("Local LLM unavailable or timed out; check runtime and timeout") from None
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ProtocolError("Local LLM returned malformed JSON") from None
        validate(result, schema)
        return result
