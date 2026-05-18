from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable


logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


class OllamaHTTPError(OllamaError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = int(status_code)
        self.code = self.status_code
        self.detail = detail
        super().__init__(f"Ollama HTTP {self.status_code}: {detail}")


class OllamaCancelled(RuntimeError):
    pass


@dataclass
class OllamaClient:
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 120
    connect_timeout: float | None = None
    request_timeout: float | None = None
    stream_idle_timeout: float | None = None

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def _connect_timeout(self) -> float:
        return float(self.connect_timeout or self.request_timeout or self.timeout or 120)

    def _request_timeout(self) -> float:
        return float(self.request_timeout or self.timeout or 120)

    def _stream_idle_timeout(self) -> float:
        return float(self.stream_idle_timeout or self.request_timeout or self.timeout or 120)

    def _set_response_timeout(self, response: Any, timeout: float) -> None:
        try:
            response.fp.raw._sock.settimeout(float(timeout))
        except Exception:
            return

    def _json_request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self._url(path), data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self._connect_timeout()) as response:
                self._set_response_timeout(response, self._request_timeout())
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaHTTPError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"Ollama connection error: {exc}") from exc
        if not raw:
            return {}
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise OllamaError(f"Unexpected Ollama response: {type(loaded).__name__}")
        return loaded

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._json_request("/api/tags")
        models = payload.get("models") or []
        return [item for item in models if isinstance(item, dict)]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        num_predict: int = 1024,
        cancel_check: Callable[[], bool] | None = None,
    ) -> str:
        if not model:
            raise OllamaError("Generation model is not configured")
        if cancel_check is not None:
            return self._chat_streaming(
                model,
                messages,
                temperature=temperature,
                num_predict=num_predict,
                cancel_check=cancel_check,
            )
        payload = self._json_request(
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": float(temperature),
                    "num_predict": int(num_predict),
                },
            },
        )
        message = payload.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else ""
        if isinstance(content, str):
            return content.strip()
        response = payload.get("response")
        if isinstance(response, str):
            return response.strip()
        raise OllamaError("Ollama did not return chat content")

    def _chat_streaming(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        num_predict: int,
        cancel_check: Callable[[], bool],
    ) -> str:
        if cancel_check():
            raise OllamaCancelled("Ollama chat canceled")
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(num_predict),
            },
        }
        request = urllib.request.Request(
            self._url("/api/chat"),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/x-ndjson",
                "Content-Type": "application/json",
            },
        )
        parts: list[str] = []
        try:
            with urllib.request.urlopen(request, timeout=self._connect_timeout()) as response:
                self._set_response_timeout(response, self._stream_idle_timeout())
                for raw_line in response:
                    if cancel_check():
                        try:
                            response.close()
                        finally:
                            raise OllamaCancelled("Ollama chat canceled")
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed Ollama stream line: %r", line[:200])
                        continue
                    if not isinstance(payload, dict):
                        continue
                    message = payload.get("message") or {}
                    content = message.get("content") if isinstance(message, dict) else ""
                    if isinstance(content, str):
                        parts.append(content)
                    response_text = payload.get("response")
                    if isinstance(response_text, str):
                        parts.append(response_text)
                    if payload.get("done"):
                        break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaHTTPError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            if cancel_check():
                raise OllamaCancelled("Ollama chat canceled") from exc
            raise OllamaError(f"Ollama connection error: {exc}") from exc
        if cancel_check():
            raise OllamaCancelled("Ollama chat canceled")
        return "".join(parts).strip()

    def embed(self, model: str, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        if not model:
            raise OllamaError("Embedding model is not configured")

        try:
            payload = self._json_request(
                "/api/embed",
                {"model": model, "input": values},
            )
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list):
                return [list(map(float, item)) for item in embeddings]
        except OllamaHTTPError as modern_error:
            if modern_error.status_code not in {404, 405}:
                raise
            legacy_vectors = []
            for text in values:
                payload = self._json_request(
                    "/api/embeddings",
                    {"model": model, "prompt": text},
                )
                embedding = payload.get("embedding")
                if not isinstance(embedding, list):
                    raise modern_error
                legacy_vectors.append(list(map(float, embedding)))
            return legacy_vectors

        raise OllamaError("Ollama did not return embeddings")


@dataclass
class OpenAIEmbeddingClient:
    base_url: str = "http://127.0.0.1:5010/v1"
    api_key: str = ""
    timeout: int = 120

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def embed(self, model: str, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        if not model:
            raise OllamaError("Embedding model is not configured")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._url("/embeddings"),
            data=json.dumps({"model": model, "input": values}).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"OpenAI embeddings HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"OpenAI embeddings connection error: {exc}") from exc

        payload = json.loads(raw or "{}")
        data = payload.get("data")
        if not isinstance(data, list):
            raise OllamaError("OpenAI-compatible endpoint did not return embeddings data")
        ordered = sorted(
            [item for item in data if isinstance(item, dict)],
            key=lambda item: int(item.get("index") or 0),
        )
        if len(ordered) != len(values):
            raise OllamaError("OpenAI-compatible endpoint returned invalid embeddings")
        embeddings: list[list[float]] = []
        for item in ordered:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise OllamaError("OpenAI-compatible endpoint returned invalid embeddings")
            embeddings.append([float(value) for value in embedding])
        return embeddings


@dataclass
class OpenAIChatClient:
    base_url: str = "http://127.0.0.1:8081/v1"
    api_key: str = ""
    timeout: float = 120
    connect_timeout: float | None = None
    request_timeout: float | None = None
    stream_idle_timeout: float | None = None

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def _connect_timeout(self) -> float:
        return float(self.connect_timeout or self.request_timeout or self.timeout or 120)

    def _request_timeout(self) -> float:
        return float(self.request_timeout or self.timeout or 120)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _json_request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(self._url(path), data=data, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout()) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaHTTPError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"OpenAI chat connection error: {exc}") from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"OpenAI chat request timed out after {self._request_timeout():g}s"
            ) from exc
        loaded = json.loads(raw or "{}")
        if not isinstance(loaded, dict):
            raise OllamaError(f"Unexpected OpenAI chat response: {type(loaded).__name__}")
        return loaded

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._json_request("/models")
        rows = payload.get("data")
        if not isinstance(rows, list):
            rows = payload.get("models")
        output: list[dict[str, Any]] = []
        iter_rows = rows if isinstance(rows, list) else []
        for item in iter_rows:
            if not isinstance(item, dict):
                continue
            name = item.get("id") or item.get("name") or item.get("model")
            if not name:
                continue
            normalized = dict(item)
            normalized["name"] = str(name)
            normalized["model"] = str(item.get("model") or name)
            output.append(normalized)
        return output

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        num_predict: int = 1024,
        cancel_check: Callable[[], bool] | None = None,
    ) -> str:
        if not model:
            raise OllamaError("Generation model is not configured")
        if cancel_check is not None and cancel_check():
            raise OllamaCancelled("OpenAI chat canceled")
        payload = self._json_request(
            "/chat/completions",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": float(temperature),
                "max_tokens": int(num_predict),
            },
        )
        if cancel_check is not None and cancel_check():
            raise OllamaCancelled("OpenAI chat canceled")
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
                text = first.get("text")
                if isinstance(text, str):
                    return text.strip()
        raise OllamaError("OpenAI-compatible endpoint did not return chat content")


@dataclass
class OpenAIRerankClient:
    base_url: str = "http://127.0.0.1:5010/v1"
    api_key: str = ""
    timeout: int = 120

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def rerank(self, model: str, query: str, documents: Iterable[str]) -> list[float]:
        values = list(documents)
        if not values:
            return []
        if not model:
            raise OllamaError("Rerank model is not configured")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._url("/rerank"),
            data=json.dumps({"model": model, "query": query, "documents": values}).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"OpenAI rerank HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"OpenAI rerank connection error: {exc}") from exc

        payload = json.loads(raw or "{}")
        scores = payload.get("scores")
        if isinstance(scores, list) and len(scores) == len(values):
            return [float(item) for item in scores]

        rows = payload.get("results") or payload.get("data")
        if not isinstance(rows, list):
            raise OllamaError("OpenAI-compatible endpoint did not return rerank results")
        output = [0.0 for _ in values]
        seen = set()
        for fallback_index, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            raw_index = item.get("index", fallback_index)
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(output):
                continue
            raw_score = (
                item.get("relevance_score")
                if "relevance_score" in item
                else item.get("score", item.get("rank_score", 0.0))
            )
            output[index] = float(raw_score or 0.0)
            seen.add(index)
        if len(seen) != len(values):
            raise OllamaError("OpenAI-compatible endpoint returned incomplete rerank results")
        return output


def make_generation_client(
    config: Any,
    *,
    request_timeout: float | None = None,
    connect_timeout: float | None = None,
    stream_idle_timeout: float | None = None,
    ollama_client_cls: type[Any] = OllamaClient,
) -> Any:
    timeout = float(
        request_timeout
        if request_timeout is not None
        else getattr(config, "request_timeout_sec", 120) or 120
    )
    provider = str(getattr(config, "deep_generation_provider", "") or "").strip().lower()
    base_url = str(getattr(config, "deep_generation_base_url", "") or "").strip()
    openai_providers = {"openai", "openai-compatible", "llamacpp", "llama.cpp", "giga"}
    if provider in openai_providers or (base_url and provider != "ollama"):
        return OpenAIChatClient(
            base_url or "http://127.0.0.1:8081/v1",
            api_key=str(getattr(config, "deep_generation_api_key", "") or ""),
            timeout=timeout,
            connect_timeout=float(
                connect_timeout
                if connect_timeout is not None
                else getattr(config, "connect_timeout_sec", timeout)
                or timeout
            ),
            request_timeout=timeout,
            stream_idle_timeout=float(
                stream_idle_timeout
                if stream_idle_timeout is not None
                else getattr(config, "stream_idle_timeout_sec", timeout)
                or timeout
            ),
        )
    return ollama_client_cls(
        getattr(config, "ollama_base_url", "http://127.0.0.1:11434"),
        timeout=timeout,
        connect_timeout=float(
            connect_timeout
            if connect_timeout is not None
            else getattr(config, "connect_timeout_sec", timeout)
            or timeout
        ),
        request_timeout=timeout,
        stream_idle_timeout=float(
            stream_idle_timeout
            if stream_idle_timeout is not None
            else getattr(config, "stream_idle_timeout_sec", timeout)
            or timeout
        ),
    )


def make_embedding_client(config: Any) -> Any:
    provider = str(getattr(config, "embedding_provider", "ollama") or "ollama").strip().lower()
    timeout = int(getattr(config, "request_timeout_sec", 120) or 120)
    connect_timeout = float(getattr(config, "connect_timeout_sec", timeout) or timeout)
    if provider in {"openai", "openai-compatible", "llamacpp", "llama.cpp", "giga"}:
        base_url = (
            getattr(config, "embedding_base_url", "")
            or getattr(config, "ollama_base_url", "")
            or "http://127.0.0.1:5010/v1"
        )
        return OpenAIEmbeddingClient(
            str(base_url),
            api_key=str(getattr(config, "embedding_api_key", "") or ""),
            timeout=timeout,
        )
    return OllamaClient(
        getattr(config, "ollama_base_url", "http://127.0.0.1:11434"),
        timeout=timeout,
        connect_timeout=connect_timeout,
        request_timeout=timeout,
        stream_idle_timeout=float(getattr(config, "stream_idle_timeout_sec", timeout) or timeout),
    )


def make_rerank_client(config: Any) -> OpenAIRerankClient | None:
    if not bool(getattr(config, "rerank_enabled", False)):
        return None
    model = str(getattr(config, "rerank_model", "") or "").strip()
    if not model:
        return None
    timeout = int(getattr(config, "request_timeout_sec", 120) or 120)
    base_url = (
        getattr(config, "embedding_base_url", "")
        or getattr(config, "ollama_base_url", "")
        or "http://127.0.0.1:5010/v1"
    )
    return OpenAIRerankClient(
        str(base_url),
        api_key=str(getattr(config, "embedding_api_key", "") or ""),
        timeout=timeout,
    )
