"""HTTP client with retry, exponential backoff and a simple rate limiter."""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)


class ApiError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, retryable: bool = False):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable


class HttpClient:
    def __init__(self, base_url: str, timeout: float = 15.0, max_retries: int = 5,
                 min_interval: float = 0.05):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout,
                                   headers={"User-Agent": "polyagent/0.1"})

    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def get(self, path: str, params: dict | None = None) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        return self._request("GET", path, params=params)

    def post(self, path: str, body: Any) -> Any:
        return self._request("POST", path, json=body)

    def _request(self, method: str, path: str, **kw) -> Any:
        delay = 1.0
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self.client.request(method, path, **kw)
                if r.status_code == 429 or r.status_code >= 500:
                    raise ApiError(f"{r.status_code} {r.text[:200]}", r.status_code, True)
                if r.status_code >= 400:
                    raise ApiError(f"{r.status_code} {path}: {r.text[:300]}", r.status_code, False)
                return r.json()
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_err = e
            except ApiError as e:
                if not e.retryable:
                    raise
                last_err = e
            if attempt < self.max_retries:
                sleep = delay * (1 + random.random() * 0.25)
                log.warning(method + " %s failed (%s); retry %d in %.1fs", path, last_err, attempt + 1, sleep)
                time.sleep(sleep)
                delay = min(delay * 2, 30)
        raise ApiError(f"{method} {path} failed after retries: {last_err}", retryable=True)

    def close(self) -> None:
        self.client.close()
