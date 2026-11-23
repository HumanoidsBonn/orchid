"""HTTPModule: call external HTTP endpoints (model APIs, perception, LLM, etc.)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
import src.core.logger as log


class HTTPModule:
    """HTTP client for calling external service endpoints."""

    def __init__(self, base_url: str, config: dict[str, Any] | None = None):
        self.base_url = base_url.rstrip("/")
        self.config = dict(config or {})

    def call_sync(
        self,
        payload: dict,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        """Make a synchronous HTTP POST request."""
        request_timeout = timeout or self.config.get("timeout", 5.0)
        merged_headers = self._build_headers(headers)

        try:
            # import requests

            log.info("HTTPModule.call_sync(%s)", self.base_url)
            # resp = requests.post(
            #     self.base_url,
            #     json=payload,
            #     headers=merged_headers,
            #     timeout=request_timeout,
            # )
            return self._post_json_urllib(payload, merged_headers, request_timeout)
            log.info('POST request successful!')
            resp.raise_for_status()
            return resp.json()
        # except ImportError:
        #     log.error("requests unavailable; falling back to urllib for %s", self.base_url)
        #     return self._post_json_urllib(payload, merged_headers, request_timeout)
        # except requests.exceptions.ConnectTimeout:
        #     log.error("❌ NETWORK ERROR: Could not establish a connection to the server machine.")
        #     return {"error": str(requests.exceptions.ConnectTimeout)}
        # except requests.exceptions.ReadTimeout:
        #     log.error("✅ NETWORK SUCCESS: Reached the machine, but the model took too long to answer.")
        #     return {"error": str(requests.exceptions.ReadTimeout)}
        
        except urllib.error.HTTPError as e:
            log.error(f"❌ SERVER ERROR: Code {e.code} - {e.read().decode('utf-8')}")
            return {"error": str(e), "error_kind": "http", "status_code": e.code,
                    "retryable": e.code in {408, 429, 500, 502, 503, 504}}
        except (urllib.error.URLError, ConnectionError) as e:
            # This handles absolute network dropouts or severe machine connection timeouts
            log.error("NETWORK TIMEOUT/UNREACHABLE: %s", e)
            return {"error": str(e), "error_kind": "network", "retryable": True}
        except TimeoutError:
            log.error("❌ READ TIMEOUT: Reached the machine but processing took over 60 seconds.")
            return {"error": "HTTP request timed out", "error_kind": "timeout", "retryable": True}
            
        except Exception as exc:  # noqa: BLE001
            log.error("HTTP call failed for %s: %s", self.base_url, exc)
            return {"error": str(exc)}

    async def call_async(
        self, payload: dict, callback=None, files=None, timeout: float | None = None
    ) -> dict:
        """Make an asynchronous HTTP POST request."""
        try:
            import httpx

            request_timeout = timeout or self.config.get("timeout", 30.0)
            merged_headers = self._build_headers()

            log.debug("HTTPModule.call_async(%s)", self.base_url)
            if files is not None:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        self.base_url,
                        files=files,
                        data=payload,
                        timeout=request_timeout,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    if callback:
                        callback(result)
                    return result
            else:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        self.base_url,
                        json=payload,
                        headers=merged_headers,
                        timeout=request_timeout,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    if callback:
                        callback(result)
                    return result
        except ImportError:
            log.error("httpx library not installed; cannot make async HTTP call")
            error_result = {"error": "httpx not installed"}
            if callback:
                callback(error_result)
            return error_result
        except Exception as exc:  # noqa: BLE001
            log.error("Async HTTP call failed for %s: %s", self.base_url, exc)
            error_result = {"error": str(exc)}
            if callback:
                callback(error_result)
            return error_result

    def _build_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.config.get("headers", {}))
        api_key = self.config.get("api_key")
        if not api_key and self.config.get("api_key_env"):
            api_key = os.getenv(self.config["api_key_env"])
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra:
            headers.update(extra)
        return headers

    def _post_json_urllib(
        self, payload: dict, headers: dict[str, str], timeout: float
    ) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {"error": f"HTTP {exc.code}: {detail}"}
