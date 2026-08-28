"""Standard-library client adapters for the GPU runtime control plane."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .segmented_ttt import LearnRequest, LearnResult


class GpuHttpClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("HTTP timeout must be finite and positive")

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GPU runtime HTTP {error.code}: {detail}") from error
        if not isinstance(payload, dict):
            raise RuntimeError("GPU runtime returned a non-object response")
        return payload

    def create_session(self, session_id: str, *, theorem_id: str | None = None,
                       experience_id: str | None = None, experience_weights_sha256: str | None = None,
                       experience_snapshot_sha256: str | None = None,
                       model_release_sha256: str | None = None, role: str | None = None,
                       expected_initialization_contract_sha256: str | None = None) -> dict[str, Any]:
        body = {}
        if theorem_id is not None:
            body["theorem_id"] = theorem_id
        if experience_id is not None:
            body["experience_id"] = experience_id
        if experience_weights_sha256 is not None:
            body["experience_weights_sha256"] = experience_weights_sha256
        if experience_snapshot_sha256 is not None:
            body["experience_snapshot_sha256"] = experience_snapshot_sha256
        if model_release_sha256 is not None:
            body["model_release_sha256"] = model_release_sha256
        if role is not None:
            body["role"] = role
        if expected_initialization_contract_sha256 is not None:
            body["expected_initialization_contract_sha256"] = expected_initialization_contract_sha256
        return self._request("POST", f"/sessions/{session_id}", body)

    def snapshot(self, session_id: str, name: str, *, for_experience: bool = False) -> dict[str, Any]:
        body = {"name": name}
        if for_experience:
            body["for_experience"] = True
        return self._request("POST", f"/sessions/{session_id}/snapshot/v1", body)

    def publish_experience(self, session_id: str, snapshot: str, experience_id: str,
                           acceptance: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/sessions/{session_id}/experience/v1",
            {"snapshot": snapshot, "experience_id": experience_id, "acceptance": acceptance})

    def restore(self, session_id: str, name: str) -> dict[str, Any]:
        return self._request("POST", f"/sessions/{session_id}/restore/v1", {"name": name})

    def delete_session(self, session_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/sessions/{session_id}")

    def retire_session(self, session_id: str, name: str, *, expected_policy_version: int,
                       reuse_snapshot: bool = False) -> dict[str, Any]:
        """Submit once; a transport error never authorizes a second retirement."""
        if type(reuse_snapshot) is not bool:
            raise ValueError("reuse_snapshot must be explicitly boolean")
        body = {"name": name, "expected_policy_version": expected_policy_version}
        if reuse_snapshot:
            body["reuse_snapshot"] = True
        return self._request("POST", f"/sessions/{session_id}/retire/v1", body)

    def retirement_receipt(self, session_id: str) -> dict[str, Any]:
        """Read one session's receipt; missing/prepared never proves release."""
        return self._request("GET", f"/sessions/{session_id}/retire/v1")

    def learn(self, request: LearnRequest) -> LearnResult:
        version = request.policy_version
        acknowledged: list[str] = []
        details: list[dict[str, Any]] = []
        for event in request.events:
            payload = self._request(
                "POST",
                f"/sessions/{request.session_id}/learn/v1",
                {
                    "expected_policy_version": version,
                    "event": asdict(event),
                },
            )
            returned_id = payload.get("event_id")
            if returned_id != event.event_id:
                raise RuntimeError(f"GPU runtime acknowledged wrong event: {returned_id!r}")
            returned_version = payload.get("policy_version")
            if not isinstance(returned_version, int) or returned_version <= version:
                raise RuntimeError("GPU runtime did not advance policy_version")
            version = returned_version
            acknowledged.append(returned_id)
            details.append(payload)
        return LearnResult(
            policy_version=version,
            event_ids=tuple(acknowledged),
            metadata={"receipts": details},
        )
