from __future__ import annotations

import json
import socketserver
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


CAPTURE_MODES = ("manual", "continuous", "interval", "trigger")


class CaptureSchedule:
    """Decide WHEN to trigger; only the main thread ever touches the camera."""

    def __init__(self, mode: str = "manual", interval_seconds: float = 2.0):
        if mode not in CAPTURE_MODES:
            raise ValueError(f"Unknown capture mode: {mode}")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self.mode = mode
        self.interval_seconds = float(interval_seconds)
        self.paused = False
        self.next_due = 0.0
        self.pending: str | None = "startup" if mode != "trigger" else None

    def set_mode(self, mode: str, now: float) -> None:
        if mode not in CAPTURE_MODES:
            raise ValueError(f"Unknown capture mode: {mode}")
        self.mode = mode
        self.paused = False
        self.pending = None
        self.next_due = now

    def request_once(self) -> None:
        # Deliberate single-step works even while automatic scanning is paused.
        self.pending = "keyboard"

    def toggle_pause(self, now: float) -> None:
        self.paused = not self.paused
        self.pending = None
        self.next_due = now + self.interval_seconds

    def take_due(self, now: float) -> str | None:
        if self.pending is not None:
            reason, self.pending = self.pending, None
            return reason
        if self.paused:
            return None
        if self.mode == "continuous":
            return "continuous"
        if self.mode == "interval" and now >= self.next_due:
            return "interval"
        return None

    def completed(self, now: float) -> None:
        # Interval is idle time AFTER processing; never catch up missed scans.
        self.next_due = now + self.interval_seconds


@dataclass
class TriggerRequest:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None


class TriggerBroker:
    """One request maximum, no backlog while acquisition/processing is busy."""

    def __init__(self):
        self._lock = threading.Lock()
        self.enabled = False
        self.pending: TriggerRequest | None = None
        self.active: TriggerRequest | None = None

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = enabled
            if not enabled and self.pending is not None:
                self.pending.result = {"status": "cancelled", "reason": "mode_changed_or_paused"}
                self.pending.done.set()
                self.pending = None

    def submit(self) -> tuple[TriggerRequest | None, str]:
        with self._lock:
            if not self.enabled:
                return None, "not_ready"
            if self.pending is not None or self.active is not None:
                return None, "busy"
            request = TriggerRequest()
            self.pending = request
            return request, "queued"

    def claim(self) -> TriggerRequest | None:
        with self._lock:
            if not self.enabled or self.pending is None:
                return None
            self.active, self.pending = self.pending, None
            return self.active

    def finish(self, request: TriggerRequest, result: dict[str, Any]) -> None:
        with self._lock:
            request.result = result
            request.done.set()
            if self.active is request:
                self.active = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ready" if self.enabled else "not_ready",
                "busy": self.pending is not None or self.active is not None,
            }


class _TriggerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class SoftwareTriggerServer:
    """Local TCP trigger: send TRIGGER\n; receive queued then done/error JSON.

    This is a software signal, not the scanner's electrical trigger input.
    A request is not retried automatically if its client disconnects.
    """

    def __init__(self, broker: TriggerBroker, host: str = "127.0.0.1", port: int = 8765):
        self.broker = broker
        self.stop_event = threading.Event()
        stop_event = self.stop_event

        class Handler(socketserver.StreamRequestHandler):
            def send_json(self, value: dict[str, Any]) -> None:
                self.wfile.write((json.dumps(value, ensure_ascii=False) + "\n").encode())
                self.wfile.flush()

            def handle(self) -> None:
                self.connection.settimeout(3.0)
                try:
                    line = self.rfile.readline(65)
                    if not line.endswith(b"\n") or len(line) > 64:
                        self.send_json({"status": "error", "reason": "send TRIGGER followed by newline"})
                        return
                    command = line.strip().upper()
                    if command == b"STATUS":
                        self.send_json(broker.status())
                        return
                    if command != b"TRIGGER":
                        self.send_json({"status": "error", "reason": "unknown_command"})
                        return
                    request, status = broker.submit()
                    if request is None:
                        self.send_json({"status": status})
                        return
                    self.send_json({"status": "queued", "request_id": request.request_id})
                    while not request.done.wait(0.2):
                        if stop_event.is_set():
                            self.send_json({"status": "cancelled", "reason": "server_stopped"})
                            return
                    self.send_json({"request_id": request.request_id, **(request.result or {})})
                except (OSError, TimeoutError):
                    # The main thread still completes an already accepted scan.
                    return

        self.server = _TriggerServer((host, port), Handler)
        self.address = self.server.server_address
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.1},
            daemon=True,
        )

    def start(self) -> "SoftwareTriggerServer":
        self.thread.start()
        return self

    def close(self) -> None:
        self.broker.set_enabled(False)
        self.stop_event.set()
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=2.0)
        self.server.server_close()

    def __enter__(self) -> "SoftwareTriggerServer":
        return self.start()

    def __exit__(self, *_args) -> None:
        self.close()
