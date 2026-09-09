"""Stream a separately packaged Draw Things helper without importing an ML runtime."""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

MAX_RECORD_BYTES = 1024 * 1024


class SubmissionUncertain(RuntimeError):
    """A remote job may have been accepted; never automatically retry it."""

    def __init__(self):
        super().__init__(
            "Draw Things submission_uncertain: the server may have accepted this job; "
            "check it before retrying"
        )


def invoke_helper(
    command: str,
    payload: dict,
    *,
    helper: Path,
    cancelled: Callable[[], bool],
    timeout: float = 3600,
) -> Iterator[dict]:
    """Yield progress immediately; publish a result only after a successful process exit.

    Stderr is drained but never forwarded: upstream diagnostics may contain tokens.
    A caller closing this iterator also terminates its request-owned process group.
    """
    from .contracts import validate_event

    if command not in {"capabilities", "estimate", "generate"}:
        raise ValueError("Unsupported Draw Things helper command")
    request_id = payload.get("requestID")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("A requestID is required")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("Timeout must be positive")
    encoded = json.dumps(payload, allow_nan=False).encode() + b"\n"
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("Helper request is too large; pass media by file path")
    if cancelled():
        raise InterruptedError("Draw Things request cancelled")
    executable = Path(helper).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError("Draw Things helper unavailable; install the connection runtime")

    process = subprocess.Popen(
        [str(executable), command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        bufsize=0,
    )
    selector = selectors.DefaultSelector()
    buffer = bytearray()
    pending = memoryview(encoded)
    result = None
    submitted = False
    started = time.monotonic()

    def event_from(raw):
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("Invalid JSON from Draw Things helper") from error
        value = validate_event(value)
        if value["requestID"] != request_id:
            raise ValueError("Mismatched helper requestID")
        return value

    try:
        for stream, mode in (
            (process.stdin, selectors.EVENT_WRITE),
            (process.stdout, selectors.EVENT_READ),
            (process.stderr, selectors.EVENT_READ),
        ):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, mode)
        while selector.get_map() or process.poll() is None:
            if cancelled():
                raise InterruptedError("Draw Things request cancelled")
            if time.monotonic() - started >= timeout:
                raise TimeoutError("Draw Things helper timed out; submission may be incomplete")
            for key, _ in selector.select(timeout=0.05):
                stream = key.fileobj
                if stream is process.stdin:
                    try:
                        size = os.write(stream.fileno(), pending)
                        pending = pending[size:]
                    except BrokenPipeError:
                        pending = memoryview(b"")
                    if not pending:
                        selector.unregister(stream)
                        stream.close()
                    continue
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                if stream is process.stderr:
                    continue
                buffer.extend(chunk)
                while b"\n" in buffer or (not chunk and buffer):
                    if b"\n" in buffer:
                        raw, _, remainder = buffer.partition(b"\n")
                        buffer[:] = remainder
                    else:
                        raw = bytes(buffer)
                        buffer.clear()
                    if len(raw) > MAX_RECORD_BYTES:
                        raise ValueError("Helper response record is too large")
                    if not raw.strip():
                        continue
                    event = event_from(raw)
                    if result is not None:
                        raise ValueError("Unexpected event after helper result")
                    if event["type"] == "result":
                        result = event
                    elif event["type"] == "error":
                        # Structured helper error codes are safe; free text may echo secrets.
                        code = event.get("code", "generation_failed")
                        safe_code = (
                            code if isinstance(code, str) and code.isidentifier() else "failed"
                        )
                        if submitted or safe_code == "submission_uncertain":
                            raise SubmissionUncertain()
                        raise RuntimeError(f"Draw Things request failed ({safe_code})")
                    else:
                        value = event.get("value")
                        if isinstance(value, dict) and value.get("stage") == "submitting":
                            submitted = True
                        yield event
                if len(buffer) > MAX_RECORD_BYTES:
                    raise ValueError("Helper response record is too large")
        status = process.wait()
        if status:
            if submitted:
                raise SubmissionUncertain()
            raise RuntimeError(f"Draw Things helper exited with status {status}")
        if result is None:
            raise RuntimeError("Draw Things helper completed without a result")
        yield result
    except (KeyboardInterrupt, InterruptedError, TimeoutError):
        if submitted:
            raise SubmissionUncertain() from None
        raise
    finally:
        selector.close()

        def stop_owned_group(sig):
            """Signal descendants, falling back to the owned child on active-group EPERM."""
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
            except PermissionError:
                # macOS may report EPERM for a group whose leader exited between
                # our bounded wait and signal. Ignore it only after Popen confirms
                # that the owned child has exited and has been reaped.
                if process.poll() is not None:
                    return
                if sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()

        # Give a helper that is already exiting a bounded chance to be reaped;
        # otherwise terminate its request-owned group and escalate once.
        try:
            process.wait(timeout=0.05)
        except subprocess.TimeoutExpired:
            stop_owned_group(signal.SIGTERM)
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                stop_owned_group(signal.SIGKILL)
                process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if not stream.closed:
                stream.close()
