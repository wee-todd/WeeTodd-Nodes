"""Exercise real helper processes without models, network, or credentials."""

import os
import sys
import time

import pytest

from wee_todd_remote.client import invoke_helper


@pytest.fixture
def helper(tmp_path):
    def make(body):
        executable = tmp_path / "helper"
        executable.write_text(
            f"#!{sys.executable}\nimport json, os, sys, time\n"
            "request = json.load(sys.stdin)\n" + body
        )
        executable.chmod(0o700)
        return executable

    return make


def run(helper, **kwargs):
    return list(
        invoke_helper(
            "generate",
            {"requestID": "test-1"},
            helper=helper,
            cancelled=kwargs.pop("cancelled", lambda: False),
            **kwargs,
        )
    )


def test_stdin_request_and_fragmented_events(helper):
    executable = helper(
        "assert sys.argv[1:] == ['generate']\n"
        "assert request['requestID'] == 'test-1'\n"
        "line = json.dumps({'type':'result','requestID':'test-1','manifest':'result.json'})\n"
        "for char in line: os.write(1, char.encode())\n"
    )
    assert run(executable) == [{"type": "result", "requestID": "test-1", "manifest": "result.json"}]


@pytest.mark.parametrize(
    "line,match",
    [
        ('{"type":"result","requestID":"another"}', "requestID"),
        ("broken-json", "JSON"),
        ('{"type":"mystery","requestID":"test-1"}', "type"),
    ],
)
def test_invalid_protocol_is_rejected(helper, line, match):
    with pytest.raises(ValueError, match=match):
        run(helper(f"print({line!r}, flush=True)\n"))


def test_nonzero_exit_never_publishes_success(helper):
    executable = helper(
        "print(json.dumps({'type':'result','requestID':'test-1'}), flush=True)\nsys.exit(7)\n"
    )
    events = []
    with pytest.raises(RuntimeError, match="status 7"):
        events.extend(
            invoke_helper(
                "generate", {"requestID": "test-1"}, helper=executable, cancelled=lambda: False
            )
        )
    assert events == []


def test_stderr_flood_does_not_deadlock_or_leak(helper):
    executable = helper(
        "sys.stderr.write('private-token-' * 200000)\n"
        "print(json.dumps({'type':'result','requestID':'test-1'}))\n"
    )
    assert run(executable)[0]["type"] == "result"


def test_stderr_is_not_used_as_user_error(helper):
    with pytest.raises(RuntimeError) as error:
        run(helper("sys.stderr.write('Bearer private-token')\nsys.exit(1)\n"))
    assert "private-token" not in str(error.value)


def test_cancel_stops_a_silent_child(helper, tmp_path):
    pid_file = tmp_path / "pid"
    executable = helper(f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(30)\n")
    started = time.monotonic()
    with pytest.raises(InterruptedError, match="cancel"):
        run(executable, cancelled=lambda: pid_file.exists())
    assert time.monotonic() - started < 5
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_missing_result_fails_even_with_zero_exit(helper):
    with pytest.raises(RuntimeError, match="result"):
        run(helper("pass\n"))


def test_timeout_stops_silent_child(helper):
    with pytest.raises(TimeoutError):
        run(helper("time.sleep(30)\n"), timeout=0.1)


def test_oversized_record_is_rejected(helper):
    with pytest.raises(ValueError, match="large"):
        run(helper("sys.stdout.write('x' * 2000000)\nsys.stdout.flush()\n"))


def test_duplicate_result_is_rejected(helper):
    with pytest.raises(ValueError, match="result"):
        run(
            helper(
                "print(json.dumps({'type':'result','requestID':'test-1'}) + '\\n' + "
                "json.dumps({'type':'result','requestID':'test-1'}))\n"
            )
        )


def test_unknown_command_does_not_start_helper(helper, tmp_path):
    marker = tmp_path / "executed"
    executable = helper(f"open({str(marker)!r}, 'w').write('yes')\n")
    with pytest.raises(ValueError, match="command"):
        list(
            invoke_helper(
                "delete", {"requestID": "test-1"}, helper=executable, cancelled=lambda: False
            )
        )
    assert not marker.exists()


def test_cleanup_permission_error_after_child_exit_preserves_helper_error(helper, monkeypatch):
    executable = helper(
        "print(json.dumps({'type':'error','requestID':'test-1',"
        "'code':'transport_failed'}), flush=True)\n"
        "time.sleep(0.2)\n"
    )
    from wee_todd_remote import client

    real_popen = client.subprocess.Popen
    child = {}

    def tracked_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        child["process"] = process
        return process

    def exited_group_is_not_signalable(pid, signal_number):
        assert pid == child["process"].pid
        child["process"].wait(timeout=1)
        raise PermissionError("simulated macOS EPERM after child exit")

    monkeypatch.setattr(client.subprocess, "Popen", tracked_popen)
    monkeypatch.setattr(client.os, "killpg", exited_group_is_not_signalable)

    with pytest.raises(RuntimeError, match="transport_failed"):
        run(executable)
    assert child["process"].returncode == 0


def test_cancel_after_submission_is_uncertain_and_never_a_clean_cancel(helper):
    from wee_todd_remote.client import SubmissionUncertain

    executable = helper(
        "print(json.dumps({'type':'progress','requestID':'test-1',"
        "'value':{'stage':'submitting'}}), flush=True)\n"
        "time.sleep(30)\n"
    )
    cancel = False
    stream = invoke_helper(
        "generate", {"requestID": "test-1"}, helper=executable, cancelled=lambda: cancel
    )
    assert next(stream)["value"]["stage"] == "submitting"
    cancel = True
    with pytest.raises(SubmissionUncertain, match="submission_uncertain"):
        next(stream)


def test_interrupt_after_submission_is_uncertain(helper):
    from wee_todd_remote.client import SubmissionUncertain

    executable = helper(
        "print(json.dumps({'type':'progress','requestID':'test-1',"
        "'value':{'stage':'submitting'}}), flush=True)\n"
        "time.sleep(30)\n"
    )
    stream = invoke_helper(
        "generate", {"requestID": "test-1"}, helper=executable, cancelled=lambda: False
    )
    next(stream)
    with pytest.raises(SubmissionUncertain):
        stream.throw(KeyboardInterrupt())


@pytest.mark.parametrize("code", ["invalid_media", "submission_uncertain"])
def test_structured_failure_after_submission_is_not_retryable(helper, code):
    from wee_todd_remote.client import SubmissionUncertain

    executable = helper(
        "print(json.dumps({'type':'progress','requestID':'test-1',"
        "'value':{'stage':'submitting'}}), flush=True)\n"
        f"print(json.dumps({{'type':'error','requestID':'test-1','code':{code!r}}}), flush=True)\n"
    )
    with pytest.raises(SubmissionUncertain):
        run(executable)
