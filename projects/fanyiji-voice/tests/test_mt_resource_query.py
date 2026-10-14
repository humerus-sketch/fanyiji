"""No GPU query/inference: byte-safe Windows CLI output regression only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mt_resource_monitor", ROOT / "tools/monitor_translation_evaluation.py")
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


@pytest.mark.parametrize("memory_field", ("used_memory", "used_gpu_memory"))
def test_owned_compute_query_omits_names_preserves_na_and_accepts_ansi_stderr(monkeypatch, memory_field):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=b"123, N/A\r\n456, 100\r\n",
                                           stderr="中文诊断".encode("gbk")))
    monkeypatch.setattr(MONITOR.subprocess, "run", run)
    result = MONITOR.gpu_query(f"--query-compute-apps=pid,process_name,{memory_field}", 123)
    assert result["query"] == f"--query-compute-apps=pid,{memory_field}"
    assert result["stdout"] == "123, N/A"
    assert "text" not in run.call_args.kwargs
    assert isinstance(result["stderr"], str)


def test_invalid_utf8_bytes_do_not_raise_or_create_a_readerthread_decoder(monkeypatch):
    monkeypatch.setattr(MONITOR.subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=0, stdout=b"GPU, \xb1\xb3\xbe\xb0", stderr=b"")))
    result = MONITOR.gpu_query("--query-gpu=name,memory.used")
    assert result["returncode"] == 0
    assert result["stdout"].startswith("GPU,")
