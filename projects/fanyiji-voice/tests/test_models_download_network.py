"""Bounded retries and safe diagnoses; synthetic bytes only, no network."""
from dataclasses import replace
import errno
import hashlib
from http.client import IncompleteRead
import io
import socket
import ssl
import threading
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from fanyiji.models_download import (
    DownloadCancelled, DownloadError, ModelCatalog, ModelDownloader,
    ModelFile, describe_download_error,
)


HF = "https://huggingface.co/test/repo/resolve/012345/file.gguf"
DOMESTIC = "https://modelscope.cn/models/test/repo/resolve/abcdef/file.gguf"


class Response(io.BytesIO):
    def __init__(self, data, *, url=HF, status=200, headers=None):
        super().__init__(data)
        self.url, self.status = url, status
        self.headers = headers or {"Content-Length": str(len(data))}

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def open(self, request, timeout):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def file(data=b"abcdef", *, fallback=True):
    return ModelFile("translation/file.gguf", HF, len(data), hashlib.sha256(data).hexdigest(),
                     (DOMESTIC,) if fallback else ())


def http_error(status, url=DOMESTIC):
    return HTTPError(url, status, "untrusted response token=never-display", {}, io.BytesIO())


def downloader(root, opener, **kwargs):
    return ModelDownloader(root, opener=opener, retry_delay=0, **kwargs)


def test_only_verified_domestic_catalog_files_and_original_hf_link_stay_pinned():
    catalogue = ModelCatalog.default()
    expected = {
        "hy-mt2-7b-q6_k": ("Tencent-Hunyuan/Hy-MT2-7B-GGUF", "0ea6c87047cb05b5bc10cd51ee484e2ed0f6cb5f"),
        "hy-mt2-1.8b-q6_k": ("Tencent-Hunyuan/Hy-MT2-1.8B-GGUF", "ef1d40b8b315575d30d1eb6579996130b8fc0bb2"),
        "qwen3-1.7b-q8_0": ("Qwen/Qwen3-1.7B-GGUF", "dc80e1956e7551cd4aa5309c914e767b69188639"),
    }
    for entry in catalogue.entries:
        for asset in entry.files:
            if entry.id in expected:
                repo, revision = expected[entry.id]
                assert asset.url.startswith("https://huggingface.co/")
                assert asset.fallback_urls == (f"https://modelscope.cn/models/{repo}/resolve/{revision}/{asset.relative_path.split('/')[-1]}",)
            else:
                assert asset.fallback_urls == ()


def test_domestic_is_preferred_but_url_and_final_sha_stay_compatible(tmp_path):
    asset = file()
    opener = Opener(Response(b"abcdef", url=DOMESTIC))
    assert downloader(tmp_path, opener).download_file(asset).read_bytes() == b"abcdef"
    assert opener.requests[0].full_url == DOMESTIC
    assert asset.url == HF


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_source_error_switches_once_to_same_hash_source(tmp_path, status):
    opener = Opener(http_error(status), Response(b"abcdef"))
    phases = []
    result = downloader(tmp_path, opener).download_file(file(), lambda *value: phases.append(value))
    assert result.read_bytes() == b"abcdef"
    assert [request.full_url for request in opener.requests] == [DOMESTIC, HF]
    assert ("source_switching", 0, 6) in phases


def test_two_permanent_source_errors_stop_without_repeating_either(tmp_path):
    opener = Opener(http_error(404), http_error(403, HF))
    with pytest.raises(DownloadError, match="HTTP 403") as error:
        downloader(tmp_path, opener).download_file(file())
    assert len(opener.requests) == 2
    assert error.value.http_status == 403 and error.value.host == "huggingface.co"


@pytest.mark.parametrize("status", [400, 407, 416])
def test_non_recoverable_http_failure_is_not_retried(tmp_path, status):
    opener = Opener(http_error(status))
    with pytest.raises(DownloadError, match=f"HTTP {status}"):
        downloader(tmp_path, opener).download_file(file())
    assert len(opener.requests) == 1


@pytest.mark.parametrize("exception", [URLError("connection refused"), TimeoutError(),
                                      URLError(socket.gaierror(-2, "cannot resolve")),
                                      ConnectionResetError(), IncompleteRead(b"", 4),
                                      http_error(408), http_error(429), http_error(503)])
def test_transient_open_failure_switches_and_recovers(tmp_path, exception):
    opener = Opener(exception, Response(b"abcdef"))
    assert downloader(tmp_path, opener).download_file(file()).read_bytes() == b"abcdef"
    assert [request.full_url for request in opener.requests] == [DOMESTIC, HF]


def test_maximum_four_attempts_even_when_network_never_recovers(tmp_path):
    opener = Opener(*(TimeoutError() for _ in range(4)))
    with pytest.raises(DownloadError, match="已尝试 4 次"):
        downloader(tmp_path, opener).download_file(file())
    assert [request.full_url for request in opener.requests] == [DOMESTIC, HF, DOMESTIC, HF]


def test_single_source_retries_and_re_resolves_original_url(tmp_path):
    opener = Opener(TimeoutError(), Response(b"abcdef"))
    phases = []
    assert downloader(tmp_path, opener).download_file(file(fallback=False), lambda *event: phases.append(event)).exists()
    assert [request.full_url for request in opener.requests] == [HF, HF]
    assert ("retrying", 0, 6) in phases


def test_expired_delivery_signature_re_resolves_once_and_then_switches(tmp_path):
    signed = "https://cdn-lfs-cn-1.modelscope.cn/asset?Signature=top-secret&Expires=123"
    opener = Opener(http_error(403, signed), http_error(403, signed), Response(b"abcdef"))
    assert downloader(tmp_path, opener).download_file(file()).read_bytes() == b"abcdef"
    assert [request.full_url for request in opener.requests] == [DOMESTIC, DOMESTIC, HF]
    assert all("Signature" not in request.full_url for request in opener.requests)


def test_cdn_signature_failures_without_fallback_stop_after_one_refresh(tmp_path):
    signed = "https://us.aws.cdn.hf.co/asset?Signature=top-secret"
    opener = Opener(http_error(403, signed), http_error(403, signed))
    with pytest.raises(DownloadError, match="HTTP 403") as result:
        downloader(tmp_path, opener).download_file(file(fallback=False))
    assert len(opener.requests) == 2
    assert "Signature" not in str(result.value) and "top-secret" not in str(result.value)


def test_short_eof_resumes_from_bytes_already_written_across_sources(tmp_path):
    opener = Opener(Response(b"abc", url=DOMESTIC, headers={"Content-Length": "6"}),
                    Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6", "Content-Length": "3"}))
    assert downloader(tmp_path, opener).download_file(file()).read_bytes() == b"abcdef"
    assert opener.requests[1].get_header("Range") == "bytes=3-"


def test_read_timeout_keeps_written_prefix_and_resumes_safely(tmp_path):
    class StallingResponse(Response):
        def read1(self, size=-1):
            if self.tell() == 0:
                return self.read(2)
            raise TimeoutError("untrusted signed download URL")
    opener = Opener(StallingResponse(b"ab", url=DOMESTIC, headers={"Content-Length": "6"}),
                    Response(b"cdef", status=206, headers={"Content-Range": "bytes 2-5/6", "Content-Length": "4"}))
    assert downloader(tmp_path, opener).download_file(file()).read_bytes() == b"abcdef"
    assert opener.requests[1].get_header("Range") == "bytes=2-"


def test_hash_mismatch_is_permanent_and_does_not_switch_sources(tmp_path):
    opener = Opener(Response(b"xxxxxx", url=DOMESTIC))
    with pytest.raises(DownloadError, match="SHA-256"):
        downloader(tmp_path, opener).download_file(file())
    assert len(opener.requests) == 1
    assert (tmp_path / "translation/file.gguf.part").read_bytes() == b"xxxxxx"


def test_invalid_range_after_network_switch_does_not_change_prefix(tmp_path):
    partial = tmp_path / "translation/file.gguf.part"
    partial.parent.mkdir()
    partial.write_bytes(b"abc")
    opener = Opener(TimeoutError(), Response(b"def", status=206, headers={"Content-Range": "bytes 2-4/6"}))
    with pytest.raises(DownloadError, match="续传范围"):
        downloader(tmp_path, opener).download_file(file())
    assert len(opener.requests) == 2 and partial.read_bytes() == b"abc"


def test_cancel_interrupts_backoff_before_second_network_attempt(tmp_path):
    cancel = threading.Event()
    opener = Opener(TimeoutError())
    def progress(phase, *_):
        if phase == "source_switching":
            cancel.set()
    with pytest.raises(DownloadCancelled):
        ModelDownloader(tmp_path, opener=opener, retry_delay=8).download_file(file(), progress, cancel)
    assert len(opener.requests) == 1


def test_cancellation_during_network_failure_is_not_classified_as_retry(tmp_path):
    cancel = threading.Event()
    class CancelOpener(Opener):
        def open(self, request, timeout):
            cancel.set()
            return super().open(request, timeout)
    opener = CancelOpener(TimeoutError())
    with pytest.raises(DownloadCancelled):
        downloader(tmp_path, opener).download_file(file(), cancel=cancel)
    assert len(opener.requests) == 1


def test_disk_space_failure_prevents_network_request(tmp_path, monkeypatch):
    monkeypatch.setattr("fanyiji.models_download.shutil.disk_usage", lambda path: SimpleNamespace(free=5))
    opener = Opener()
    with pytest.raises(DownloadError, match="磁盘空间不足"):
        downloader(tmp_path, opener).download_file(file())
    assert not opener.requests and not (tmp_path / "translation/file.gguf.part").exists()


def test_space_check_requires_only_remaining_bytes_for_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("fanyiji.models_download.shutil.disk_usage", lambda path: SimpleNamespace(free=3))
    partial = tmp_path / "translation/file.gguf.part"
    partial.parent.mkdir()
    partial.write_bytes(b"abc")
    opener = Opener(Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6", "Content-Length": "3"}))
    assert downloader(tmp_path, opener).download_file(file()).read_bytes() == b"abcdef"


def test_certificate_failure_is_not_retried_or_tls_verification_disabled(tmp_path):
    opener = Opener(URLError(ssl.SSLCertVerificationError(1, "unsafe certificate")))
    with pytest.raises(DownloadError, match="TLS 证书验证失败"):
        downloader(tmp_path, opener).download_file(file())
    assert len(opener.requests) == 1


@pytest.mark.parametrize("exception, expected", [
    (URLError(ssl.SSLCertVerificationError(1, "sensitive URL")), "TLS 证书"),
    (URLError(ssl.SSLError(1, "sensitive URL")), "TLS 安全"),
    (URLError(TimeoutError("sensitive URL")), "超时"),
    (URLError(socket.gaierror(-2, "sensitive URL")), "DNS"),
    (URLError("https://user:password@proxy.test?Signature=top-secret"), "网络连接"),
    (http_error(407, "https://user:password@proxy.test?Signature=top-secret"), "代理身份验证"),
    (OSError(errno.ENOSPC, "sensitive local path"), "磁盘空间"),
    (OSError(errno.EACCES, "sensitive local path"), "权限"),
    (RuntimeError("https://user:password@proxy.test?Signature=top-secret"), "模型下载失败"),
])
def test_safe_error_diagnoses_never_show_untrusted_exception_messages(exception, expected):
    message = describe_download_error(exception)
    assert expected in message
    assert all(value not in message for value in ("https://", "password@", "top-secret", "sensitive"))


def test_safe_host_is_displayed_without_signed_url_query():
    message = describe_download_error(DownloadError("读取超时", host="us.aws.cdn.hf.co"))
    assert message == "读取超时（下载主机：us.aws.cdn.hf.co）"
    for host in ("evil.test", "user:password@huggingface.co", "huggingface.co?Signature=top-secret"):
        assert describe_download_error(DownloadError("读取超时", host=host)) == "读取超时"


def test_verified_existing_file_does_not_require_more_disk_space(tmp_path, monkeypatch):
    asset = file()
    target = tmp_path / asset.relative_path
    target.parent.mkdir()
    target.write_bytes(b"abcdef")
    monkeypatch.setattr("fanyiji.models_download.shutil.disk_usage", lambda path: SimpleNamespace(free=0))
    opener = Opener()
    assert downloader(tmp_path, opener).download_file(asset) == target
    assert not opener.requests


def test_disk_permission_failure_does_not_retry_or_remove_partial(tmp_path, monkeypatch):
    original_open = type(tmp_path).open
    partial = tmp_path / "translation/file.gguf.part"
    partial.parent.mkdir()
    partial.write_bytes(b"abc")
    def open_path(path, mode="r", *args, **kwargs):
        if path == partial and mode == "ab":
            raise PermissionError(errno.EACCES, "private path must not leak")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(type(tmp_path), "open", open_path)
    opener = Opener(Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6", "Content-Length": "3"}))
    with pytest.raises(DownloadError, match="权限") as error:
        downloader(tmp_path, opener).download_file(file())
    assert "private path" not in str(error.value)
    assert len(opener.requests) == 1 and partial.read_bytes() == b"abc"


def local_error_cases():
    cases = [
        (PermissionError(errno.EACCES, "private path"), "权限"),
        (OSError(errno.EPERM, "private path"), "权限"),
        (OSError(errno.EROFS, "private path"), "权限"),
        (OSError(errno.ENOSPC, "private path"), "磁盘空间"),
    ]
    for code, message in ((112, "磁盘空间"), (5, "权限"), (32, "权限")):
        error = OSError("private Windows path")
        error.winerror = code
        cases.append((error, message))
    return cases + [(URLError(error), message) for error, message in cases]


@pytest.mark.parametrize("layer", ["open", "read"])
@pytest.mark.parametrize("exception, expected", local_error_cases())
def test_filesystem_errors_in_transport_layers_are_permanent_and_safely_classified(tmp_path, layer, exception, expected):
    class FailedRead(Response):
        def read1(self, size=-1):
            if self.tell() == 0:
                return self.read(3)
            raise exception
    reply = exception if layer == "open" else FailedRead(b"abc", headers={"Content-Length": "6"})
    opener = Opener(reply)
    phases = []
    with pytest.raises(DownloadError, match=expected) as result:
        downloader(tmp_path, opener).download_file(file(), lambda *value: phases.append(value))
    assert not result.value.retryable
    assert "private" not in str(result.value)
    assert len(opener.requests) == 1
    assert not any(phase[0] in ("retrying", "source_switching") for phase in phases)
    if layer == "read":
        assert (tmp_path / "translation/file.gguf.part").read_bytes() == b"abc"
    assert not (tmp_path / "translation/file.gguf").exists()


def test_local_fsync_failure_keeps_complete_partial_and_does_not_retry(tmp_path, monkeypatch):
    opener = Opener(Response(b"abcdef"))
    def no_space(_):
        raise OSError(errno.ENOSPC, "private local path")
    monkeypatch.setattr("fanyiji.models_download.os.fsync", no_space)
    with pytest.raises(DownloadError, match="磁盘空间"):
        downloader(tmp_path, opener).download_file(file())
    assert len(opener.requests) == 1
    assert (tmp_path / "translation/file.gguf.part").read_bytes() == b"abcdef"
    assert not (tmp_path / "translation/file.gguf").exists()


@pytest.mark.parametrize("exception", [OSError(errno.ENETUNREACH, "untrusted endpoint"),
                                      OSError(errno.ECONNREFUSED, "untrusted endpoint"),
                                      URLError(OSError(errno.EHOSTUNREACH, "untrusted endpoint"))])
def test_known_socket_errno_remains_retryable_with_network_diagnosis(tmp_path, exception):
    assert "网络连接" in describe_download_error(exception)
    opener = Opener(exception, Response(b"abcdef"))
    assert downloader(tmp_path, opener).download_file(file()).read_bytes() == b"abcdef"
    assert len(opener.requests) == 2


@pytest.mark.parametrize("exception", [OSError(errno.EIO, "private path"),
                                      URLError(OSError(errno.EIO, "private path"))])
def test_unknown_os_error_is_not_blindly_retried(tmp_path, exception):
    opener = Opener(exception)
    with pytest.raises(DownloadError) as result:
        downloader(tmp_path, opener).download_file(file())
    assert not result.value.retryable and len(opener.requests) == 1


def test_legacy_download_error_text_redacts_embedded_urls_and_credentials():
    message = describe_download_error(DownloadError("错误 https://user:pass@hf.co?Signature=secret token=hidden-secret"))
    assert "下载地址已隐藏" in message and "凭证已隐藏" in message
    assert "hidden-secret" not in message and "Signature" not in message and "user:pass" not in message


@pytest.mark.parametrize("fallback", [("http://modelscope.cn/file",), ("https://evil.test/file",),
                                     ("https://user:password@modelscope.cn/file",)])
def test_fallback_urls_still_use_existing_official_https_allowlist(fallback):
    with pytest.raises(DownloadError):
        replace(file(), fallback_urls=fallback)


def test_local_model_cannot_gain_download_urls_from_fallback():
    with pytest.raises(ValueError):
        replace(file(), url=None)


@pytest.mark.parametrize("kwargs", [{"max_attempts": 0}, {"max_attempts": 5}, {"max_attempts": True}, {"retry_delay": -1}])
def test_retry_configuration_is_bounded(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ModelDownloader(tmp_path, **kwargs)
