from types import SimpleNamespace
import threading

import numpy as np
import pytest

from fanyiji.audio_processing import AudioProcessor, GTCRN_CONTEXT_SAMPLES, SAMPLE_RATE
from fanyiji.audio_processing_models import processing_model_path
from fanyiji.models_download import VerificationResult


def ready(*args):
    return VerificationResult(True, "ready", "verified")


class Denoiser:
    def __init__(self):
        self.calls = []
    def __call__(self, audio, rate):
        self.calls.append((audio.copy(), rate))
        return SimpleNamespace(sample_rate=rate, samples=audio * np.float32(0.5))


def create(tmp_path, native, **kwargs):
    return AudioProcessor(tmp_path, backend_factory=lambda *args: native, verifier=ready, **kwargs)


@pytest.mark.parametrize("length", [1, 128, 511, 512, 1600, 16017, 480000])
def test_enhancement_preserves_exact_clock_and_does_not_mutate_input(tmp_path, length):
    native = Denoiser()
    processor = create(tmp_path, native)
    source = np.linspace(-0.5, 0.5, length, dtype=np.float32)
    original = source.copy()
    result = processor.process(source)
    assert result.metadata["action"] == "enhanced"
    assert result.metadata["input_samples"] == result.metadata["output_samples"] == length
    assert result.samples.shape == source.shape and result.samples.dtype == np.float32
    np.testing.assert_array_equal(source, original)
    np.testing.assert_allclose(result.samples, original * 0.6, atol=1e-7)
    padded, rate = native.calls[0]
    assert rate == SAMPLE_RATE and padded.flags.c_contiguous
    assert not padded[:GTCRN_CONTEXT_SAMPLES].any()
    assert not padded[-GTCRN_CONTEXT_SAMPLES:].any()
    assert not np.shares_memory(source, result.samples)


def test_frame_tail_rounding_is_cropped_using_original_origin(tmp_path):
    class Framed:
        def __call__(self, audio, rate):
            return SimpleNamespace(sample_rate=rate, samples=audio[:len(audio)//256*256])
    source = np.zeros(16017, np.float32)
    source[15000] = 0.7
    result = create(tmp_path, Framed()).process(source)
    assert result.metadata["action"] == "enhanced"
    np.testing.assert_allclose(result.samples, source)
    assert np.argmax(result.samples) == 15000


@pytest.mark.parametrize("output", [
    SimpleNamespace(sample_rate=48000, samples=np.ones(17024, np.float32)),
    SimpleNamespace(sample_rate=16000, samples=np.array([np.nan], np.float32)),
    SimpleNamespace(sample_rate=16000, samples=np.ones((2, 16000), np.float32)),
    SimpleNamespace(sample_rate=16000, samples=np.ones(10, np.float32)),
    SimpleNamespace(sample_rate=16000, samples=np.ones(19000, np.float32)),
])
def test_invalid_native_output_preserves_original_and_reports_bypass(tmp_path, output):
    source = np.full(16000, 0.25, np.float32)
    result = create(tmp_path, lambda *args: output).process(source)
    assert result.metadata["action"] == "bypass" and result.metadata["reason"] == "invalid_output"
    np.testing.assert_array_equal(result.samples, source)


def test_exception_metadata_does_not_expose_message_paths_or_secrets(tmp_path):
    def fails(*args):
        raise RuntimeError("C:/private/path COOKIE=secret credential")
    result = create(tmp_path, fails).process(np.ones(16000, np.float32))
    assert result.metadata["reason"] == "processing_error"
    assert result.metadata["error_type"] == "RuntimeError"
    assert "secret" not in repr(result.metadata) and "private" not in repr(result.metadata)


@pytest.mark.parametrize("mode", ["off", "gtcrn"])
def test_disabled_empty_and_oversize_audio_never_load_native(tmp_path, mode):
    def forbidden(*args):
        pytest.fail("backend must not load")
    processor = AudioProcessor(tmp_path, mode, backend_factory=forbidden, verifier=forbidden)
    for source in (np.zeros(0, np.float32), np.ones(480001, np.float32)):
        result = processor.process(source)
        np.testing.assert_array_equal(result.samples, source)
        assert result.metadata["action"] == ("off" if mode == "off" else "bypass")


def test_invalid_nonfinite_input_never_reaches_native(tmp_path):
    def forbidden(*args):
        pytest.fail("invalid input must not reach native code")
    source = np.array([np.nan, np.inf, -np.inf, 0.25], np.float32)
    result = AudioProcessor(tmp_path, backend_factory=forbidden, verifier=forbidden).process(source)
    assert result.metadata["reason"] == "invalid_input"
    np.testing.assert_array_equal(result.samples, [0, 0, 0, 0.25])
    assert np.isnan(source[0])


def test_invalid_and_missing_models_are_not_silently_loaded(tmp_path):
    calls = []
    def factory(*args):
        calls.append(args)
        return Denoiser()
    source = np.full(16000, 0.25, np.float32)
    processor = AudioProcessor(tmp_path, backend_factory=factory)
    assert processor.preload()["state"] == "missing"
    assert processor.process(source).metadata["reason"] == "model_missing"
    target = processing_model_path(tmp_path, "gtcrn")
    target.parent.mkdir()
    target.write_bytes(b"invalid")
    result = processor.process(source)
    assert result.metadata["reason"] == "model_invalid"
    assert not calls
    np.testing.assert_array_equal(result.samples, source)


def test_missing_model_can_be_loaded_after_download_without_new_instance(tmp_path):
    native = Denoiser()
    def verification(root, mode):
        exists = processing_model_path(root, mode).exists()
        return VerificationResult(exists, "ready" if exists else "missing", "")
    processor = AudioProcessor(tmp_path, verifier=verification, backend_factory=lambda *args: native)
    assert not processor.preload()["ready"]
    target = processing_model_path(tmp_path, "gtcrn")
    target.parent.mkdir()
    target.write_bytes(b"fake verified weight")
    assert processor.preload()["ready"]
    assert processor.process(np.zeros(16000, np.float32)).metadata["action"] == "enhanced"


def test_load_error_only_retries_when_model_file_changes(tmp_path):
    calls = []
    def fails(*args):
        calls.append(1)
        raise OSError("secret path")
    processor = AudioProcessor(tmp_path, verifier=ready, backend_factory=fails)
    for _ in range(3):
        result = processor.process(np.ones(16000, np.float32))
        assert result.metadata["reason"] == "load_error"
    assert len(calls) == 1


def approximate_resampler(audio, src, dst):
    n = round(len(audio) * dst / src)
    return np.interp(np.arange(n) * src / dst, np.arange(len(audio)), audio).astype(np.float32)


def test_uvr_uses_vocals_first_channels_first_and_restores_16k_exact_length(tmp_path):
    class Separator:
        def process(self, *, samples, sample_rate):
            assert sample_rate == 44100 and samples.shape == (2, 44100)
            assert samples.dtype == np.float32 and samples.flags.c_contiguous
            np.testing.assert_array_equal(samples[0], samples[1])
            return SimpleNamespace(sample_rate=44100, stems=[
                SimpleNamespace(data=np.full((2, 44100), 0.25, np.float32)),
                SimpleNamespace(data=np.full((2, 44100), 0.9, np.float32))])
    source = np.full(16000, 0.5, np.float32)
    result = create(tmp_path, Separator(), mode="uvr", resampler=approximate_resampler).process(source)
    assert result.metadata["action"] == "separated"
    assert result.metadata["sample_rate"] == 16000 and len(result.samples) == 16000
    np.testing.assert_allclose(result.samples, 0.3, atol=1e-7)


@pytest.mark.parametrize("output", [
    SimpleNamespace(sample_rate=16000, stems=[]),
    SimpleNamespace(sample_rate=44100, stems=[SimpleNamespace(data=np.ones((2, 44100)))]),
    SimpleNamespace(sample_rate=44100, stems=[SimpleNamespace(data=np.ones((44100, 2))), None]),
    SimpleNamespace(sample_rate=44100, stems=[SimpleNamespace(data=np.ones((2, 100))), None]),
    SimpleNamespace(sample_rate=44100, stems=[SimpleNamespace(data=np.full((2, 44100), np.inf)), None]),
])
def test_uvr_bad_stems_keep_original(tmp_path, output):
    native = SimpleNamespace(process=lambda **kwargs: output)
    source = np.full(16000, 0.5, np.float32)
    result = create(tmp_path, native, mode="uvr", resampler=approximate_resampler).process(source)
    assert result.metadata["reason"] == "invalid_output"
    np.testing.assert_array_equal(result.samples, source)


def test_uvr_rounding_only_preserves_at_most_two_original_tail_samples(tmp_path):
    class Separator:
        def process(self, *, samples, sample_rate):
            return SimpleNamespace(sample_rate=44100, stems=[SimpleNamespace(data=samples),
                                                               SimpleNamespace(data=samples)])
    def resampler(audio, src, dst):
        restored = approximate_resampler(audio, src, dst)
        return restored[:-1] if dst == 16000 else restored
    source = np.linspace(0, 0.5, 16017, dtype=np.float32)
    result = create(tmp_path, Separator(), mode="uvr", resampler=resampler).process(source)
    assert result.metadata["action"] == "separated" and len(result.samples) == len(source)
    assert result.samples[-1] == source[-1]


def test_independent_calls_do_not_keep_previous_waveform(tmp_path):
    native = Denoiser()
    processor = create(tmp_path, native)
    processor.process(np.ones(16000, np.float32))
    result = processor.process(np.zeros(16000, np.float32))
    assert not result.samples.any()
    assert not native.calls[1][0].any()


def test_close_waits_for_inflight_native_call_and_cannot_reopen(tmp_path):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    class Blocking(Denoiser):
        def __call__(self, audio, rate):
            entered.set()
            assert release.wait(2)
            return super().__call__(audio, rate)
    processor = create(tmp_path, Blocking())
    worker = threading.Thread(target=lambda: processor.process(np.ones(16000, np.float32)))
    worker.start()
    assert entered.wait(2)
    closer = threading.Thread(target=lambda: (processor.close(), closed.set()))
    closer.start()
    assert not closed.wait(0.02)
    release.set()
    worker.join(2)
    closer.join(2)
    assert closed.is_set() and processor.backend_info["state"] == "closed"
    assert processor.preload()["state"] == "closed"
    result = processor.process(np.ones(16000, np.float32))
    assert result.metadata["reason"] == "closed"
    np.testing.assert_array_equal(result.samples, np.ones(16000, np.float32))
