"""Protect spoken samples from uncertain or isolated soundtrack decisions."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fanyiji.engines import ModelAssetsError
from fanyiji.speech_filter import CLASS_COUNT, SpeechOnlyFilter


class ScriptedTagger:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.waveforms = []

    def create_stream(self):
        def accept_waveform(*, sample_rate, waveform):
            assert sample_rate == 16000
            self.waveforms.append(waveform.copy())
        return SimpleNamespace(accept_waveform=accept_waveform)

    def compute(self, stream, *, top_k):
        assert top_k == CLASS_COUNT
        values = next(self.scores)
        return [SimpleNamespace(index=index, prob=values.get(index, 0.0)) for index in range(CLASS_COUNT)]


def audio(seconds):
    return np.linspace(-0.4, 0.4, round(seconds * 16000), dtype=np.float32)


def test_short_greeting_preserved_without_loading_or_classifying(tmp_path):
    tagger = ScriptedTagger([])
    original = audio(0.616)
    result = SpeechOnlyFilter(tmp_path, tagger=tagger).process(original)
    np.testing.assert_array_equal(result.samples, original)
    assert not np.shares_memory(result.samples, original)
    assert result.metadata["reason"] == "short_audio_preserved_without_classification"
    assert result.metadata["windows"] == [] and not tagger.waveforms


def test_speech_with_strong_music_and_singing_preserved(tmp_path):
    original = audio(4)
    gate = SpeechOnlyFilter(tmp_path, tagger=ScriptedTagger([{0: .95, 137: .99, 27: .8}] * 3))
    result = gate.process(original)
    assert result.metadata["action"] == "keep"
    np.testing.assert_array_equal(result.samples, original)
    assert result.metadata["masked_duration_seconds"] == 0


def test_confirmed_song_without_spoken_evidence_skips_all_audio(tmp_path):
    original = audio(4)
    preserved_original = original.copy()
    gate = SpeechOnlyFilter(tmp_path, tagger=ScriptedTagger([{0: .01, 27: .55, 255: .8, 137: .9}] * 3))
    result = gate.process(original)
    assert result.skip and result.metadata["action"] == "skip"
    assert result.samples.shape == original.shape and result.samples.dtype == np.float32
    assert not np.any(result.samples)
    np.testing.assert_array_equal(original, preserved_original)


def test_isolated_bad_singing_window_cannot_remove_overlapping_speech(tmp_path):
    original = audio(4)
    tagger = ScriptedTagger([{0: .98}, {0: .17, 27: .58}, {0: .96}])
    result = SpeechOnlyFilter(tmp_path, tagger=tagger).process(original)
    assert [window["decision"] for window in result.metadata["windows"]] == ["keep", "reject", "keep"]
    np.testing.assert_array_equal(result.samples, original)
    assert result.metadata["action"] == "keep"


def test_contiguous_music_gap_masked_with_spoken_boundaries_preserved(tmp_path):
    original = audio(6)
    tagger = ScriptedTagger([{0: .9}, {0: .01, 137: .95}, {0: .01, 137: .95},
                            {0: .01, 137: .95}, {0: .9}])
    result = SpeechOnlyFilter(tmp_path, tagger=tagger).process(original)
    assert result.metadata["action"] == "masked"
    assert result.samples.size == original.size
    np.testing.assert_array_equal(result.samples[:35200], original[:35200])
    np.testing.assert_array_equal(result.samples[60800:], original[60800:])
    assert not np.any(result.samples[35200:60800])
    assert result.metadata["masked_duration_seconds"] == pytest.approx(1.6)


def test_final_window_uses_two_seconds_context_instead_of_partial_tail(tmp_path):
    original = audio(2.3)
    tagger = ScriptedTagger([{0: .01, 137: .95}, {0: .9}])
    result = SpeechOnlyFilter(tmp_path, tagger=tagger).process(original)
    assert [waveform.size for waveform in tagger.waveforms] == [32000, 32000]
    assert [window["start_seconds"] for window in result.metadata["windows"]] == [0, .3]
    # The protected .3 s final-start boundary extends .2 s towards the start;
    # the remaining .1 s gap is too short to mask.
    np.testing.assert_array_equal(result.samples, original)


def test_weak_music_score_without_spoken_evidence_is_preserved_as_uncertain(tmp_path):
    original = audio(2)
    result = SpeechOnlyFilter(tmp_path, tagger=ScriptedTagger([{0: .02, 137: .31}])).process(original)
    assert result.metadata["action"] == "keep"
    assert result.metadata["windows"][0]["reason"] == "speech_or_uncertain"


@pytest.mark.parametrize("samples", [[1.0], np.ones(10, dtype=np.float64),
                                     np.ones((2, 5), dtype=np.float32),
                                     np.array([np.nan], dtype=np.float32),
                                     np.array([np.inf], dtype=np.float32)])
def test_invalid_audio_is_rejected_before_model_use(tmp_path, samples):
    with pytest.raises(ValueError):
        SpeechOnlyFilter(tmp_path, tagger=ScriptedTagger([])).process(samples)


def test_missing_or_invalid_class_output_is_not_silently_treated_as_speech(tmp_path):
    class IncompleteTagger(ScriptedTagger):
        def compute(self, stream, *, top_k):
            return [SimpleNamespace(index=0, prob=.9)]
    with pytest.raises(RuntimeError, match="完整类别"):
        SpeechOnlyFilter(tmp_path, tagger=IncompleteTagger([])).process(audio(2))
    with pytest.raises(RuntimeError, match="无效类别"):
        SpeechOnlyFilter(tmp_path, tagger=ScriptedTagger([{0: float("nan")}])).process(audio(2))


def test_missing_assets_raise_local_model_error(tmp_path):
    with pytest.raises(ModelAssetsError, match="本地声音分类模型"):
        SpeechOnlyFilter(tmp_path).preload()


def test_close_releases_reference_and_rejects_later_processing(tmp_path):
    gate = SpeechOnlyFilter(tmp_path, tagger=ScriptedTagger([]))
    assert gate.backend_info["model_loaded"]
    gate.close()
    gate.close()
    assert not gate.backend_info["model_loaded"]
    with pytest.raises(RuntimeError, match="已关闭"):
        gate.process(audio(.5))


def test_empty_audio_has_explicit_skip_without_model_work(tmp_path):
    result = SpeechOnlyFilter(tmp_path, tagger=ScriptedTagger([])).process(np.array([], dtype=np.float32))
    assert result.skip and result.metadata["reason"] == "empty_audio"
