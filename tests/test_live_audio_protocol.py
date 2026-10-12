import sys

import numpy as np
import pytest

from fanyiji.live_audio_protocol import (
    LiveAudioFrame, LiveAudioTarget, normalize_live_room, normalize_reference_video_url,
    process_loopback_supported,
)


@pytest.mark.parametrize("value,expected", [
    ("https://www.bilibili.com/video/BV1xx411c7mD", "BV1xx411c7mD"),
    (" https://www.bilibili.com/video/BV1xx411c7mD/?p=2&cookie=private#reply ", "BV1xx411c7mD"),
    ("https://www.bilibili.com:443/video/av00123/?from=search", "av123"),
    ("https://WWW.BILIBILI.COM/video/av123", "av123"),
])
def test_reference_video_url_is_canonical_and_has_no_query_or_fragment(value, expected):
    assert normalize_reference_video_url(value) == "https://www.bilibili.com/video/" + expected


@pytest.mark.parametrize("value", [
    None, 1, "", "BV1xx411c7mD", "123", "https://live.bilibili.com/1698225",
    "http://www.bilibili.com/video/av123", "https://bilibili.com/video/av123",
    "https://www.bilibili.com.evil/video/av123", "https://www.bilibili.com:1234/video/av123",
    "https://www.bilibili.com:wrong/video/av123", "https://user:private@www.bilibili.com/video/av123",
    "https://www.bilibili.com/video/av0", "https://www.bilibili.com/video/av-1",
    "https://www.bilibili.com/video/av1234567890123", "https://www.bilibili.com/video/av１２３",
    "https://www.bilibili.com/video/BV1xx411c7m", "https://www.bilibili.com/video/BV1xx411c7mDD",
    "https://www.bilibili.com/video/bv1xx411c7mD", "https://www.bilibili.com/video/BV1xx411c7mＤ",
    "https://www.bilibili.com/video/av123/extra", "https://www.bilibili.com/video/%61v123",
    "https://www.bilibili.com/video/av123\t?cookie=private",
    "https://www.bilibili.com\\@evil.test/video/av123",
    "https://www.bilibili.com/video/av123?" + "x" * 2048,
])
def test_reference_video_url_rejects_live_pages_and_unsafe_or_non_video_urls(value):
    with pytest.raises(ValueError, match="^invalid_reference_video_url$"):
        normalize_reference_video_url(value)


@pytest.mark.parametrize("value,expected", [
    ("1698225", "1698225"), ("001698225", "1698225"),
    (" https://live.bilibili.com/1698225/?from=search ", "1698225"),
])
def test_live_room_canonical_identity(value, expected):
    assert normalize_live_room(value) == expected


@pytest.mark.parametrize("value", [None, 1, "", "0", "-1", "1/2", "１２３", "1#fragment",
    "https://evil.example/1698225", "https://live.bilibili.com.evil/123", "10000000000000"])
def test_live_room_rejects_non_room_navigation(value):
    with pytest.raises(ValueError, match="invalid_live_room"):
        normalize_live_room(value)


@pytest.mark.parametrize("platform,build,expected", [
    ("win32", 20347, False), ("win32", 20348, True), ("win32", 26300, True),
    ("linux", 26300, False), ("darwin", 26300, False), ("win32", True, False),
])
def test_supported_process_capture_requires_supported_windows(platform, build, expected):
    assert process_loopback_supported(platform, build) is expected


@pytest.mark.parametrize("pid,generation", [(0, 1), (-1, 1), (True, 1), (2**32, 1),
    (123, -1), (123, True), (123, "1")])
def test_source_identity_is_strict(pid, generation):
    with pytest.raises(ValueError):
        LiveAudioTarget(pid, "1698225", generation)


def test_video_audio_has_no_fabricated_live_room():
    assert LiveAudioTarget(1234, "", 7).room == ""


@pytest.mark.parametrize("room", [None, 1, False, [], " ", "0", "video", "https://evil.example/123"])
def test_capture_target_rejects_malformed_room_except_explicit_video_empty_room(room):
    with pytest.raises(ValueError):
        LiveAudioTarget(1234, room, 7)


def test_audio_frame_contract_and_shared_data_are_readonly():
    frame = LiveAudioFrame("1698225", 7, 0, np.zeros(512, dtype=np.float32), 10, 10.032,
                           10.05, 0, 512)
    assert frame.sample_rate == 16000
    assert frame.audio_end_monotonic - frame.audio_start_monotonic == pytest.approx(.032)
    with pytest.raises(ValueError):
        frame.samples[0] = 1


@pytest.mark.parametrize("samples", [np.zeros(10, dtype=np.float64),
    np.zeros((10, 1), dtype=np.float32), np.array([np.nan], dtype=np.float32),
    np.empty(0, dtype=np.float32)])
def test_invalid_pcm_rejected(samples):
    with pytest.raises(ValueError, match="invalid_live_pcm"):
        LiveAudioFrame("1698225", 7, 0, samples, 10, 11, 11, 0, samples.size)


def test_invalid_timestamp_and_sample_range_rejected():
    samples = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError, match="invalid_live_timing"):
        LiveAudioFrame("1698225", 7, 0, samples, 12, 11, 11, 0, 10)
    with pytest.raises(ValueError, match="invalid_live_sample_range"):
        LiveAudioFrame("1698225", 7, 0, samples, 10, 11, 11, 0, 11)
