"""Strict source identity and PCM contracts for isolated browser audio capture."""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np

SAMPLE_RATE = 16000
MIN_WINDOWS_BUILD = 20348


def normalize_live_room(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid_live_room")
    value = value.strip()
    if value.startswith("https://live.bilibili.com/"):
        value = value[len("https://live.bilibili.com/"):].split("?", 1)[0].rstrip("/")
    if not re.fullmatch(r"[0-9]{1,12}", value) or int(value) <= 0:
        raise ValueError("invalid_live_room")
    return str(int(value))


def normalize_reference_video_url(value: str) -> str:
    """Recognize only ordinary Bilibili videos; never retain URL credentials or queries."""
    if not isinstance(value, str):
        raise ValueError("invalid_reference_video_url")
    value = value.strip()
    if not value or len(value) > 2048 or any(ord(char) < 33 for char in value):
        raise ValueError("invalid_reference_video_url")
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or parsed.hostname != "www.bilibili.com"
                or parsed.port not in (None, 443) or parsed.username is not None
                or parsed.password is not None):
            raise ValueError("invalid_reference_video_url")
        match = re.fullmatch(r"/video/(BV[A-Za-z0-9]{10}|av[0-9]{1,12})/?", parsed.path)
        if match is None:
            raise ValueError("invalid_reference_video_url")
        video = match[1]
        if video.startswith("av"):
            if int(video[2:]) <= 0:
                raise ValueError("invalid_reference_video_url")
            video = "av" + str(int(video[2:]))
        return "https://www.bilibili.com/video/" + video
    except ValueError:
        raise ValueError("invalid_reference_video_url") from None


def process_loopback_supported(platform: str | None = None, build: int | None = None) -> bool:
    platform = sys.platform if platform is None else platform
    if platform != "win32":
        return False
    if build is None:
        build = sys.getwindowsversion().build
    return type(build) is int and build >= MIN_WINDOWS_BUILD


@dataclass(frozen=True)
class LiveAudioTarget:
    process_id: int
    room: str
    generation: int

    def __post_init__(self) -> None:
        if type(self.process_id) is not int or not 0 < self.process_id <= 0xFFFFFFFF:
            raise ValueError("invalid_live_process")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("invalid_live_generation")
        # Ordinary videos have no live room. Their browser/source generation
        # identifies the stream; a made-up room must never reach subtitles/chat.
        if self.room != "":
            object.__setattr__(self, "room", normalize_live_room(self.room))


@dataclass(frozen=True)
class LiveAudioStatus:
    process_id: int
    room: str
    generation: int
    state: str
    code: str = ""


@dataclass(frozen=True)
class LiveAudioFrame:
    room: str
    generation: int
    sequence: int
    samples: np.ndarray
    audio_start_monotonic: float
    audio_end_monotonic: float
    captured_at: float
    start_sample: int
    end_sample: int
    sample_rate: int = SAMPLE_RATE
    discontinuity: bool = False

    def __post_init__(self) -> None:
        if (self.sample_rate != SAMPLE_RATE or self.samples.dtype != np.float32
                or self.samples.ndim != 1 or not self.samples.size
                or not np.isfinite(self.samples).all()):
            raise ValueError("invalid_live_pcm")
        if (self.end_sample - self.start_sample != self.samples.size
                or self.start_sample < 0 or self.sequence < 0):
            raise ValueError("invalid_live_sample_range")
        if (not all(math.isfinite(value) for value in (
                self.audio_start_monotonic, self.audio_end_monotonic, self.captured_at))
                or self.audio_end_monotonic < self.audio_start_monotonic):
            raise ValueError("invalid_live_timing")
        # Consumers must not mutate PCM shared with diagnostics or the ASR queue.
        self.samples.setflags(write=False)
