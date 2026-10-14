"""CPU inference using fixed local assets; no downloads or audio-device access.

M2M100's actual tokenizer code (not its obsolete MBART docstring) uses
``[__source__] + SentencePiece tokens + [</s>]``. CTranslate2 receives the
target language through ``target_prefix=[[__target__]]``.
References:
https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/m2m_100/tokenization_m2m_100.py
https://opennmt.net/CTranslate2/guides/transformers.html#m2m-100
"""

from __future__ import annotations

import importlib
import json
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .language_policy import AsrResult, normalize_detected_language, validate_asr_language
from .translation_output import ensure_chinese_output, is_chinese_output

if TYPE_CHECKING:
    import numpy as np


SAMPLE_RATE = 16_000
SUPPORTED_LANGUAGES = frozenset({"zh", "ja"})
_SENSEVOICE_TAG = re.compile(r"<\|[^<>\r\n]*\|>")
_LANGUAGE_TOKEN = re.compile(r"__[a-z][a-z0-9_-]*__")
_SPECIAL_TOKENS = frozenset({"<s>", "</s>", "<pad>", "<unk>"})


class ModelAssetsError(RuntimeError):
    """The fixed offline model assets are missing, corrupt, or inconsistent."""


def _require_files(model_dir: Path, filenames: Iterable[str]) -> None:
    invalid = []
    for name in filenames:
        path = model_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            invalid.append(str(path))
    if invalid:
        raise ModelAssetsError(
            "本地模型文件缺失或为空：\n" + "\n".join(invalid)
            + "\n请完成本地模型准备；程序不会联网下载。"
        )


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"无法加载离线推理依赖 {name}：{exc}") from exc


def _language(language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"只支持 zh 或 ja，收到：{language!r}")
    return language


def sanitize_sensevoice(text: str) -> str:
    """Remove SenseVoice metadata while preserving punctuation and word spaces."""
    return _SENSEVOICE_TAG.sub("", text).strip()


class RecognizerEngine:
    """One lazily loaded SenseVoice recognizer per language, with serial access.

    ``samples`` must be mono float32 at 16 kHz. Resampling, segmentation, and
    device capture belong to the caller. Two language instances may allocate
    two ONNX sessions; this class makes no claim of shared model memory.
    """

    def __init__(self, model_dir: str | Path, num_threads: int = 2) -> None:
        if not 1 <= num_threads <= 4:
            raise ValueError("ASR num_threads 必须在 1 到 4 之间")
        self.model_dir = Path(model_dir).resolve()
        self.num_threads = num_threads
        self._recognizers: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._closed = False
        _require_files(self.model_dir, ("model.int8.onnx", "tokens.txt"))

    @property
    def backend_info(self) -> dict[str, Any]:
        return {"backend": "sherpa_onnx", "model_kind": "sense_voice", "model_id": "sensevoice-int8",
                "device": "cpu", "num_threads": self.num_threads,
                "recognizer_instances": len(self._recognizers)}

    def _recognizer(self, language: str) -> Any:
        language = validate_asr_language(language)
        if self._closed:
            raise RuntimeError("本地识别引擎已关闭")
        if language not in self._recognizers:
            model = getattr(self, "_sensevoice_model", self.model_dir / "model.int8.onnx")
            tokens = getattr(self, "_sensevoice_tokens", self.model_dir / "tokens.txt")
            _require_files(self.model_dir, (str(model), str(tokens)))
            sherpa = _dependency("sherpa_onnx")
            try:
                recognizer = sherpa.OfflineRecognizer.from_sense_voice(
                    model=str(model),
                    tokens=str(tokens),
                    num_threads=self.num_threads,
                    sample_rate=SAMPLE_RATE,
                    provider="cpu",
                    language=language,
                    use_itn=True,
                    debug=False,
                )
            except Exception as exc:
                raise ModelAssetsError(f"无法加载 {language} 本地 ASR 模型：{exc}") from exc
            self._recognizers[language] = recognizer
        return self._recognizers[language]

    def preload(self, languages: Iterable[str] = ("zh", "ja")) -> None:
        with self._lock:
            for language in languages:
                self._recognizer(language)

    def close(self) -> None:
        # The no-profile SenseVoice startup path owns native sessions too.
        # Wait for any current CPU call before releasing them, just like the
        # backend.json adapter, and prohibit loading again after shutdown.
        with self._lock:
            self._closed = True
            self._recognizers.clear()

    def recognize(self, samples: np.ndarray, language: str) -> str:
        return self.recognize_detailed(samples, language).text

    def recognize_detailed(self, samples: np.ndarray, language: str) -> AsrResult:
        language = validate_asr_language(language)
        numpy = _dependency("numpy")
        if not isinstance(samples, numpy.ndarray):
            raise TypeError("samples 必须是 numpy float32 数组")
        if samples.ndim != 1 or samples.dtype != numpy.float32:
            raise ValueError("samples 必须是 16 kHz 单声道的一维 float32 数组")
        if not numpy.isfinite(samples).all():
            raise ValueError("samples 包含 NaN 或无穷值")
        if samples.size == 0:
            return AsrResult("")
        waveform = numpy.ascontiguousarray(samples)
        with self._lock:
            recognizer = self._recognizer(language)
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, waveform)
            recognizer.decode_stream(stream)
            detected = normalize_detected_language(getattr(stream.result, "lang", None))
            return AsrResult(sanitize_sensevoice(stream.result.text), detected,
                             "native_result" if detected else "unknown")


class _M2M100Tokenizer:
    """M2M100's SentencePiece path without mutable ``src_lang`` or Transformers.

    CTranslate2 works with token strings. The original vocab still determines
    unknown-piece handling and is checked against CT2's converted vocabulary.
    The SentencePiece processor is used only while the engine lock is held.
    """

    def __init__(self, processor: Any, vocabulary: Any, shared_vocabulary: Any) -> None:
        if not isinstance(vocabulary, dict) or not vocabulary:
            raise ModelAssetsError("vocab.json 必须是非空的 token → id 对象")
        if any(
            not isinstance(token, str) or not isinstance(index, int)
            or isinstance(index, bool) or index < 0
            for token, index in vocabulary.items()
        ):
            raise ModelAssetsError("vocab.json 包含无效的 token 或 id")
        if not isinstance(shared_vocabulary, list) or not all(
            isinstance(token, str) for token in shared_vocabulary
        ):
            raise ModelAssetsError("shared_vocabulary.json 必须是 CT2 token 列表")
        shared_set = set(shared_vocabulary)
        required = _SPECIAL_TOKENS | {"__zh__", "__ja__"}
        if not required <= shared_set or not _SPECIAL_TOKENS <= vocabulary.keys():
            raise ModelAssetsError("M2M100 词表缺少特殊标记或 __zh__/__ja__")
        # Reject mismatched tokenizer/converted-model assets before inference.
        for token, index in vocabulary.items():
            if index >= len(shared_vocabulary) or shared_vocabulary[index] != token:
                raise ModelAssetsError("vocab.json 与 shared_vocabulary.json 不匹配")
        self._processor = processor
        self._vocabulary = vocabulary
        self._special = _SPECIAL_TOKENS | {
            token for token in shared_set if _LANGUAGE_TOKEN.fullmatch(token)
        }

    @classmethod
    def from_directory(cls, model_dir: Path) -> _M2M100Tokenizer:
        sentencepiece = _dependency("sentencepiece")
        try:
            with (model_dir / "vocab.json").open(encoding="utf-8") as stream:
                vocabulary = json.load(stream)
            with (model_dir / "shared_vocabulary.json").open(encoding="utf-8") as stream:
                shared_vocabulary = json.load(stream)
            processor = sentencepiece.SentencePieceProcessor()
            # SentencePiece's native path loader can reject Unicode Windows paths.
            # Python opens the file, then the native loader receives bytes only.
            if not processor.LoadFromSerializedProto(
                (model_dir / "sentencepiece.bpe.model").read_bytes()
            ):
                raise ModelAssetsError("无法加载 sentencepiece.bpe.model")
            return cls(processor, vocabulary, shared_vocabulary)
        except (OSError, ValueError, RuntimeError) as exc:
            if isinstance(exc, ModelAssetsError):
                raise
            raise ModelAssetsError(f"无法加载本地 M2M100 tokenizer：{exc}") from exc

    def encode(self, text: str, language: str) -> list[str]:
        language = _language(language)
        # ASR content is plain text, never a sequence of tokenizer control tokens.
        if any(token in text for token in self._special):
            raise ValueError("待翻译文本含有模型控制标记")
        pieces = self._processor.encode(text, out_type=str)
        pieces = [piece if piece in self._vocabulary else "<unk>" for piece in pieces]
        return [f"__{language}__", *pieces, "</s>"]

    def decode(self, tokens: Iterable[str]) -> str:
        # CT2 includes the forced target prefix in its hypothesis. Strip language
        # markers explicitly, then decode the lexical pieces with SentencePiece.
        pieces = [
            token for token in tokens
            if token not in self._special and token in self._vocabulary
        ]
        return self._processor.decode(pieces).strip()


class TranslationEngine:
    """A single CPU INT8 translator shared by both directions in this instance."""

    REQUIRED_FILES = (
        "model.bin", "shared_vocabulary.json", "vocab.json", "sentencepiece.bpe.model",
    )

    def __init__(self, model_dir: str | Path, num_threads: int = 3) -> None:
        if not 2 <= num_threads <= 4:
            raise ValueError("翻译 num_threads 必须在 2 到 4 之间")
        self.model_dir = Path(model_dir).resolve()
        self.num_threads = num_threads
        self._translator: Any = None
        self._tokenizer: _M2M100Tokenizer | None = None
        self._lock = threading.RLock()
        _require_files(self.model_dir, self.REQUIRED_FILES)

    def _load(self) -> None:
        if self._translator is not None:
            return
        _require_files(self.model_dir, self.REQUIRED_FILES)
        tokenizer = _M2M100Tokenizer.from_directory(self.model_dir)
        ctranslate2 = _dependency("ctranslate2")
        try:
            translator = ctranslate2.Translator(
                str(self.model_dir), device="cpu", compute_type="int8",
                inter_threads=1, intra_threads=self.num_threads,
            )
        except Exception as exc:
            raise ModelAssetsError(f"无法加载本地 CT2 INT8 翻译模型：{exc}") from exc
        self._tokenizer = tokenizer
        self._translator = translator

    def preload(self) -> None:
        with self._lock:
            self._load()

    @property
    def backend_info(self) -> dict[str, Any]:
        return {"backend": "ctranslate2", "model_kind": "m2m100", "device": "cpu",
                "num_threads": self.num_threads, "compute_type": "int8", "beam_size": 2,
                "supports_source_context": False}

    def translate(self, text: str, source_lang: str, target_lang: str, *, context: str = "") -> str:
        source_lang, target_lang = _language(source_lang), _language(target_lang)
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        if not isinstance(context, str):
            raise TypeError("context 必须是字符串")
        text = text.strip()
        if not text or (source_lang == target_lang and (target_lang != "zh" or is_chinese_output(text))):
            return text
        with self._lock:
            self._load()
            assert self._tokenizer is not None
            # This seq2seq checkpoint has no reference-only input channel.
            # Concatenating history would also translate already shown text.
            tokens = self._tokenizer.encode(text, source_lang)
            if len(tokens) > 512:
                raise ValueError("待翻译文本超过 512 token，请缩短音频片段")
            results = self._translator.translate_batch(
                [tokens], target_prefix=[[f"__{target_lang}__"]],
                beam_size=2, max_input_length=512, max_decoding_length=256,
                num_hypotheses=1,
            )
            if not results or not results[0].hypotheses:
                raise RuntimeError("本地翻译模型未返回结果")
            translation = self._tokenizer.decode(results[0].hypotheses[0])
            if not translation:
                raise RuntimeError("本地翻译模型返回了空译文")
            return ensure_chinese_output(translation, target_lang)


def create_translation_engine(model_dir: str | Path, num_threads: int = 4):
    """Legacy directories stay CPU M2M100; GGUF is explicitly selected locally."""
    model_dir = Path(model_dir).resolve()
    backend_path = model_dir / "backend.json"
    if backend_path.is_file():
        try:
            config = json.loads(backend_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ModelAssetsError("无法读取本地翻译后端配置") from exc
        if not isinstance(config, dict):
            raise ModelAssetsError("本地翻译后端配置必须是 JSON 对象")
        if config.get("backend") == "llama_cpp":
            from .gguf_engine import GgufTranslationEngine
            return GgufTranslationEngine(model_dir, num_threads=num_threads, config=config)
    return TranslationEngine(model_dir, num_threads=num_threads)
