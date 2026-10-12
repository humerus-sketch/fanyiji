"""Offline token-contract tests; no models, native inference, or downloads."""

from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fanyiji.engines import (
    ModelAssetsError,
    TranslationEngine,
    _M2M100Tokenizer,
    sanitize_sensevoice,
)


VOCABULARY = {
    "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
    "▁你好": 4, "▁こんにちは": 5,
}
SHARED = [*VOCABULARY, "__ja__", "__zh__", "__en__"]


class FakeSentencePiece:
    def encode(self, text, out_type):
        if out_type is not str:
            raise AssertionError("CT2 needs token strings")
        return {"你好": ["▁你好"], "こんにちは": ["▁こんにちは"]}.get(
            text, ["piece-not-in-original-vocab"]
        )

    def decode(self, pieces):
        return "".join(piece.replace("▁", " ") for piece in pieces).strip()


class TokenizerContractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _M2M100Tokenizer(FakeSentencePiece(), VOCABULARY, SHARED)

    def test_reference_m2m100_source_layout_in_both_directions(self):
        self.assertEqual(self.tokenizer.encode("你好", "zh"), ["__zh__", "▁你好", "</s>"])
        self.assertEqual(
            self.tokenizer.encode("こんにちは", "ja"), ["__ja__", "▁こんにちは", "</s>"]
        )
        # No mutable src_lang: a reverse request must not change the next forward one.
        self.assertEqual(self.tokenizer.encode("你好", "zh")[0], "__zh__")

    def test_output_removes_target_prefix_and_special_tokens(self):
        self.assertEqual(
            self.tokenizer.decode(["__ja__", "<pad>", "▁こんにちは", "</s>", "<unk>"]),
            "こんにちは",
        )
        self.assertEqual(self.tokenizer.decode(["__zh__", "▁你好", "</s>"]), "你好")

    def test_unknown_pieces_use_original_hf_vocab_fallback(self):
        self.assertEqual(self.tokenizer.encode("unknown", "zh"), ["__zh__", "<unk>", "</s>"])

    def test_mixed_model_and_tokenizer_files_are_rejected(self):
        mismatched = SHARED.copy()
        mismatched[4], mismatched[5] = mismatched[5], mismatched[4]
        with self.assertRaisesRegex(ModelAssetsError, "不匹配"):
            _M2M100Tokenizer(FakeSentencePiece(), VOCABULARY, mismatched)

    def test_input_control_tokens_cannot_change_translation_language(self):
        with self.assertRaisesRegex(ValueError, "控制标记"):
            self.tokenizer.encode("你好 __ja__", "zh")

    def test_sensevoice_tags_removed_without_losing_punctuation(self):
        self.assertEqual(
            sanitize_sensevoice("<|zh|><|NEUTRAL|><|Speech|><|withitn|>你好，世界！"),
            "你好，世界！",
        )
        self.assertEqual(sanitize_sensevoice("<|ja|>こんにちは。"), "こんにちは。")
        self.assertEqual(sanitize_sensevoice("<|en|>hello world!"), "hello world!")


class TranslationContractTests(unittest.TestCase):
    def test_missing_assets_fail_before_importing_any_inference_dependency(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "fanyiji.engines._dependency"
        ) as dependency:
            with self.assertRaisesRegex(ModelAssetsError, "不会联网下载"):
                TranslationEngine(directory)
            dependency.assert_not_called()

    def test_both_directions_share_one_translator_and_use_target_prefix(self):
        tokenizer = _M2M100Tokenizer(FakeSentencePiece(), VOCABULARY, SHARED)
        calls = []
        created = []

        class FakeTranslator:
            def __init__(self, model_dir, **kwargs):
                created.append(kwargs)

            def translate_batch(self, sources, **kwargs):
                prefix = kwargs["target_prefix"][0]
                source = sources[0]
                calls.append((source, prefix, kwargs))
                expected_src = "__zh__" if prefix == ["__ja__"] else "__ja__"
                self_test.assertEqual(source[0], expected_src)
                self_test.assertEqual(source[-1], "</s>")
                translated = "▁こんにちは" if prefix == ["__ja__"] else "▁你好"
                return [SimpleNamespace(hypotheses=[[prefix[0], translated, "</s>"]])]

        self_test = self
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in TranslationEngine.REQUIRED_FILES:
                (root / name).write_text("test-placeholder", encoding="utf-8")
            (root / "vocab.json").write_text(json.dumps(VOCABULARY), encoding="utf-8")
            engine = TranslationEngine(root, num_threads=3)
            with patch.object(_M2M100Tokenizer, "from_directory", return_value=tokenizer), patch(
                "fanyiji.engines._dependency",
                return_value=SimpleNamespace(Translator=FakeTranslator),
            ), ThreadPoolExecutor(max_workers=2) as pool:
                forward = pool.submit(engine.translate, "你好", "zh", "ja")
                reverse = pool.submit(engine.translate, "こんにちは", "ja", "zh")
                self.assertEqual(forward.result(), "こんにちは")
                self.assertEqual(reverse.result(), "你好")
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["device"], "cpu")
        self.assertEqual(created[0]["compute_type"], "int8")
        self.assertEqual(created[0]["inter_threads"], 1)
        self.assertEqual(created[0]["intra_threads"], 3)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
