"""Chinese output contract at inference, presentation and failure persistence."""
from unittest.mock import Mock, patch

import pytest

from fanyiji.codex_engine import CodexTranslationEngine
from fanyiji.translation_output import ChineseOutputError, ensure_chinese_output, is_chinese_output
from fanyiji.danmaku_store import DanmakuStore
from fanyiji.gguf_engine import GgufTranslationEngine


@pytest.mark.parametrize("text", ["蕾莲 谢谢你的留言 小库也谢谢", "今天还有3秒，不能开门。", "１２点开会", "【现在请大家冷静一点】", "温度25℃ 价格￥50", "中文𠮷姓", "123"])
def test_chinese_and_language_neutral_characters_are_allowed(text):
    assert is_chinese_output(text)


@pytest.mark.parametrize("text", ["レレン谢谢你的留言", "クちゃん也谢谢", "Mio小姐", "ＳＣ留言", "ｸﾁｬﾝ", "大家thank you", "中文한글", "中文русский", "中文é", "中文ー", "中文🙂", "中\u200b文", "", "   ", None])
def test_foreign_letters_and_hidden_symbols_are_rejected(text):
    assert not is_chinese_output(text)


def test_reverse_translation_is_not_subject_to_chinese_output_policy():
    assert ensure_chinese_output("ありがとうございます", "ja") == "ありがとうございます"
    with pytest.raises(ChineseOutputError):
        ensure_chinese_output("レレン，谢谢", "zh")


def codex_stub(*outputs):
    engine = CodexTranslationEngine()
    engine._translate_payload = Mock(side_effect=outputs)
    engine.reset_conversation = Mock()
    return engine


def test_codex_foreign_name_is_retranslated_once_without_deleting_it():
    engine = codex_stub("レレン，谢谢留言。", "蕾莲，谢谢留言。")
    assert engine.translate("レレン コメントありがとう", "ja", "zh") == "蕾莲，谢谢留言。"
    assert engine._translate_payload.call_count == 2
    second = engine._translate_payload.call_args
    assert "不删除内容" in second.args[2]
    assert "output_requirement" not in second.args[0]
    assert second.kwargs["reuse"] is False
    engine.reset_conversation.assert_called_once_with("default")


def test_codex_still_foreign_after_one_correction_is_not_returned():
    engine = codex_stub("レレン，谢谢。", "レレン，留言谢谢。")
    with pytest.raises(ChineseOutputError):
        engine.translate("レレン コメントありがとう", "ja", "zh")
    assert engine._translate_payload.call_count == 2


def test_chinese_source_with_foreign_fragments_does_not_use_passthrough():
    engine = codex_stub("米奥 谢谢留言")
    assert engine.translate("Mio 谢谢留言", "zh", "zh") == "米奥 谢谢留言"
    assert engine._translate_payload.call_count == 1


def test_codex_batch_ids_remain_unchanged_while_only_values_must_be_chinese():
    engine = codex_stub({"en-id": "Mio谢谢", "ja-id": "蕾莲谢谢"},
                        {"en-id": "米奥谢谢", "ja-id": "蕾莲谢谢"})
    items = [{"id": "en-id", "text": "Mio ありがとう"}, {"id": "ja-id", "text": "レレンありがとう"}]
    assert engine.translate_batch(items, "mixed", "zh") == {"en-id": "米奥谢谢", "ja-id": "蕾莲谢谢"}
    assert engine._translate_payload.call_count == 2


@pytest.mark.parametrize("outputs,expected", [
    (["レレン 谢谢留言", "蕾莲 谢谢留言"], "蕾莲 谢谢留言"),
    (["レレン 谢谢留言", "レレン 再次谢谢"], None),
])
def test_local_correction_is_bounded_and_keeps_healthy_runtime(tmp_path, outputs, expected):
    (tmp_path / "test.gguf").write_bytes(b"test model")
    engine = GgufTranslationEngine(tmp_path, config={"model_file": "test.gguf", "model_kind": "hy"})
    requests = []

    def request(method, path, body):
        if path == "/apply-template":
            return {"prompt": body["messages"][0]["content"]}
        if path == "/tokenize":
            return {"tokens": [1] * 100}
        assert path == "/v1/chat/completions"
        requests.append(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": outputs[len(requests) - 1]}}]}

    with patch.object(engine, "_ensure_loaded"), patch.object(engine, "_request", side_effect=request), \
            patch.object(engine, "_stop_runtime") as stop:
        if expected is None:
            with pytest.raises(ChineseOutputError):
                engine.translate("レレン コメントありがとう", "ja", "zh")
        else:
            assert engine.translate("レレン コメントありがとう", "ja", "zh") == expected
        assert len(requests) == 2
        assert "不删除内容" in requests[1]["messages"][0]["content"]
        stop.assert_not_called()


def test_unqueued_language_failure_survives_restart_and_reserves_unique_id(tmp_path):
    path = tmp_path / "queue.sqlite3"
    store = DanmakuStore(path)
    identifier = store.stage_failed_handoff("1698225", "【レレン 谢谢你的留言】")
    assert store.head() is None
    assert len(store.failed_handoffs()) == 1
    store.reset_for_restart()
    store.connection.close()
    restored = DanmakuStore(path)
    assert restored.failed_handoffs()[0]["queue_id"] == identifier
    assert restored.enqueue("1698225", "system", "new", 1, "现在是正常的中文译文", ["【现在是正常的中文译文】"])
    assert restored.head()["id"] > identifier
    restored.connection.close()
