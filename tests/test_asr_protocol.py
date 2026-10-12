import pytest

from fanyiji.asr_protocol import AsrProtocolError, contains_asr_protocol, require_subtitle_text


@pytest.mark.parametrize("text", [
    "language None<asr_text>", "语言 None<asr_text>", "language None", "【语言：None】",
    "<asr_text>原文", "hello <asr_text", "<ASR_TEXT>hello", "&lt;asr_text&gt;",
    "<|im_start|>assistant", "text <|im_end|>",
])
def test_control_tokens_fail_closed(text):
    assert contains_asr_protocol(text)
    with pytest.raises(AsrProtocolError):
        require_subtitle_text(text)


@pytest.mark.parametrize("text", [
    "", "はい。", "打开背包。", "We need more ammo.", "language Japanese is supported",
    "他说 None 是空值。", "血量 < 20", "这个词是 language None 的报错", "language NoneType is a name",
])
def test_ordinary_words_and_punctuation_are_not_rewritten(text):
    assert not contains_asr_protocol(text)
    assert require_subtitle_text(text) == text
