"""Source history isolation, expiration, and sentence-hint boundaries."""
import pytest

from fanyiji.text_context import (MAX_CONTEXT_CHARACTERS, MAX_CONTEXT_TOKENS,
                                         SourceTextContext, bounded_source_context,
                                         join_continuation_text, looks_incomplete)


def test_history_never_crosses_audio_channels_or_language_directions():
    history = SourceTextContext(clock=lambda: 1.0)
    history.remember("playback", ("ja", "zh"), "ミコさんと話しました。")
    history.remember("microphone", ("zh", "ja"), "明天再联系。")
    assert history.get("playback", ("ja", "zh")) == "ミコさんと話しました。"
    assert history.get("microphone", ("zh", "ja")) == "明天再联系。"
    assert history.get("playback", ("zh", "ja")) == ""
    assert history.get("microphone", ("ja", "zh")) == ""


def test_only_two_newest_source_segments_are_retained():
    history = SourceTextContext(max_segments=2, clock=lambda: 1.0)
    for text in ("first", "second", "third"):
        history.remember("playback", ("ja", "zh"), text)
    assert history.get("playback", ("ja", "zh")) == "second\nthird"


def test_defaults_keep_three_recent_segments_with_unchanged_token_budget():
    history = SourceTextContext(clock=lambda: 1.0)
    assert history.max_segments == 3 and history.ttl_seconds == 20.0
    assert MAX_CONTEXT_CHARACTERS == 384 and MAX_CONTEXT_TOKENS == 256
    for text in ("first", "second", "third", "fourth"):
        history.remember("system", ("ja", "zh"), text)
    assert history.get("system", ("ja", "zh")) == "second\nthird\nfourth"


def test_default_expiration_is_twenty_seconds_from_capture_time():
    history = SourceTextContext(clock=lambda: 9999.0)
    history.remember("system", ("ja", "zh"), "今日も配信します。", now=10.0)
    assert history.get("system", ("ja", "zh"), now=29.99) == "今日も配信します。"
    assert history.get("system", ("ja", "zh"), now=30.0) == ""


def test_character_budget_selects_complete_newest_segments_without_sliced_older_words():
    history = SourceTextContext(clock=lambda: 1.0)
    old = "古" * 200
    recent = "実際には違います。" * 10
    latest = "明日は休みません。" * 15
    for text in (old, recent, latest):
        history.remember("system", ("ja", "zh"), text)
    expected = recent + "\n" + latest
    assert len(old + "\n" + expected) > MAX_CONTEXT_CHARACTERS
    assert history.get("system", ("ja", "zh")) == expected


def test_token_budget_drops_whole_older_segment_and_keeps_latest_negation():
    history = SourceTextContext(clock=lambda: 1.0, token_counter=len)
    old = "古" * 160
    latest = "配信をやめるわけではありません。" * 6
    for text in (old, latest):
        history.remember("system", ("ja", "zh"), text)
    assert len(old + "\n" + latest) < MAX_CONTEXT_CHARACTERS
    assert len(old + "\n" + latest) > MAX_CONTEXT_TOKENS
    assert history.get("system", ("ja", "zh")) == latest


def test_oversized_latest_segment_is_only_case_using_a_partial_suffix():
    history = SourceTextContext(clock=lambda: 1.0)
    history.remember("system", ("ja", "zh"), "古い話題。")
    latest = "新" * 500 + "終わりません。"
    history.remember("system", ("ja", "zh"), latest)
    assert history.get("system", ("ja", "zh")) == latest[-MAX_CONTEXT_CHARACTERS:]


def test_oversized_latest_token_budget_does_not_fill_with_older_segment():
    history = SourceTextContext(clock=lambda: 1.0, token_counter=lambda text: len(text) * 2)
    history.remember("system", ("ja", "zh"), "古い話題。")
    latest = "新" * 300
    history.remember("system", ("ja", "zh"), latest)
    result = history.get("system", ("ja", "zh"))
    assert result and len(result) * 2 <= MAX_CONTEXT_TOKENS
    assert set(result) == {"新"}


@pytest.mark.parametrize("counter", [lambda text: True, lambda text: -1, lambda text: None])
def test_unusable_token_count_falls_back_to_complete_character_bounded_segments(counter):
    history = SourceTextContext(clock=lambda: 1.0, token_counter=counter)
    old, latest = "古" * 300, "新" * 100
    history.remember("system", ("ja", "zh"), old)
    history.remember("system", ("ja", "zh"), latest)
    assert history.get("system", ("ja", "zh")) == latest


def test_failing_tokenizer_keeps_character_budget_and_complete_segments():
    def unavailable(text):
        raise RuntimeError("tokenizer unavailable")

    history = SourceTextContext(clock=lambda: 1.0, token_counter=unavailable)
    history.remember("system", ("ja", "zh"), "古" * 300)
    history.remember("system", ("ja", "zh"), "最新の話題です。")
    assert history.get("system", ("ja", "zh")) == "古" * 300 + "\n最新の話題です。"


@pytest.mark.parametrize("text,budget,expected", [
    ("古い話題がまだ続いています。\n新しい話題です。", 12, "新しい話題です。"),
    ("古い話題。\n新しい話題です。", 8, "新しい話題です。"),
    ("一行だけの長い前文です。", 5, "前文です。"),
])
def test_character_trimming_prefers_complete_newer_line(text, budget, expected):
    assert bounded_source_context(text, max_characters=budget) == expected


def test_token_shrinking_tries_whole_latest_line_before_slicing_that_line():
    latest = "配信をやめるわけではありません。"
    text = "昔。\n" + latest
    assert bounded_source_context(text, token_counter=len, max_tokens=len(latest)) == latest


def test_many_line_token_shrinking_remains_bounded_and_keeps_complete_lines():
    calls = []

    def counter(text):
        calls.append(text)
        return len(text) * 3

    result = bounded_source_context("\n".join("reference" for _ in range(200)), token_counter=counter)
    assert len(result) * 3 <= MAX_CONTEXT_TOKENS
    assert all(line == "reference" for line in result.splitlines())
    assert len(calls) <= 8


def test_history_uses_capture_time_ttl_and_rejects_clock_reversal():
    history = SourceTextContext()
    history.remember("playback", ("ja", "zh"), "first", now=10)
    history.remember("playback", ("ja", "zh"), "second", now=20)
    assert history.get("playback", ("ja", "zh"), now=30) == "second"
    assert history.get("playback", ("ja", "zh"), now=40) == ""
    history.remember("playback", ("ja", "zh"), "future", now=40)
    assert history.get("playback", ("ja", "zh"), now=39) == ""


def test_valid_older_reference_cannot_hide_a_future_same_channel_entry():
    history = SourceTextContext()
    history.remember("system", ("ja", "zh"), "过去", now=30)
    history.remember("system", ("ja", "zh"), "未来", now=40)
    assert history.get("system", ("ja", "zh"), now=35) == "过去"


def test_older_other_channel_capture_time_does_not_clear_newer_source_history():
    history = SourceTextContext()
    history.remember("system", ("ja", "zh"), "newer system audio", now=40)
    history.remember("mic", ("zh", "ja"), "older mic audio", now=20)
    assert history.get("mic", ("zh", "ja"), now=21) == "older mic audio"
    assert history.get("system", ("ja", "zh"), now=41) == "newer system audio"
    # Expiring one source with a later capture time also leaves the other
    # source available for its own in-order timeline.
    assert history.get("mic", ("zh", "ja"), now=100) == ""
    assert history.get("system", ("ja", "zh"), now=42) == "newer system audio"


def test_older_different_direction_time_does_not_clear_newer_direction_history():
    history = SourceTextContext()
    history.remember("system", ("ja", "zh"), "Japanese audio", now=40)
    history.remember("system", ("zh", "ja"), "Chinese audio", now=20)
    assert history.get("system", ("zh", "ja"), now=21) == "Chinese audio"
    assert history.get("system", ("ja", "zh"), now=41) == "Japanese audio"


def test_clear_one_channel_or_all_history():
    history = SourceTextContext(clock=lambda: 1.0)
    history.remember("playback", ("ja", "zh"), "日本語")
    history.remember("microphone", ("zh", "ja"), "中文")
    history.clear("playback")
    assert history.get("playback", ("ja", "zh")) == ""
    assert history.get("microphone", ("zh", "ja")) == "中文"
    history.clear()
    assert history.get("microphone", ("zh", "ja")) == ""


def test_reference_is_character_bounded_without_tokenizer_and_preserves_newest_text():
    history = SourceTextContext(clock=lambda: 1.0)
    history.remember("playback", ("ja", "zh"), "古" * 400)
    history.remember("playback", ("ja", "zh"), "新" * 100)
    result = history.get("playback", ("ja", "zh"))
    assert result == "新" * 100
    assert bounded_source_context("古" * 400 + "new") == "古" * 381 + "new"


def test_real_token_counter_bounds_reference_and_failing_counter_uses_characters():
    result = bounded_source_context("語" * 256, token_counter=lambda text: len(text) * 3)
    assert len(result) * 3 <= 256

    def unavailable(text):
        raise RuntimeError("no tokenizer")

    assert len(bounded_source_context("語" * 400, token_counter=unavailable)) == 384
    assert len(bounded_source_context("語" * 400, token_counter=lambda text: True)) == 384


@pytest.mark.parametrize("text,language", [
    ("森の。", "ja"), ("雨が降っているので。", "ja"), ("明日は行くけど。", "ja"),
    ("田中さんが。", "ja"), ("因为。", "zh"), ("明天如果。", "zh"), ("这是他的。", "zh"),
    ("彼女は今日は。", "ja"), ("雨が止まなければ。", "ja"), ("雨が降れば。", "ja"),
    ("雨が止んだら。", "ja"), ("如果不着急，", "zh"), ("假如明天下雨，", "zh"),
    ("要是不方便。", "zh"), ("除非雨停了。", "zh"),
])
def test_asr_period_does_not_hide_continuation_candidate(text, language):
    assert looks_incomplete(text, language)


@pytest.mark.parametrize("text,language", [
    ("はい。", "ja"), ("いいえ！", "ja"), ("そうですね。", "ja"), ("ありがとうございます。", "ja"),
    ("是。", "zh"), ("不是。", "zh"), ("好的！", "zh"), ("不用。", "zh"),
    ("雨が降っています。", "ja"), ("明天开会。", "zh"), ("", "ja"),
    ("こんにちは。", "ja"), ("そうなの。", "ja"), ("何の？", "ja"), ("谁的？", "zh"),
    ("こんばんは。", "ja"), ("如果下雨，我们就取消活动。", "zh"),
    ("如果下雨那么活动取消。", "zh"), ("如果下雨则取消活动。", "zh"),
    ("除非雨停了，否则我不会去。", "zh"), ("除非雨停了我才会去。", "zh"),
    ("如果下雨，那么这个安排就是他的。", "zh"),
    ("如果不着急请明天再发。", "zh"), ("因为我很忙。", "zh"), ("虽然今天很忙。", "zh"),
])
def test_short_replies_and_ordinary_complete_statements_are_not_forced_to_merge(text, language):
    assert not looks_incomplete(text, language)


@pytest.mark.parametrize("kwargs", [{"max_segments": 4}, {"max_segments": True},
                                    {"ttl_seconds": 0}, {"ttl_seconds": float("nan")}])
def test_invalid_history_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SourceTextContext(**kwargs)


@pytest.mark.parametrize("previous,current,language,expected", [
    ("彼女は今日は。", "歌わないと言いました。", "ja", "彼女は今日は歌わないと言いました。"),
    ("森の。", "妖精みたいな名前だと思います。", "ja", "森の妖精みたいな名前だと思います。"),
    ("雨が止まなければ。", "明日のイベントは中止です。", "ja", "雨が止まなければ明日のイベントは中止です。"),
    ("雨が降れば．", "中止です。", "ja", "雨が降れば中止です。"),
    ("雨が止んだら.", "出かけます。", "ja", "雨が止んだら出かけます。"),
    ("如果不着急，", "请明天再发。", "zh", "如果不着急，请明天再发。"),
    ("假如明天下雨。", "我们就取消活动。", "zh", "假如明天下雨我们就取消活动。"),
    ("除非雨停了。", "否则我不会去。", "zh", "除非雨停了否则我不会去。"),
    ("这是他的。", "300元，不是200元。", "zh", "这是他的300元，不是200元。"),
    ("番号3.14の。", "商品ではありません。", "ja", "番号3.14の商品ではありません。"),
    ("如果价格是3.14元，", "请留下2.07元。", "zh", "如果价格是3.14元，请留下2.07元。"),
    ("  森の。  ", "  妖精です。 ", "ja", "  森の妖精です。 "),
])
def test_safe_continuation_changes_only_boundary_punctuation_and_whitespace(previous, current, language, expected):
    assert join_continuation_text(previous, current, language) == expected


@pytest.mark.parametrize("previous,current,language", [
    ("森の？", "妖精です。", "ja"), ("森の！", "妖精です。", "ja"),
    ("森の...", "妖精です。", "ja"), ("森の…", "妖精です。", "ja"),
    ("森の。。", "妖精です。", "ja"), ("森の．．．", "妖精です。", "ja"),
    ("「森の。」", "妖精です。", "ja"), ("「森の」。", "妖精です。", "ja"),
    ("（森の）。", "妖精です。", "ja"), ("\"森の\".", "妖精です。", "ja"),
    ("こんにちは。", "明日は歌いません。", "ja"), ("こんばんは。", "明日は歌いません。", "ja"),
    ("はい。", "歌いません。", "ja"), ("いいえ。", "歌います。", "ja"),
    ("何の？", "妖精です。", "ja"), ("谁的？", "是他的。", "zh"),
    ("如果不着急，请明天再发。", "金额是300元。", "zh"),
    ("除非雨停了，否则我不会去。", "谢谢。", "zh"),
    ("因为我很忙。", "请明天再发。", "zh"),
    ("300元を返してください。", "200元は残してください。", "ja"),
    ("价格是3.14。", "不是2.07。", "zh"),
    ("森の", "妖精です。", "ja"), ("", "妖精です。", "ja"), ("森の。", "", "ja"),
])
def test_unsafe_or_complete_boundary_is_preserved_verbatim(previous, current, language):
    assert join_continuation_text(previous, current, language) == previous + "\n" + current


def test_join_never_edits_negation_currency_numbers_or_inner_punctuation():
    previous = "金額2077.14元の。"
    current = "取引ではなく、300元を返して200元を残す話です。"
    joined = join_continuation_text(previous, current, "ja")
    assert joined == previous[:-1] + current
    for exact in ("2077.14元", "ではなく", "300元", "200元"):
        assert exact in joined


@pytest.mark.parametrize("text", [
    "別にそう。", "別にそういう。", "そういう意味で。",
    "その煽り文句だって気がする。別にそう。",
    "そういう意味で言ったわけでは。",
])
def test_explicit_japanese_unfinished_negation_prefaces_are_continuation_candidates(text):
    assert looks_incomplete(text, "ja")


@pytest.mark.parametrize("text", [
    "そう。", "そうです。", "そうですね。", "別にそうでもない。",
    "別にそういう意味ではない。", "そういう意味です。", "そういう意味。",
    "私は別にそう思わない。", "田中さんなら大丈夫です。",
])
def test_complete_japanese_responses_are_not_swallowed_by_negation_preface_hint(text):
    assert not looks_incomplete(text, "ja")
    assert join_continuation_text(text, "金額は300元です。", "ja") == text + "\n金額は300元です。"


@pytest.mark.parametrize("previous,current,expected", [
    ("その煽り文句だって気がする。別にそう。", "意味いったわけじゃないと思うけど。",
     "その煽り文句だって気がする。別にそう意味いったわけじゃないと思うけど。"),
    ("別にそういう。", "意味で言ったわけじゃない。", "別にそういう意味で言ったわけじゃない。"),
    ("そういう意味で。", "田中さんが300元ではなく2077.14元と言ったわけではない。",
     "そういう意味で田中さんが300元ではなく2077.14元と言ったわけではない。"),
])
def test_negation_continuation_keeps_all_actual_asr_words_without_grammar_repair(previous, current, expected):
    joined = join_continuation_text(previous, current, "ja")
    assert joined == expected
    assert joined == previous[:-1] + current


@pytest.mark.parametrize("previous", [
    "別にそう？", "別にそう！", "別にそう…", "別にそう。。",
    "「別にそう。」", "『別にそう』。", "（別にそう）。",
])
def test_explicit_negation_preface_never_removes_question_or_quote_boundary(previous):
    current = "言ったわけじゃない。"
    assert join_continuation_text(previous, current, "ja") == previous + "\n" + current
