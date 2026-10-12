"""Small UI-only catalog. Audio languages and subtitle content are independent."""

from __future__ import annotations

from functools import lru_cache
import re
from string import Formatter

SUPPORTED_LANGUAGES = ("zh-CN", "ja", "en")
LANGUAGE_NAMES = {"zh-CN": "简体中文", "ja": "日本語", "en": "English"}
_language = "zh-CN"

_RECOVERY_ACTION_LABELS = {
    "analyze_with_codex": "正在请求 Codex 分析失败正文",
    "apply_local_edits": "正在应用已检查的局部修改",
    "retry_current": "按发送间隔自动重试当前条",
    "verify_only": "自动只读核验发送结果",
    "recheck_login": "自动复检登录和页面状态",
    "repacketize": "按账号允许长度重新拆分",
    "check_environment": "自动检查浏览器和页面状态",
    "wait_for_login": "保留消息，等待登录或验证",
    "repair_local_file": "保留消息，请修复本地名单文件",
    "review_on_restart": "下次启动自动重试分析",
    "transfer_failure": "失败消息交由独立处理，继续后续队列",
    "retry_failure_storage": "正在重试保存失败记录",
}

_RECOVERY_CATEGORY_LABELS = {
    "ai_review": "AI分析（原因尚未证实）",
    "platform_response": "平台回执（原因未确定）",
    "uncertain_delivery": "投递结果未确认",
    "length": "消息长度限制",
    "platform_rejection": "平台明确拒绝（原因未确定）",
    "authentication": "登录或凭证状态",
    "local_feature": "本地功能或配置",
    "connection": "浏览器或页面连接",
}

# Values are Japanese and English respectively; Chinese source keys are the fallback.
_MESSAGES: dict[str, tuple[str, str]] = {
    "等待自动分析失败正文": ("失敗本文の自動分析待ち", "Waiting for automatic analysis of failed text"),
    "正在独立分析和验证失败正文": ("失敗本文を個別に分析・検証中", "Independently analyzing and validating failed text"),
    "上一测试正文：{0}": ("前回のテスト本文：{0}", "Previous test text: {0}"),
    "上一测试直播间：{0}": ("前回のテストルーム：{0}", "Previous test room: {0}"),
    "上一测试未通过，记录保留；当前正文须重新独立验证。": (
        "前回のテストは失敗しました。記録を保持し、現在の本文は別途再検証が必要です。", "The previous test did not pass and its record is retained; the current text requires a new independent test."),
    " · 累计移入独立处理 {0} 段": (" · 個別処理へ移動済み合計 {0} 件", " · Total moved to independent processing {0}"),
    "AI未提出可验证的局部修正 下次启动重新分析": (
        "AI は検証可能な部分修正を提示しませんでした。次回起動時に再分析します", "AI provided no local correction that can be tested; analyze again on the next launch"),
    "失败消息移入独立处理记录，后续队列继续发送；记录保留并自动更新处理状态。返回 f 不代表句中存在禁词，词库候选不会自动生效；AI经验规则也不代表已确认禁词。": (
        "失敗したメッセージは個別処理記録に移動し、後続のキュー送信を継続します。記録を保持し、処理状態を自動更新します。f 応答は文中に禁止語がある証明ではなく、辞書候補は自動適用されません。AI の経験ルールも禁止語の確認を意味しません。",
        "Failed messages move to independent processing records while later queued messages continue. Records are retained and update automatically. An f response does not prove a banned term. Dictionary candidates are not applied automatically, and AI experience rules do not confirm banned words."),
    "失败已移入独立处理，后续队列继续发送": (
        "失敗したメッセージを個別処理へ移動し、後続のキュー送信を継続", "Failed message moved to independent processing; continuing later queued messages"),
    "失败记录保存异常，正在重试写入": (
        "失敗記録を保存できません。書き込みを再試行しています", "Failure record could not be saved; retrying storage"),
    "正在重试保存失败记录": (
        "失敗記録の保存を再試行中", "Retrying failure-record storage"),
    "失败消息交由独立处理，继续后续队列": (
        "失敗したメッセージは個別処理に渡し、後続のキューを継続", "Process the failed message separately and continue later queued messages"),
    " · 独立处理 {0} 段": (" · 個別処理 {0} 件", " · Independent processing {0}"),
    "待核验请求属于原测试配置，请恢复对应房间设置": (
        "確認待ちの要求は元のテスト設定に属します。対応するルーム設定に戻してください", "The pending check belongs to the previous test configuration; restore its room settings"),
    "保存后生效。启动后请手动打开正式直播间，不自动启动浏览器；只发送终稿，每段【译文】连同括号须为 11 到 40 字符。正文去标点，断句用空格，保留外层【】。10 秒内重复内容跳过，发送间隔至少 3.5 秒。本轮队列不过期；失败消息保存到独立处理记录，交给 Codex 分析，后续队列继续发送；修改后正文须在独立测试房间验证通过才生成规则，结果未知的测试只读核验，不重复提交；验证码、登录和禁言仍须在浏览器处理，失败内容保留。": (
        "保存後に反映します。起動後は本番ルームを手動で開いてください。ブラウザーは自動起動しません。確定訳のみ送信し、【】を含めて 11～40 文字にします。本文の句読点は空白に置き換え、外側の【】は保持します。10 秒以内の重複を省略し、送信間隔は最低 3.5 秒です。この実行中のキューは期限切れになりません。失敗したメッセージは個別処理記録に保存して Codex に分析を依頼し、後続の送信を継続します。修正本文は独立したテストが成功してからルールを生成します。結果不明のテストは再送せず、読み取り専用で確認します。認証コード、ログイン、コメント禁止はブラウザーで対応してください。失敗本文は保持します。",
        "Saved settings take effect immediately. Open the production room manually after launch; the browser does not start automatically. Send final translations only, with 11–40 characters including 【】. Replace punctuation with spaces and retain the outer brackets. Skip duplicates within 10 seconds and allow at least 3.5 seconds between posts. The current-run queue does not expire. Failed messages are retained in independent processing records for Codex analysis while later messages continue. Edited text must pass a separate test before rules are generated. Unknown test results are checked without resubmitting. Handle verification, sign-in, and chat bans in the browser; failed text is retained."),
    "验证记录暂存失败 自动重试保存": (
        "検証記録の保存に失敗しました。保存を自動再試行します", "Validation record could not be saved; retrying storage automatically"),
    "等待手动打开已登录的验证直播间": (
        "ログイン済みの検証ルームを手動で開くまで待機", "Waiting for you to open the signed-in validation live room"),
    "等待需要验证的修改后正文": (
        "検証が必要な修正本文を待機中", "Waiting for edited text requiring validation"),
    "仅修改发送弹幕的文字，不修改字幕。手工规则由你确认；自动恢复使用 Codex 生成修改后正文，独立测试成功后才比对原句保存局部规则。": (
        "送信するコメントの文字のみを変更し、字幕は変更しません。手動ルールは利用者が確認します。自動復旧は Codex で修正本文を生成し、独立したテストが成功した後に元の文と比較して部分的なルールを保存します。",
        "Change outgoing chat text only; subtitles stay unchanged. You confirm manual rules. Automatic recovery uses Codex to generate edited text; local rules are saved by comparison with the original only after a separate test succeeds."),
    "未验证的测试正文只保存在失败记录中，不创建过滤规则，也不归档；测试成功后生成的规则可编辑或停用。": (
        "未検証のテキストは失敗記録にのみ保存し、置換ルールの作成やアーカイブは行いません。テスト成功後に生成されたルールは編集または無効化できます。",
        "Unverified test text is stored only in failure records, without creating filter rules or archiving the record. Rules generated after a successful test can be edited or disabled."),
    "正在验证修改后正文": ("修正本文を検証中", "Verifying the edited text"),
    "测试正文等待验证，成功后才比对原句生成局部规则": (
        "テスト本文は検証待ちです。成功後にのみ元の文と比較して部分的なルールを生成します", "Test text awaits verification; local rules are generated by comparison with the original only after success"),
    "修改后正文须在独立测试房间验证通过，才比对原句生成局部规则并归档。测试通过只说明该测试房间可发，不代表正式房间禁词原因已确认。": (
        "修正本文は独立した検証ルームで成功した後にのみ、元の文と比較して部分的なルールを生成し、記録をアーカイブします。成功はその検証ルームで送信できることだけを示し、本番ルームの禁止語の原因を証明しません。",
        "Edited text must pass a separate test room before it is compared with the original to generate local rules and archive the record. Passing only shows that the test room accepts the text; it does not confirm banned-word causes in the production room."),
    "修改后正文已通过本地检查，等待独立测试验证": (
        "修正本文はローカル検査済み、独立したテストの確認待ち", "Edited text passed local checks; awaiting separate test-room verification"),
    "正在按发送间隔重试测试正文": (
        "送信間隔に従ってテスト本文を再送中", "Retrying the test text at the send interval"),
    "修改后正文等待独立测试房间验证，尚未生成规则": (
        "修正本文は独立した検証ルームでの確認待ちで、ルールは未生成です", "Edited text awaits separate test-room verification; no rules have been generated"),
    "正在独立测试房间验证修改后正文": (
        "独立した検証ルームで修正本文を確認中", "Verifying the edited text in a separate test room"),
    "测试正文未通过，保留记录且不生成规则": (
        "テスト本文は検証に失敗しました。記録を保持し、ルールは生成しません", "Test text failed verification; the record is retained and no rules are generated"),
    "专用测试房间验证通过（正式房间原因未证实）": (
        "専用検証ルームで確認済み（本番ルームの原因未証明）", "Passed dedicated test-room validation (production-room cause unproven)"),
    "等待配置独立测试房间": ("独立した検証ルームの設定待ち", "Waiting for separate test-room configuration"),
    "等待打开验证直播间": ("検証ルームを開くまで待機", "Waiting for the validation live room to open"),
    "正在验证AI建议": ("AI 提案を検証中", "Verifying AI suggestions"),
    "验证未通过，等待继续测试": ("検証未成功、次のテスト待ち", "Validation did not pass; waiting to continue testing"),
    "等待独立测试房间验证": ("独立した検証ルームでの確認待ち", "Waiting for separate test-room validation"),
    "建议规则保持停用，验证通过后才能启用或归档": (
        "提案ルールは無効のままで、検証成功後にのみ有効化またはアーカイブできます", "Suggestion rules remain disabled; they can be enabled or archived only after passing validation"),
    "独立测试直播间：{0}": ("独立検証ルーム：{0}", "Separate test room: {0}"),
    "正式直播间：{0}": ("本番ルーム：{0}", "Production room: {0}"),
    "测试尝试次数：{0}": ("検証の試行回数：{0}", "Validation attempts: {0}"),
    "原始正文：{0}": ("元の本文：{0}", "Original text: {0}"),
    "待验证正文：{0}": ("検証対象の本文：{0}", "Text to validate: {0}"),
    "测试处理说明：{0}": ("検証の対応内容：{0}", "Validation details: {0}"),
    "测试通过只说明测试房间可发，正式房间效果和禁词原因仍未确认。": (
        "検証成功は検証ルームで送信できることだけを示します。本番ルームでの効果と禁止語の原因は未確認です。", "Passing only shows that the test room accepts the text. Production-room results and banned-word causes remain unconfirmed."),
    "AI建议已通过测试房间验证（正式房间原因未证实）": (
        "AI 提案は検証ルームで確認済み（本番ルームの原因未証明）", "AI suggestion passed test-room validation (production-room cause unproven)"),
    "独立测试房间验证通过，正式房间效果未确认": (
        "独立した検証ルームで成功、本番ルームの効果は未確認", "Separate test-room validation passed; production-room results remain unconfirmed"),
    "AI验证直播间": ("AI 検証用配信ルーム", "AI validation live room"),
    "填写独立测试房间号，不能与正式直播间相同": (
        "本番の配信ルームと異なる検証用ルーム番号を入力", "Enter a separate test room ID, different from the production room"),
    "AI建议须在独立测试房间完成发送验证后才能启用或归档。测试通过只说明该测试房间可发，不代表正式房间禁词原因已确认。": (
        "AI 提案は独立した検証用ルームで送信確認後にのみ有効化またはアーカイブできます。成功はその検証ルームで送信できることだけを示し、本番ルームの禁止語の原因を証明しません。",
        "AI suggestions can only be enabled or archived after send verification in a separate test room. Passing only shows that the test room accepts the text; it does not confirm banned-word causes in the production room."),
    "AI验证直播间不能与正式直播间相同": (
        "AI 検証用ルームは本番ルームと別にしてください", "The AI validation room must differ from the production room"),
    "AI验证直播间格式无效": ("AI 検証用ルームの形式が無効です", "Invalid AI validation room"),
    "打开验证直播间": ("検証用ルームを開く", "Open validation live room"),
    "AI建议未验证时保持停用，不归档；独立测试房间验证通过后才可启用，可编辑或停用。": (
        "未検証の AI 提案は無効のままでアーカイブしません。独立した検証ルームで成功してから有効化でき、編集または無効化できます。",
        "Untested AI suggestions remain disabled and unarchived. They can be enabled only after passing a separate test room, and remain editable or disableable."),
    "AI建议（尚未验证，已停用，原因未证实）": (
        "AI 提案（未検証、無効、原因未証明）", "AI suggestion (unverified, disabled; cause unproven)"),
    "编辑AI建议规则：未验证时保持停用，可修改内容，验证通过后才可启用。": (
        "AI 提案ルールを編集：未検証の場合は無効のままです。内容を変更でき、検証成功後にのみ有効化できます。", "Edit an AI suggestion rule: it remains disabled until verified. You can edit its content; it can be enabled only after passing validation."),
    "下次启动自动重试分析": ("次回起動時に分析を自動再試行", "Retrying analysis automatically on next launch"),
    "AI分析暂不可用 下次启动自动重试": (
        "AI 分析を現在利用できません。次回起動時に自動再試行します", "AI analysis is temporarily unavailable; retrying automatically on next launch"),
    "AI建议等待独立测试房间验证，尚未启用": (
        "AI 提案は独立した検証ルームでの確認待ちで、まだ無効です", "AI suggestions await separate test-room verification and remain disabled"),
    "正在独立测试房间验证AI建议": (
        "独立した検証ルームで AI 提案を確認中", "Verifying AI suggestions in a separate test room"),
    "AI建议测试未通过，保持停用并保留记录": (
        "AI 提案は検証に失敗しました。無効のままで記録を保持します", "AI suggestions failed validation; remaining disabled with records retained"),
    "每次启动清空上一轮待发队列与运行统计，失败处理记录和过滤规则保留并自动分析。": (
        "起動ごとに前回の未送信キューと動作統計を消去します。失敗対応記録と置換ルールは保持し、自動分析します。", "Each launch clears the previous unsent queue and runtime statistics. Failure recovery records and filter rules are retained and analyzed automatically."),
    "平台回执（原因未确定）": ("プラットフォーム応答（原因不明）", "Platform response (cause undetermined)"),
    "投递结果未确认": ("配信結果未確認", "Delivery result unconfirmed"),
    "消息长度限制": ("メッセージ長の制限", "Message length limit"),
    "平台明确拒绝（原因未确定）": ("プラットフォームが明確に拒否（原因不明）", "Explicit platform rejection (cause undetermined)"),
    "登录或凭证状态": ("ログインまたは認証情報の状態", "Sign-in or credential state"),
    "本地功能或配置": ("ローカル機能または設定", "Local feature or configuration"),
    "浏览器或页面连接": ("ブラウザーまたはページ接続", "Browser or page connection"),
    "平台返回 f 原因未确定 正在自动分析并保留顺序重试": (
        "f 応答の原因は不明です。自動分析し、順序を保持して再送中です", "Platform returned f; cause unknown. Analyzing automatically and retrying in queue order"),
    "发送结果未知 正在持续只读核验 不重复提交": (
        "送信結果不明です。再送せずに読み取り専用の確認を継続中です", "Send result unknown; continuously checking without resubmitting"),
    "正在按账号长度上限重新拆分": (
        "アカウントのメッセージ長の上限に合わせて再分割中", "Splitting to the account's message length limit"),
    "平台明确拒绝 原因未确定 正在自动检查登录和页面状态": (
        "プラットフォームが明確に拒否しました。原因は不明で、ログインとページ状態を自動確認中です", "Explicit platform rejection; cause unknown. Automatically checking sign-in and page state"),
    "账号未登录 完成登录后自动恢复当前消息": (
        "アカウントは未ログインです。ログイン後に現在のメッセージを自動復旧します", "Account is signed out; automatically recovering the current message after sign-in"),
    "登录凭证检查失败 正在自动复检": (
        "認証情報の確認が失敗したため、自動再確認中", "Sign-in credential check failed; rechecking automatically"),
    "本机弹幕功能不可用 消息保留": (
        "この PC のコメント機能を利用できません。メッセージは保持されます", "Local chat sending is unavailable; messages are retained"),
    "请求尚未提交 正在自动检查连接和页面": (
        "要求は未送信です。接続とページを自動確認中", "Request has not been submitted; checking the connection and page automatically"),
    "仅修改发送弹幕的文字，不修改字幕。手工规则由你确认；自动恢复可使用 Codex 分析，通过本地检查的建议可保存为 AI建议规则，页面观察后的经验规则另行标记。": (
        "送信するコメントの文字のみを変更し、字幕は変更しません。手動ルールは利用者が確認します。自動復旧は Codex の分析を使用でき、ローカル検査済みの提案は AI 提案ルールとして保存され、ページでの表示に基づく経験ルールは別に表示されます。",
        "Change outgoing chat text only; subtitles stay unchanged. You confirm manual rules. Automatic recovery can use Codex analysis. Locally checked suggestions can be saved as AI suggestion rules, with rules based on page observations labeled separately."),
    "AI建议规则尚未实测，保存后默认启用，可编辑或停用；不代表平台确认禁词。": (
        "AI 提案ルールは未実測です。保存後は既定で有効になり、編集または無効化できます。プラットフォームの禁止語が確認されたことを示しません。",
        "AI suggestion rules are untested and enabled by default when saved. You can edit or disable them; they are not platform-confirmed banned terms."),
    "AI建议（尚未实测，原因未证实）": (
        "AI 提案（未実測、原因未証明）", "AI suggestion (untested; cause unproven)"),
    "分析已完成，未提出局部修改": (
        "分析完了、部分変更の提案なし", "Analysis complete; no local edits proposed"),
    "分析已完成，已保存局部建议规则（尚未实测）": (
        "分析完了、部分的な提案ルールを保存（未実測）", "Analysis complete; local suggestion rules saved (untested)"),
    "编辑AI建议规则：尚未实测，可修改或停用，保存后生效。": (
        "AI 提案ルールを編集：未実測です。変更または無効化し、保存すると適用されます。", "Edit an AI suggestion rule: it is untested. Change or disable it, then save to apply."),
    "处理记录归档不代表弹幕已经送达。": (
        "対応記録のアーカイブはコメントの配信済みを示しません。", "Archiving a recovery record does not mean the message was delivered."),
    " · 处置 {0}": (" · 対応 {0}", " · Action {0}"),
    "正在自动分析和处理当前失败消息，队列顺序保持。": (
        "現在の失敗メッセージを自動分析・対応中です。キューの順序は保持されます。", "Analyzing and handling the current failed message automatically while preserving queue order."),
    "仅修改发送弹幕的文字，不修改字幕。手工规则由你确认；自动恢复可使用 Codex 分析，局部修改通过检查并观察页面回显后才保存为经验规则。": (
        "送信するコメントの文字のみを変更し、字幕は変更しません。手動ルールは利用者が確認します。自動復旧は Codex の分析を使用でき、部分変更は検査とページでの表示確認後に経験ルールとして保存されます。",
        "Change outgoing chat text only; subtitles stay unchanged. You confirm manual rules. Automatic recovery can use Codex analysis; local edits become empirical rules only after checks and observed page visibility."),
    "失败记录会跟随队列自动更新处理状态。返回 f 不代表句中存在禁词，词库候选不会自动生效；AI经验规则也不代表已确认禁词。": (
        "失敗記録はキューの処理状況に合わせて自動更新されます。f 応答は文に禁止語があることを示しません。辞書の候補は自動適用されず、AI の経験ルールも禁止語の証明にはなりません。",
        "Failure records update automatically with queue processing. An f response does not prove banned terms are present. Lexicon candidates do not activate automatically, and empirical AI rules are not confirmed banned terms."),
    "正在请求 Codex 分析失败正文": (
        "失敗した本文の分析を Codex に依頼中", "Asking Codex to analyze the failed text"),
    "正在应用已检查的局部修改": (
        "検査済みの部分変更を適用中", "Applying checked local edits"),
    "按发送间隔自动重试当前条": (
        "送信間隔に従って現在のメッセージを自動再送", "Automatically retrying the current message at the send interval"),
    "自动只读核验发送结果": (
        "送信結果を読み取り専用で自動確認", "Automatically checking send results without resending"),
    "自动复检登录和页面状态": (
        "ログインとページ状態を自動再確認", "Automatically rechecking sign-in and page state"),
    "按账号允许长度重新拆分": (
        "アカウントの許容長に合わせて再分割", "Splitting again to the account's allowed message length"),
    "自动检查浏览器和页面状态": (
        "ブラウザーとページ状態を自動確認", "Automatically checking the browser and page state"),
    "保留消息，等待登录或验证": (
        "メッセージを保持してログインまたは認証を待機", "Message retained; waiting for sign-in or verification"),
    "保留消息，请修复本地名单文件": (
        "メッセージを保持しました。ローカルルールファイルを修復してください", "Message retained; repair the local rules file"),
    "自动处理当前失败消息": (
        "現在の失敗メッセージを自動対応", "Automatically handling the current failed message"),
    "AI建议已通过本地检查 等待只读核验后修正": (
        "AI 提案はローカル検査済みです。読み取り専用の確認後に修正します", "AI suggestions passed local checks; awaiting read-only verification before editing"),
    "AI建议不可用 继续原有自动恢复": (
        "AI 提案を利用できません。既存の自動復旧を継続します", "AI suggestions are unavailable; continuing existing automatic recovery"),
    "已应用AI局部修改 按发送间隔自动重试": (
        "AI の部分変更を適用しました。送信間隔に従って自動再送します", "AI local edits applied; retrying automatically at the send interval"),
    "原文出现回显 不学习过滤词": (
        "原文がページに表示されました。置換ルールは学習しません", "Original text appeared on the page; no filter rule is learned"),
    "修改后出现回显 已保存局部经验规则": (
        "修正文がページに表示され、部分的な経験ルールを保存しました", "Edited text appeared on the page; a local empirical rule was saved"),
    "AI修正后可见（经验规则，原因未证实）": (
        "AI 修正後に表示（経験ルール、原因未証明）", "Visible after AI edits (empirical rule; cause unproven)"),
    "AI分析（原因尚未证实）": (
        "AI 分析（原因は未証明）", "AI analysis (cause unproven)"),
    "规则来源": ("ルールの出所", "Rule origin"),
    "手工确认": ("手動確認", "Manually confirmed"),
    "诊断分类：{0}": ("診断分類：{0}", "Diagnosis category: {0}"),
    "处理动作：{0}": ("対応方法：{0}", "Recovery action: {0}"),
    "原因未确定": ("原因不明", "Cause undetermined"),
    "AI局部修正后的页面可见结果仅作为经验规则，不代表该词已被平台证实禁用；可编辑或停用。": (
        "AI の部分修正後の表示結果は経験ルールのみです。プラットフォームの禁止語が証明されたことを示しません。編集または無効化できます。",
        "Page visibility after local AI edits is empirical only, not proof that the platform bans a term. You can edit or disable the rule."),
    "发送结果未知，正在自动只读核验": (
        "送信結果不明、自動で読み取り専用の確認中", "Send result unknown; automatically checking without resending"),
    "连续发送失败，正在自动复检账号和页面": (
        "送信が連続で失敗したため、アカウントとページを自動再確認中", "Repeated send failures; automatically rechecking the account and page"),
    "正在自动恢复当前条，保留队列顺序": (
        "キューの順序を保持して現在のメッセージを自動復旧中", "Recovering the current message automatically in queue order"),
    "自动核验中": ("自動確認中", "Checking automatically"),
    "正在自动只读核验，不会重复提交结果未知的消息。": (
        "読み取り専用で自動確認中です。結果不明のメッセージは再送しません。", "Checking automatically without resending messages whose send results are unknown."),
    " · 过滤规则 {0} · 未处理失败 {1}": (
        " · 置換ルール {0} · 未処理の失敗 {1}", " · Filter rules {0} · Unhandled failures {1}"),
    " · 已处理历史 {0}": (" · 処理済み履歴 {0}", " · Handled history {0}"),
    "保存后生效。启动后请手动打开正式直播间，不自动启动浏览器；只发送终稿，每段【译文】连同括号须为 11 到 40 字符。正文去标点，断句用空格，保留外层【】。10 秒内重复内容跳过，发送间隔至少 3.5 秒。本轮队列不过期；返回 f 后按发送间隔自动重试当前条，保留顺序；连续失败自动复检登录和页面状态，结果未知时自动只读核验；验证码、禁言和明确拒绝仍需人工处理，消息保留。": (
        "保存後に適用します。起動後は本番の配信ルームを手動で開いてください。ブラウザーは自動起動しません。確定訳のみ送信し、各【翻訳】は括弧込みで 11〜40 文字です。本文の句読点を除き、文の区切りは空白にして外側の【】を保持します。10 秒以内の重複を除外し、送信間隔は 3.5 秒以上です。この起動中の待機メッセージは期限切れになりません。f 応答の場合は送信間隔に従って現在のコメントを自動再送し、順序を保持します。連続失敗時はログインとページ状態を自動再確認し、結果不明時は読み取り専用で自動確認します。認証、投稿禁止、明確な拒否は手動対応が必要で、メッセージは保持します。",
        "Save to apply. After launch, open the production room manually; the browser does not start automatically. Send finals only, with 11 to 40 characters including brackets per segment. Remove body punctuation, separate sentences with spaces and retain the outer 【】. Skip duplicates within 10 seconds and wait at least 3.5 seconds between sends. Queued messages do not expire during this run. After f, retry the current message automatically at the send interval, preserving queue order. Repeated failures trigger automatic account and page checks; unknown results are checked without resending. Verification, mute restrictions and explicit rejections require manual action; messages are retained."),
    "失败记录会跟随队列自动更新处理状态。返回 f 不代表句中存在禁词，候选不会自动生效，确认原词并保存规则后才参与过滤。": (
        "失敗記録はキューの処理状況に合わせて自動更新されます。f 応答は文に禁止語があることを示しません。候補は自動適用されず、原語を確認してルールを保存した後に適用されます。",
        "Failure records update automatically with queue processing. An f response does not prove banned terms are present. Candidates only become filter rules after you confirm a term and save it."),
    "显示已处理历史": ("処理済み履歴を表示", "Show handled history"),
    "失败原句": ("失敗した原文", "Original failed text"),
    "处理状态": ("処理状況", "Recovery status"),
    "处置说明": ("対応内容", "Recovery details"),
    "失败处理记录": ("失敗対応記録", "Failure recovery records"),
    "选择失败句查看自动诊断、处理进展和本地检测结果": (
        "失敗した文を選択して自動診断、対応状況、ローカル検査結果を表示", "Select a failed sentence to view automatic diagnosis, recovery progress and local screening"),
    "已加载 {0} 条规则、{1} 条失败处理记录。": (
        "ルール {0} 件、失敗対応記録 {1} 件を読み込みました。", "Loaded {0} rules and {1} failure recovery records."),
    "未处理 {0} 条 · 已处理历史 {1} 条": (
        "未処理 {0} 件 · 処理済み履歴 {1} 件", "Unhandled {0} · Handled history {1}"),
    "待处理": ("対応待ち", "Awaiting recovery"),
    "自动处理中": ("自動対応中", "Recovering automatically"),
    "等待结果确认": ("結果確認待ち", "Awaiting result confirmation"),
    "已处理（历史）": ("処理済み（履歴）", "Handled (history)"),
    "需要登录或人工处理": ("ログインまたは手動対応が必要", "Sign-in or manual action required"),
    "页面出现同文，来源未确认": (
        "ページに同じ文が表示されましたが、送信元は未確認です", "Matching text appeared on the page; its source is unconfirmed"),
    "等待发送队列自动处理": ("送信キューの自動対応待ち", "Waiting for automatic queue recovery"),
    "按队列顺序自动恢复当前条": ("キューの順序で現在のメッセージを自動復旧", "Recovering the current message automatically in queue order"),
    "保留消息，自动检查发送结果": ("メッセージを保持して送信結果を自動確認", "Message retained; checking send results automatically"),
    "处理完成，原始记录保留": ("対応完了、元の記録は保持", "Recovery complete; original record retained"),
    "保留队列，等待登录或验证": ("キューを保持してログインまたは認証を待機", "Queue retained; waiting for sign-in or verification"),
    "处理状态：{0}": ("処理状況：{0}", "Recovery status: {0}"),
    "处置说明：{0}": ("対応内容：{0}", "Recovery details: {0}"),
    "关联队列编号：{0}": ("関連キュー番号：{0}", "Linked queue ID: {0}"),
    "处理时间（UTC+8）：{0}": ("対応日時（UTC+8）：{0}", "Handled at (UTC+8): {0}"),
    "页面出现同文只代表页面观察结果，不能证明由程序自动发送；人工发送也可能产生相同文字。": (
        "同じ文の表示はページでの観測結果のみです。プログラムからの自動送信を証明せず、手動送信でも同じ文が表示されます。",
        "Matching text is only a page observation, not proof of an automatic send. A manual send can produce the same text."),
    "仅修改发送弹幕的文字，不修改字幕。替换或删除由你逐条确认，不调用 Codex 自动生成；中性替换模式留空时使用“已过滤”。": (
        "送信するコメントの文字だけを変更し、字幕は変更しません。置換または削除は各ルールで手動確認し、Codex で自動生成しません。中立的な置換先が空の場合は「已过滤」を使用します。",
        "Change outgoing chat text only; subtitles stay unchanged. Confirm each replacement or removal yourself, without automatic Codex generation. An empty neutral replacement uses 已过滤."),
    "实测确认后可选择“从弹幕删除”，仅删除弹幕正文中的原词，字幕终稿保持原文。": (
        "実測で確認した後に「コメントから削除」を選択できます。原語はコメント本文からのみ削除され、確定字幕は変更されません。",
        "After confirming the observed behavior, choose Remove from chat to remove the original term only from outgoing chat. Final subtitles retain the original text."),
    "新增规则：填写原词，选择中性替换或从弹幕删除，保存后生效。": (
        "新規ルール：原語を入力し、中立的な置換またはコメントからの削除を選択して保存してください。",
        "New rule: enter the original term, choose a neutral replacement or removal from chat, then save to apply it."),
    "中性替换": ("中立的な置換", "Neutral replacement"),
    "从弹幕删除": ("コメントから削除", "Remove from chat"),
    "处理方式": ("処理方法", "Action"),
    "处理结果": ("処理結果", "Result"),
    "已确认删除规则：{0}（只从弹幕删除）": (
        "確認済み削除ルール：{0}（コメントからのみ削除）", "Confirmed removal rule: {0} (remove from chat only)"),
    "名单和失败候选保存在本机用户数据目录，不进入分发包。发送中或结果未知的消息不会因编辑规则而改写；已确认失败的消息重试前应用最新规则。": (
        "ルールと失敗候補はこの PC のユーザーデータに保存され、配布パッケージには入りません。送信中または結果不明のメッセージはルールの編集で書き換えられません。失敗が確認されたメッセージは再試行前に最新のルールを適用します。",
        "Rules and failed candidates are stored in this PC's user data, outside distribution packages. Editing rules does not rewrite messages being sent or with unknown send results. Confirmed failed messages use the latest rules before retrying."),
    "检测结果": ("検査結果", "Screening result"),
    "已检测": ("検査済み", "Screened"),
    "未配置词库": ("辞書未設定", "No lexicon configured"),
    "检测失败": ("検査失敗", "Screening failed"),
    "尚未检测": ("未検査", "Not yet screened"),
    "已完成本地检测：命中当前直播间的实测触发词。": (
        "ローカル検査完了：この配信ルームで実測した拒否対象語が一致しました。", "Local screening complete: a term observed to trigger rejection in this live room matched."),
    "选择失败句查看自动本地检测结果": ("失敗した文を選択してローカル検査結果を表示", "Select a failed sentence to view its local screening results"),
    "社区参考词库仅检测提示，命中不代表 B 站禁用；不会自动替换。游戏用词容易误报，可将确认正常的词加入白名单。": (
        "コミュニティ辞書は検査と通知専用です。一致しても Bilibili の禁止語とは限らず、自動置換しません。ゲーム用語の誤検出は許可リストで除外できます。",
        "The community lexicon provides screening hints only. A match does not prove Bilibili bans the term, and it does not auto-replace text. Allowlist confirmed ordinary game terms to reduce false positives."),
    "搜索参考词、分类或来源": ("参照語、分類、出典を検索", "Search reference terms, categories or sources"),
    "参考词": ("参照語", "Reference term"),
    "分类": ("分類", "Category"),
    "词库来源": ("辞書の出典", "Lexicon source"),
    "检测白名单（每行一个完整词，保存后重新检测失败句）": (
        "検査用許可リスト（1行に1語、保存後に失敗文を再検査）", "Screening allowlist (one whole term per line; failed sentences are rescanned after saving)"),
    "重新加载词库并检测": ("辞書を再読込して検査", "Reload lexicon and screen"),
    "保存检测白名单": ("検査用許可リストを保存", "Save screening allowlist"),
    "参考词库与检测": ("参照辞書と検査", "Reference lexicon and screening"),
    "未配置本地参考词库，尚未进行词库检测。": (
        "ローカル参照辞書が未設定のため、辞書検査は未実施です。", "No local reference lexicon is configured; lexicon screening has not run."),
    "本地词库不可用，检测失败；字幕与原队列继续保留。": (
        "ローカル辞書を利用できず、検査に失敗しました。字幕と既存キューは保持されます。", "Local lexicon is unavailable; screening failed. Subtitles and the original queue are retained."),
    "已完成本地检测：未命中参考词或已确认规则。发送失败原因仍未确定。": (
        "ローカル検査完了：参照語や確認済みルールの一致はありません。送信失敗の原因は不明です。", "Local screening complete: no reference terms or confirmed rules matched. The send failure cause remains unknown."),
    "已完成本地检测：发现参考词命中，请结合上下文核对。": (
        "ローカル検査完了：参照語が一致しました。文脈を確認してください。", "Local screening complete: reference terms matched; review the context."),
    "已完成本地检测：未命中参考词，但命中已确认规则。发送失败原因仍未确定。": (
        "ローカル検査完了：参照語の一致はありませんが、確認済みルールが一致しました。送信失敗の原因は不明です。", "Local screening complete: no reference terms matched, but confirmed rules matched. The send failure cause remains unknown."),
    "尚未检测，请刷新列表。": ("未検査です。一覧を更新してください。", "Not yet screened; refresh the list."),
    "检测时间（UTC+8）：{0}": ("検査日時（UTC+8）：{0}", "Screened at (UTC+8): {0}"),
    "参考命中：{0} · 原文：{1} · 分类：{2} · 来源：{3} · 位置：{4}–{5}": (
        "参照一致：{0} · 原文：{1} · 分類：{2} · 出典：{3} · 位置：{4}–{5}", "Reference match: {0} · Original: {1} · Category: {2} · Source: {3} · Position: {4}–{5}"),
    "已确认规则：{0} → {1}": ("確認済みルール：{0} → {1}", "Confirmed rule: {0} → {1}"),
    "命中过多，仅显示前 100 项。": ("一致が多いため、最初の100件のみ表示します。", "Too many matches; showing the first 100."),
    "检测结果不等于平台判定；未命中也不能保证发送成功。": (
        "検査結果はプラットフォームの判定ではありません。一致がなくても送信成功は保証されません。", "Screening is not a platform verdict; no matches do not guarantee delivery."),
    "本地参考词库：{0} 个词 · {1} 个来源 · 离线检测": (
        "ローカル参照辞書：{0} 語 · {1} 出典 · オフライン検査", "Local reference lexicon: {0} terms · {1} sources · Offline screening"),
    "白名单保存失败，原词库未修改。": (
        "許可リストを保存できません。既存の辞書は変更されていません。", "Failed to save the allowlist; the original lexicon is unchanged."),
    "检测白名单已保存，失败句已重新检测。": (
        "検査用許可リストを保存し、失敗した文を再検査しました。", "Screening allowlist saved; failed sentences have been rescanned."),
    "本地过滤词管理": ("ローカル置換ルール管理", "Local text filter rules"),
    "过滤名单异常，待发消息保留": (
        "ローカルルールに問題があります。待機メッセージは保持されます", "Filter rules have an error; queued messages are retained"),
    " · 过滤名单异常，待发消息保留": (
        " · ローカルルールに問題があり、待機メッセージを保持", " · Filter rule error; queued messages retained"),
    " · 过滤规则 {0} · 待分析失败句 {1}": (
        " · 置換ルール {0} · 分析待ち失敗文 {1}", " · Filter rules {0} · Failed sentences for review {1}"),
    " · 失败候选保存失败，请检查本地文件": (
        " · 失敗候補を保存できません。ローカルファイルを確認してください", " · Failed to save failure candidates; check local files"),
    "仅修改发送弹幕的文字，不修改字幕。替换词由你逐条确认，不调用 Codex 自动生成；替换内容留空时使用“已过滤”。": (
        "送信するコメントの文字だけを変更し、字幕は変更しません。置換語は各ルールで手動確認し、Codex で自動生成しません。置換先が空の場合は「已过滤」を使用します。",
        "Change outgoing chat text only; subtitles stay unchanged. Confirm each replacement yourself, without automatic Codex generation. An empty replacement uses 已过滤."),
    "失败句仅供分析，返回 f 不代表句中存在禁词。候选不会自动生效，确认原词并保存规则后才参与过滤。": (
        "失敗した文は分析用です。f 応答は文に禁止語があることを示しません。候補は自動適用されず、原語を確認してルールを保存した後に適用されます。",
        "Failed sentences are for review; f does not prove that a sentence contains a banned term. Candidates do not activate automatically. Confirm the original term and save its rule to apply it."),
    "启用本地过滤名单": ("ローカル置換ルールを有効化", "Enable local filter rules"),
    "保存名单开关": ("有効設定を保存", "Save filter enabled setting"),
    "刷新列表": ("一覧を更新", "Refresh lists"),
    "启用": ("有効", "Enabled"),
    "原词": ("原語", "Original term"),
    "中性替换词": ("中立的な置換語", "Neutral replacement"),
    "适用直播间": ("適用する配信ルーム", "Applicable live room"),
    "更新时间（UTC+8）": ("更新日時（UTC+8）", "Updated at (UTC+8)"),
    "已保存词规则": ("保存済みの置換ルール", "Saved term rules"),
    "待分析失败句": ("分析待ちの失敗した文", "Failed sentences for review"),
    "失败次数": ("失敗回数", "Failure count"),
    "最近失败（UTC+8）": ("最終失敗日時（UTC+8）", "Last failure (UTC+8)"),
    "失败原因": ("失敗理由", "Failure reason"),
    "选择失败句查看完整内容；此处不是已启用规则": (
        "失敗した文を選択すると全文を表示します。ここは有効なルールの一覧ではありません",
        "Select a failed sentence to see its full text; these are not active rules"),
    "从候选填写规则": ("候補からルールを入力", "Prepare rule from candidate"),
    "新增规则：填写原词和中性替换词，保存后用于后续发送。": (
        "新規ルール：原語と中立的な置換語を入力すると、保存後の送信に適用されます。",
        "New rule: enter the original term and neutral replacement; save to apply it to subsequent sends."),
    "留空使用“已过滤”": ("空欄の場合は「已过滤」を使用", "Leave empty to use 已过滤"),
    "本直播间": ("この配信ルーム", "This live room"),
    "全部直播间": ("すべての配信ルーム", "All live rooms"),
    "所有已保存规则统一适用于全部直播间；失败记录中的房间号仅表示来源和验证记录。": (
        "保存済みのすべてのルールは全配信ルームに共通で適用されます。失敗記録のルーム番号は発生元と検証記録のみを示します。",
        "All saved rules apply to every live room. Room IDs in failure records identify only their source and validation history."),
    "数字直播间号": ("数字の配信ルーム番号", "Numeric live room ID"),
    "适用范围": ("適用範囲", "Scope"),
    "启用此规则": ("このルールを有効化", "Enable this rule"),
    "新增规则": ("新規ルール", "New rule"),
    "保存规则": ("ルールを保存", "Save rule"),
    "删除所选规则": ("選択したルールを削除", "Delete selected rule"),
    "本地名单文件：{0}": ("ローカルルールファイル：{0}", "Local rules file: {0}"),
    "名单和失败候选保存在本机用户数据目录，不进入分发包。已提交或结果未知的消息不会因编辑规则而改写。": (
        "ルールと失敗候補はこの PC のユーザーデータに保存され、配布パッケージには入りません。送信済みまたは結果不明のメッセージはルールの編集で書き換えられません。",
        "Rules and failed candidates are stored in this PC's user data, outside distribution packages. Editing rules does not rewrite messages already submitted or with unknown send results."),
    "过滤词不能为空": ("原語を入力してください", "The original term cannot be empty"),
    "过滤词不能超过80个字符（按 UTF-16 计数）": (
        "原語は UTF-16 換算で80文字以下にしてください", "The original term cannot exceed 80 UTF-16 units"),
    "替换内容不能超过80个字符（按 UTF-16 计数）": (
        "置換先は UTF-16 換算で80文字以下にしてください", "The replacement cannot exceed 80 UTF-16 units"),
    "过滤词不能包含控制字符或隐藏字符": (
        "原語に制御文字や隠し文字は使用できません", "The original term cannot contain control or hidden characters"),
    "替换内容不能包含控制字符或隐藏字符": (
        "置換先に制御文字や隠し文字は使用できません", "The replacement cannot contain control or hidden characters"),
    "该直播间范围已存在相同过滤词，请编辑原规则": (
        "同じ適用範囲の原語がすでに存在します。既存のルールを編集してください", "This term already exists in that room scope; edit its existing rule"),
    "已存在相同过滤词，请编辑原规则": (
        "同じ原語がすでに存在します。既存のルールを編集してください", "This term already exists; edit its existing rule"),
    "确认的过滤词必须来自所选失败内容": (
        "確認した原語は選択した失敗文に含まれている必要があります", "The confirmed term must come from the selected failed sentence"),
    "请填写数字直播间号，或选择全部直播间。": (
        "数字の配信ルーム番号を入力するか、すべての配信ルームを選択してください。", "Enter a numeric live room ID or choose all live rooms."),
    "无法保存本地过滤名单，原规则未修改。": (
        "ローカルルールを保存できません。既存ルールは変更されていません。", "Could not save local filter rules; existing rules are unchanged."),
    "过滤规则操作失败，请检查内容和本地名单文件。": (
        "ルール操作に失敗しました。内容とローカルルールファイルを確認してください。", "Filter rule operation failed; check the content and local rules file."),
    "本地过滤名单暂不可用，请检查词库文件。": (
        "ローカルルールを利用できません。ルールファイルを確認してください。", "Local filter rules are unavailable; check the rules file."),
    "已启用": ("有効", "Enabled"),
    "已停用": ("無効", "Disabled"),
    "平台返回 f（原因未确定）": ("f 応答（理由不明）", "Platform returned f (reason undetermined)"),
    "平台拒绝（原因待确认）": ("プラットフォームが拒否（理由は要確認）", "Platform rejected the message (reason needs review)"),
    "发送结果未知": ("送信結果不明", "Send result unknown"),
    "发送失败（原因待确认）": ("送信失敗（理由は要確認）", "Send failed (reason needs review)"),
    "已加载 {0} 条规则、{1} 条待分析失败句。": (
        "ルール {0} 件、分析待ちの失敗文 {1} 件を読み込みました。", "Loaded {0} rules and {1} failed sentences for review."),
    "编辑已保存规则：更改后点击保存规则。": (
        "保存済みルールを編集し、変更後に「ルールを保存」をクリックしてください。", "Edit a saved rule, then click Save rule."),
    "请填写已确认需要过滤的原词；失败句不会自动成为规则。": (
        "置換が必要と確認した原語を入力してください。失敗文は自動でルールになりません。", "Enter the confirmed original term to filter; the failed sentence does not become a rule automatically."),
    "过滤规则已保存，用于后续发送。": (
        "置換ルールを保存しました。今後の送信に適用されます。", "Filter rule saved for subsequent sends."),
    "过滤名单开关已保存。": ("ルールの有効設定を保存しました。", "Filter enabled setting saved."),
    "删除选中的本地过滤规则？后续弹幕将不再使用这条规则。": (
        "選択したローカルルールを削除しますか？今後のコメント送信にこのルールは適用されなくなります。", "Delete the selected local filter rule? Subsequent chat messages will no longer use it."),
    "所选过滤规则已删除。": ("選択した置換ルールを削除しました。", "Selected filter rule deleted."),
    "直播浏览器": ("ライブブラウザー", "Live browser"),
    "直播弹幕设置": ("ライブコメント設定", "Live chat settings"),
    "弹幕队列暂不可用，字幕翻译继续运行。": (
        "コメント待機キューが利用できません。字幕翻訳は継続します。",
        "The chat queue is unavailable; subtitle translation continues."),
    "弹幕设置未生效，请检查直播间和队列目录。": (
        "コメント設定を適用できません。配信ルームと待機キューの保存先を確認してください。",
        "Chat settings were not applied; check the live room and queue directory."),
    "弹幕推送：{0} · 待发 {1} · 页面可见 {2}": (
        "コメント送信：{0} · 待機 {1} · ページ表示 {2}",
        "Chat sending: {0} · Queued {1} · Page-visible {2}"),
    "等待终稿": ("確定訳待ち", "Waiting for final"),
    "正在发送": ("送信中", "Sending"),
    "正在核验": ("確認中", "Verifying"),
    "等待浏览器连接": ("ブラウザー接続待ち", "Waiting for browser connection"),
    "等待登录": ("ログイン待ち", "Waiting for sign-in"),
    "待核实，队列保留": ("確認待ち、待機メッセージは保持", "Awaiting review; queue retained"),
    "同步终稿译文到直播间": ("確定訳を配信ルームに送信", "Send final translations to the live room"),
    "发送来源": ("送信する音声入力", "Translation source"),
    "电脑声音的译文": ("PC 音声の翻訳", "Computer audio translations"),
    "麦克风的译文": ("マイク音声の翻訳", "Microphone translations"),
    "两路译文": ("両方の音声入力の翻訳", "Translations from both sources"),
    "发送间隔": ("送信間隔", "Send interval"),
    " 秒": (" 秒", " s"),
    "保存推送设置": ("送信設定を保存", "Save sending settings"),
    "恢复推送": ("送信を再開", "Resume sending"),
    "清空待发队列": ("未送信キューを消去", "Clear unsent queue"),
    "将清理 {0} 段尚未发送的消息。已提交或结果未确认的 {1} 段消息保留，不会撤回已发送弹幕。是否继续？": (
        "未送信の {0} 件を消去します。送信済みまたは結果未確認の {1} 件は保持され、送信済みコメントは取り消しません。続行しますか？",
        "Clear {0} unsent messages. Keep {1} submitted or unconfirmed messages; already sent chat will not be recalled. Continue?"),
    "待发队列已清理": ("未送信キューを消去しました", "Unsent queue cleared"),
    " · 已清理 {0} 段未发送消息": (" · 未送信 {0} 件を消去済み", " · Cleared {0} unsent messages"),
    " · 平台码 {0}": (" · プラットフォームコード {0}", " · Platform code {0}"),
    " · 当前单条长度 {0}（UTF-16）": (
        " · 現在のメッセージ長 {0}（UTF-16）", " · Current message length {0} (UTF-16)"),
    " · 当前账号加权上限 {0}": (
        " · 現在のアカウントの加重上限 {0}", " · Current account weighted limit {0}"),
    "平台返回 f，按发送间隔自动重试当前条": (
        "プラットフォームが f を返しました。送信間隔に従って現在のコメントを自動再送します",
        "Platform returned f; automatically retrying the current message at the send interval"),
    "平台返回 f，等待自动重试当前条": (
        "プラットフォームが f を返しました。現在のコメントの自動再送を待機中です",
        "Platform returned f; waiting to retry the current message automatically"),
    "正在自动重试当前条，保留队列顺序": (
        "待機順序を保持して現在のコメントを自動再送中です",
        "Automatically retrying the current message in queue order"),
    " · f 自动重试累计 {0} 次": (
        " · f 自動再送合計 {0} 回", " · Total f retries {0}"),
    "自动重试中": ("自動再送中", "Retrying automatically"),
    "按发送间隔自动重试当前条，无需手动恢复。": (
        "送信間隔に従って現在のコメントを自動再送します。手動での再開は不要です。",
        "Retry the current message automatically at the send interval; no manual resume is needed."),
    "核验待定消息": ("未確定メッセージを確認", "Verify pending message"),
    "只读核验页面是否收到待定消息，不重新发送。": (
        "未確定メッセージをページで受信したか読み取り専用で確認します。再送しません。",
        "Check whether the page received the pending message without resending it."),
    "弹幕推送未开启": ("コメント送信は無効です", "Chat sending is disabled"),
    "弹幕推送仅本机启用，分发版不开放": (
        "コメント送信はこの PC のみで利用でき、配布版では提供されません",
        "Chat sending is available on this PC only; distribution builds do not include access"),
    "保存后生效。开启时自动连接已登录的内置直播页；只发送终稿，每段【译文】含括号最多 40 字符。正文去标点，断句用空格，保留外层【】。10 秒内重复内容跳过，发送间隔至少 3.5 秒。队列不过期；返回 f 后按发送间隔自动重试当前条，保留顺序；其他平台拒绝或结果未知时保留消息等待核实。": (
        "保存後に適用します。有効にするとログイン済みの内蔵配信ページに自動接続します。確定訳のみ送信し、各【翻訳】は括弧込みで最大 40 文字です。本文の句読点を除き、文の区切りは空白にして外側の【】を保持します。10 秒以内の重複を除外し、送信間隔は 3.5 秒以上です。待機メッセージは期限切れになりません。f 応答の場合は送信間隔に従って現在のコメントを自動再送し、順序を保持します。その他の拒否や結果不明の場合は確認のため保持します。",
        "Save to apply. When enabled, connect automatically to the signed-in built-in live page. Send finals only, up to 40 characters including brackets per segment. Remove body punctuation, separate sentences with spaces and retain the outer 【】. Skip duplicates within 10 seconds and wait at least 3.5 seconds between sends. Queued messages do not expire. After f, retry the current message automatically at the send interval, preserving queue order. Other rejections or unknown results are retained for review."),
    "WebView2 Evergreen 由微软自动下载更新。安装新版后重新打开浏览器生效；联网检查失败时不会标记为最新版。": (
        "WebView2 Evergreen は Microsoft が自動更新します。新バージョンはブラウザーの再起動後に有効になります。オンライン確認に失敗した場合は最新版とは表示しません。",
        "Microsoft automatically updates WebView2 Evergreen. Reopen the browser to use an installed update. Failed online checks are not marked up to date."),
    "直播间格式或推送设置无效": ("配信ルームまたは送信設定が無効です", "Invalid live room or sending settings"),
    "推送设置已保存": ("送信設定を保存しました", "Sending settings saved"),
    "等待已登录的目标直播页": ("ログイン済みの対象配信ページを待機中", "Waiting for the signed-in target live page"),
    "等待终稿译文": ("確定訳を待機中", "Waiting for a final translation"),
    "正在核实账号登录状态": ("アカウントのログイン状態を確認中", "Checking account sign-in status"),
    "等待内置浏览器连接": ("内蔵ブラウザーの接続を待機中", "Waiting for the built-in browser connection"),
    "请在内置浏览器登录账号": ("内蔵ブラウザーでログインしてください", "Sign in in the built-in browser"),
    "待发消息属于其他直播间，请核对设置": (
        "待機メッセージは別の配信ルーム宛てです。設定を確認してください",
        "Queued messages belong to another room; check the settings"),
    "队列保存失败，请检查本机存储": (
        "待機メッセージを保存できません。ローカルストレージを確認してください",
        "Could not save the queue; check local storage"),
    "正在只读核实页面是否收到弹幕": (
        "ページでコメントを受信したか読み取り専用で確認中",
        "Checking page receipt without sending another message"),
    " · 还需等待 {0:.1f} 秒": (" · あと {0:.1f} 秒待機", " · Wait {0:.1f}s more"),
    "按队列等待发送": ("送信待ちの順番を待機中", "Waiting in the send queue"),
    "正在提交弹幕": ("コメントを送信中", "Submitting chat message"),
    "接口回执已接收，尚未确认页面收到": (
        "API の応答を受信しました。ページでの受信は未確認です",
        "API acknowledgement received; page receipt is unconfirmed"),
    "直播页面已收到弹幕": ("配信ページでコメントを受信しました", "The live page received the chat message"),
    "等待发送间隔": ("送信間隔を待機中", "Waiting for the send interval"),
    "推送已暂停，队列保留": ("送信を一時停止しました。待機メッセージは保持されています", "Sending paused; queue retained"),
    "平台拒绝，消息保留等待核实": ("プラットフォームが拒否しました。確認のためメッセージを保持します", "Platform rejected the send; message retained for review"),
    "发送结果未知，消息保留等待核实": ("送信結果が不明です。確認のためメッセージを保持します", "Send result unknown; message retained for review"),
    "{0} · 待发 {1} 段 · 接口回执 {2} · 页面收到 {3}": (
        "{0} · 待機 {1} 件 · API 応答 {2} · ページ受信 {3}",
        "{0} · Queued {1} · API acknowledgements {2} · Page receipts {3}"),
    "先核实直播页和账号状态。发送结果未知时不会自动重发。": (
        "配信ページとアカウントの状態を先に確認してください。送信結果が不明な場合は自動再送しません。",
        "Check the live page and account status first. Unknown send results are not resent automatically."),
    "直播间": ("配信ルーム", "Live room"),
    "登录账号": ("ログイン", "Sign in"),
    "打开直播间": ("配信ルームを開く", "Open live room"),
    "刷新网页": ("ページを更新", "Reload page"),
    "关闭浏览器": ("ブラウザーを閉じる", "Close browser"),
    "首次扫码登录一次，后续复用独立登录状态。无需安装字幕弹幕桥扩展。": (
        "初回に QR コードでログインすると、専用のログイン状態が保存されます。字幕ブリッジ拡張は不要です。",
        "Scan to sign in once. Your dedicated session is retained; no subtitle bridge extension is needed."),
    "直播间号或 HTTPS 直播间地址": ("ルーム番号または HTTPS URL", "Room number or HTTPS live room URL"),
    "自动检查浏览器内核更新": ("ブラウザーエンジンの更新を自動確認", "Automatically check browser engine updates"),
    "正在读取浏览器版本…": ("ブラウザーのバージョンを確認中…", "Reading browser versions…"),
    "浏览器尚未打开": ("ブラウザーは未起動です", "Browser has not been opened"),
    "检查更新": ("更新を確認", "Check for updates"),
    "微软官方内核下载": ("Microsoft 公式エンジンダウンロード", "Microsoft runtime download"),
    "WebView2 Evergreen 由微软自动下载更新。安装新版后重新打开浏览器生效；联网检查失败时不会标记为最新版。本窗口只管理浏览器，尚未接入弹幕自动发送。": (
        "WebView2 Evergreen は Microsoft が自動更新します。新バージョンはブラウザーの再起動後に有効になります。オンライン確認に失敗した場合は最新版とは表示しません。この画面はブラウザー管理のみで、コメント自動送信は未接続です。",
        "Microsoft automatically updates WebView2 Evergreen. Reopen the browser to use an installed update. Failed online checks are not marked up to date. This window manages the browser; automatic chat sending is not connected yet."),
    "直播间格式无效": ("配信ルームの形式が無効です", "Invalid live room format"),
    "浏览器设置保存失败": ("ブラウザー設定を保存できません", "Could not save browser settings"),
    "正在检查浏览器内核更新…": ("ブラウザーエンジンの更新を確認中…", "Checking browser engine updates…"),
    "正在启动内置浏览器…": ("内蔵ブラウザーを起動中…", "Starting built-in browser…"),
    "内置浏览器已打开": ("内蔵ブラウザーが開きました", "Built-in browser is open"),
    "浏览器已关闭，登录状态保留": ("ブラウザーを閉じました。ログイン状態は保存されています", "Browser closed; sign-in session retained"),
    "本机已有新版内核，关闭并重新打开浏览器后生效": (
        "新しいエンジンがインストール済みです。ブラウザーを閉じて開き直すと有効になります",
        "A newer engine is installed. Close and reopen the browser to use it"),
    "浏览器启动或连接失败：{0}。请检查 WebView2 Evergreen 是否已安装。": (
        "ブラウザーの起動・接続に失敗しました：{0}。WebView2 Evergreen のインストールを確認してください。",
        "Browser startup or connection failed: {0}. Check whether WebView2 Evergreen is installed."),
    "正在运行：{0} · 本机可用：{1}": ("実行中：{0} · ローカルで利用可能：{1}", "Running: {0} · Available locally: {1}"),
    "正在运行：{0} · 本机安装：{1} · 微软目录：{2}": (
        "実行中：{0} · インストール済み：{1} · Microsoft 公開情報：{2}",
        "Running: {0} · Installed: {1} · Microsoft catalog: {2}"),
    "未启动": ("未起動", "Not running"),
    "未确认": ("未確認", "Unconfirmed"),
    "未安装": ("未インストール", "Not installed"),
    "请先安装 WebView2 Evergreen 内核": ("WebView2 Evergreen をインストールしてください", "Install the WebView2 Evergreen runtime first"),
    "尚未联网检查最新版": ("最新版のオンライン確認は未実施です", "Latest version has not been checked online"),
    "联网版本检查未完成，不能确认是否为最新版": (
        "オンライン確認が完了していないため、最新版かどうかは未確認です",
        "Online version check did not complete; latest version is unconfirmed"),
    "微软目录已有新版，Evergreen 将按微软更新策略更新": (
        "Microsoft 公開情報に新バージョンがあります。Evergreen は Microsoft の方針に従って更新されます",
        "Microsoft lists a newer version. Evergreen updates according to Microsoft's rollout policy"),
    "本机版本不低于微软公开目录版本": (
        "ローカルバージョンは Microsoft 公開情報のバージョン以上です",
        "Local version is at least as new as Microsoft's public catalog"),
    "最近终稿：全程 {0:.2f} 秒 · 句尾等待 {1:.2f} 秒": (
        "直近の確定訳：全体 {0:.2f} 秒 · 発話終了後 {1:.2f} 秒",
        "Latest final: total {0:.2f}s · after audio {1:.2f}s"),
    "最近暂译：全程 {0:.2f} 秒": ("直近の仮訳：全体 {0:.2f} 秒", "Latest draft: total {0:.2f}s"),
    "全程含本段音频时长；句尾等待从音频结束计时，均到原文确认后终稿标签完成 Qt 绘制。识别开始到终稿及 ASR、翻译排队、请求和 UI 各阶段耗时见下方或 LATENCY 日志。": (
        "全体は区間の音声の長さを含み、発話終了後は音声の終わりから、原文確定後の確定訳ラベルの Qt 描画完了までを計測します。認識開始から確定訳まで、ASR・翻訳待ち・リクエスト・UI の各時間は以下または LATENCY ログで確認できます。",
        "Total includes the segment's audio duration; after audio starts at its end. Both end when Qt paints the final label after source confirmation. Recognition-to-final and ASR, translation queue, request and UI times appear below or in LATENCY logs."),
    "识别开始到终稿：{0:.2f} 秒": ("認識開始から確定訳：{0:.2f} 秒", "Recognition to final: {0:.2f}s"),
    "{0}：{1:.2f} 秒": ("{0}：{1:.2f} 秒", "{0}: {1:.2f}s"),
    "语音分段": ("音声区切り", "Segmentation"),
    "识别排队": ("認識待ち", "ASR queue"),
    "讲话筛选": ("発話フィルター", "Speech filter"),
    "半句拼接": ("文の結合", "Sentence assembly"),
    "翻译排队": ("翻訳待ち", "Translation queue"),
    "翻译请求": ("翻訳リクエスト", "Translation request"),
    "原文确认": ("原文確定", "Source confirmation"),
    "界面分发": ("UI 配信", "UI dispatch"),
    "界面更新": ("UI 更新", "UI update"),
    "界面绘制": ("UI 描画", "UI paint"),
    "混合翻译（本地暂译 + Codex 终稿）": ("ハイブリッド翻訳（ローカル仮訳 + Codex 確定訳）", "Hybrid (local draft + Codex final)"),
    "本地模型与 Codex 同时翻译同一份原文。本地结果先显示为暂译，Codex 返回后替换为终稿。请在模型管理中选择已下载的本地翻译模型；Codex 不可用时保留暂译并标记未确认。": (
        "ローカルモデルと Codex が同じ原文を並行翻訳します。ローカルの仮訳を先に表示し、Codex の確定訳で更新します。モデル管理でダウンロード済みのローカル翻訳モデルを選択してください。Codex が利用できない場合は仮訳を未確認として保持します。",
        "The local model and Codex translate the same source in parallel. A local draft appears first and is replaced by the Codex final. Select a downloaded local translation model in Model Manager. If Codex is unavailable, the draft remains marked unconfirmed."),
    "暂译：{0}": ("仮訳：{0}", "Draft: {0}"),
    "终稿：{0}": ("確定訳：{0}", "Final: {0}"),
    "本地暂译 · 等待 Codex 终稿": ("ローカル仮訳 · Codex 確定訳待ち", "Local draft · Waiting for Codex final"),
    "本地暂译 · Codex 未确认": ("ローカル仮訳 · Codex 未確認", "Local draft · Codex unconfirmed"),
    "Codex 终稿": ("Codex 確定訳", "Codex final"),
    "Codex 译文 · 等待后半句确认": ("Codex 訳文 · 文の後半の確定待ち", "Codex translation · Waiting for sentence completion"),
    "译文 · 等待后半句确认": ("訳文 · 文の後半の確定待ち", "Translation · Waiting for sentence completion"),
    "终稿": ("確定訳", "Final"),
    "Codex 未确认 · 等待本地暂译": ("Codex 未確認 · ローカル仮訳待ち", "Codex unconfirmed · Waiting for local draft"),
    "本地暂译不可用 · 等待 Codex 终稿": ("ローカル仮訳不可 · Codex 確定訳待ち", "Local draft unavailable · Waiting for Codex final"),
    "（翻译未完成）": ("（翻訳未完了）", "(Translation incomplete)"),
    "翻译未完成": ("翻訳未完了", "Translation incomplete"),
    "本地与 Codex 翻译均未完成": ("ローカル・Codex 翻訳とも未完了", "Both local and Codex translations incomplete"),
    "正在准备本地暂译和 Codex 终稿…": ("ローカル仮訳と Codex 確定訳を準備中…", "Preparing local draft and Codex final…"),
    "正在听 · 本地暂译 / Codex 终稿": ("音声入力中 · ローカル仮訳 / Codex 確定訳", "Listening · Local draft / Codex final"),
    "界面语言": ("表示言語", "Interface language"),
    "界面语言（重新启动后生效）": ("表示言語（再起動後に反映）", "Interface language (applies after restart)"),
    "界面语言与识别语言、翻译方向相互独立。保存后重新启动软件生效。": (
        "表示言語は音声認識の言語・翻訳方向とは別の設定です。保存後、アプリを再起動すると反映されます。",
        "Interface language is independent of recognition language and translation direction. Save and restart to apply."),
    "界面语言已保存，重新启动软件后生效。": ("表示言語を保存しました。アプリの再起動後に反映されます。", "Interface language saved. Restart the application to apply it."),
    "界面语言无效": ("表示言語が無効です", "Invalid interface language"),
    "中文 → 日文": ("中国語 → 日本語", "Chinese → Japanese"),
    "日文 → 中文": ("日本語 → 中国語", "Japanese → Chinese"),
    "中→日": ("中→日", "ZH→JA"),
    "日→中": ("日→中", "JA→ZH"),
    "低（low）": ("低（low）", "Low (low)"),
    "中（medium）": ("中（medium）", "Medium (medium)"),
    "高（high）": ("高（high）", "High (high)"),
    "很高（xhigh）": ("非常に高い（xhigh）", "Very high (xhigh)"),
    "最高（max）": ("最大（max）", "Maximum (max)"),
    "综合候选": ("総合評価候補", "Overall candidate"),
    "速度候选": ("速度比較候補", "Speed candidate"),
    "质量对照": ("品質比較", "Quality comparison"),
    "未加载": ("未読み込み", "Not loaded"),
    "Codex CLI · 模型待确认": ("Codex CLI · モデル未確認", "Codex CLI · Model unconfirmed"),
    " · ChatGPT 订阅": (" · ChatGPT サブスクリプション", " · ChatGPT subscription"),
    "云端{0}": ("クラウド{0}", "Cloud{0}"),
    "本地识别模型": ("ローカル音声認識モデル", "Local recognition model"),
    "唯一识别模型；结合 Silero VAD 分句，支持自动识别或固定通道语言。": ("音声認識はこのモデルのみを使用します。Silero VAD で分句し、自動言語認識または入力言語の固定に対応します。", "The sole recognition model, paired with Silero VAD segmentation. Supports automatic or fixed channel language."),
    "当前识别方案": ("現在の認識構成", "Current recognition setup"),
    "本机同音频对照显示较低识别延迟；无人工参考文本，真实识别准确率尚未评分。": ("本 PC の同一音声比較では認識遅延が低下しました。人による参照テキストがないため、実際の認識精度は未評価です。", "Same-audio local comparisons showed lower recognition latency. Recognition accuracy remains unscored without a human reference transcript."),
    "CPU运行；识别结果交由所选翻译源处理": ("CPU で動作し、認識結果を選択した翻訳エンジンに渡します", "Runs on CPU; recognition output is passed to the selected translation source"),
    "SenseVoice 使用的静音分句依赖。": ("SenseVoice で使用する無音による分句モデル。", "Silence-based segmentation dependency for SenseVoice."),
    "当前仅支持 SenseVoice 识别模型。": ("現在、音声認識は SenseVoice のみ対応しています。", "Only SenseVoice recognition models are currently supported."),
    "本地翻译模型": ("ローカル翻訳モデル", "Local translation model"),
    "原有识别模型，保留作低延迟备用；支持自动识别或固定通道语言。": ("従来の認識モデルを低遅延の予備として保持しています。自動言語認識・入力言語の固定に対応します。", "Original recognition model retained as a low-latency fallback. Supports automatic or fixed channel language."),
    "低延迟备用": ("低遅延の予備", "Low-latency fallback"),
    "低延迟备用；本机一次工具对照中，55.6秒日语音频识别耗时1.38秒。有限样本无人工真值，人名需核对。": ("低遅延の予備。本 PC の1回のツール比較では55.6秒の日本語音声の認識に1.38秒かかりました。限定サンプルで人による正解確認はなく、人名は確認が必要です。", "Low-latency fallback. In one local tool comparison, recognizing 55.6s of Japanese audio took 1.38s. Samples are limited, without human ground truth; verify names."),
    "CPU运行；追求更快出字幕时可选": ("CPU で動作。字幕の速度を優先する場合の候補", "Runs on CPU; an option for faster subtitles"),
    "zengshuishui/Wasser1462第三方ONNX转换，由sherpa官方文档推荐；不是Qwen官方ONNX发布。许可Apache-2.0，支持自动识别及固定中文/日语/英语。": ("sherpa の公式文書で紹介された zengshuishui/Wasser1462 の第三者 ONNX 変換です。Qwen 公式の ONNX 配布ではありません。Apache-2.0。自動言語認識・中国語／日本語／英語の固定に対応。", "Third-party ONNX conversion by zengshuishui/Wasser1462, referenced by official sherpa documentation. Not an official Qwen ONNX release. Apache-2.0; supports automatic detection and fixed Chinese/Japanese/English."),
    "CPU运行；双路同时连续讲话可能排队": ("CPU で動作。両入力で連続して話すと処理待ちが生じる場合があります", "Runs on CPU; continuous speech on both sources may queue"),
    "sherpa维护者导出的Whisper多语种模型，许可MIT；本版已修复输出丢字，模型仍可能误识别。": ("sherpa メンテナーが変換した多言語 Whisper モデル、MIT ライセンス。本版では出力の文字欠落を修正していますが、モデルの誤認識は起こり得ます。", "Multilingual Whisper model exported by a sherpa maintainer, MIT license. Output character loss was fixed in this version, but recognition errors remain possible."),
    "CPU运行；双路同时连续讲话可能增加内存和等待": ("CPU で動作。両入力の連続発話はメモリ使用量と待ち時間を増やす場合があります", "Runs on CPU; continuous speech on both sources may increase memory use and waiting"),
    "CPU运行；双路同时连续讲话可能增加内存和等待；不适合实时连续字幕": ("CPU で動作。両入力の連続発話はメモリ使用量と待ち時間を増やす場合があります。リアルタイムの連続字幕には不向き", "Runs on CPU; simultaneous continuous speech may increase memory use and waiting. Unsuitable for continuous real-time subtitles"),
    "待完整评测": ("全体評価待ち", "Full evaluation pending"),
    "待评测": ("評価待ち", "Evaluation pending"),
    "手动离线对照": ("手動オフライン比較", "Manual offline comparison"),
    "本机有限日语样本，无人工真值；推荐指数不是准确率，人名仍可能误识别。": ("本 PC の限られた日本語サンプルで、人による正解確認はありません。推奨度は精度ではなく、人名の誤認識は起こり得ます。", "Limited local Japanese samples without human ground truth. Recommendation is not accuracy; names may be misrecognized."),
    "准确优先选择：本机8段日语工具对照，55.6秒音频识别耗时35.9秒；中文官方测试音频也已运行通过。有限样本无人工真值，人名仍可能误识别，推荐指数不是准确率。": ("精度優先の候補。本 PC の8区間の日本語ツール比較で55.6秒の音声認識に35.9秒かかりました。公式中国語テスト音声も動作確認済み。限定サンプルで人による正解確認はなく、人名の誤認識はあり得ます。推奨度は精度を意味しません。", "Accuracy-priority candidate: eight local Japanese test segments, 55.6s audio recognized in 35.9s. The official Chinese test audio also ran successfully. Limited samples without human ground truth; names may be misrecognized. Recommendation is not accuracy."),
    "速度备用：本机8段日语工具对照，55.6秒音频识别耗时17.7秒；中文待实测。有限样本无人工真值，人名仍可能误识别，推荐指数不是准确率。": ("速度重視の予備。本 PC の8区間の日本語比較で55.6秒の音声認識に17.7秒かかりました。中国語は実測待ち。限定サンプルで人による正解確認はなく、人名の誤認識はあり得ます。推奨度は精度を意味しません。", "Speed fallback: eight local Japanese test segments, 55.6s audio recognized in 17.7s. Chinese testing is pending. Limited samples without human ground truth; names may be misrecognized. Recommendation is not accuracy."),
    "手动离线对照：本机8段日语工具对照，55.6秒音频识别耗时215.9秒。有限样本无人工真值，人名仍可能误识别，推荐指数不是准确率。": ("手動オフライン比較。本 PC の8区間の日本語比較で55.6秒の音声認識に215.9秒かかりました。限定サンプルで人による正解確認はなく、人名の誤認識はあり得ます。推奨度は精度を意味しません。", "Manual offline comparison: eight local Japanese test segments, 55.6s audio recognized in 215.9s. Limited samples without human ground truth; names may be misrecognized. Recommendation is not accuracy."),
    "本机8段日语工具对照，55.6秒音频识别耗时28.0秒；仍有人名误识别，待完整评测。有限样本无人工真值，推荐指数不是准确率。": ("本 PC の8区間の日本語比較で55.6秒の音声認識に28.0秒かかりました。人名の誤認識があり、全体評価は未完了。限定サンプルで人による正解確認はなく、推奨度は精度を意味しません。", "Eight local Japanese test segments, 55.6s audio recognized in 28.0s. Names were still misrecognized; full evaluation is pending. Limited samples without human ground truth. Recommendation is not accuracy."),
    "所有识别模型共用的静音分句依赖。": ("すべての認識モデルで共有する無音による分句用モデル。", "Shared silence-based segmentation dependency for all recognition models."),
    "识别在 CPU 运行": ("音声認識は CPU で動作", "Recognition runs on CPU"),
    "必需：中日语音识别、词表与静音分句。请保留官方模型许可。": ("必須：中国語・日本語の音声認識、語彙表、無音による分句。モデルの公式ライセンスを保持してください。", "Required: Chinese/Japanese recognition, vocabulary and silence segmentation. Retain the official model license."),
    "本地区分讲话、唱歌和音乐；开启讲话筛选时需要。许可Apache-2.0。": ("発話・歌唱・音楽をローカルで分類します。発話フィルターに必要です。Apache-2.0。", "Locally classifies speech, singing and music; required when speech filtering is enabled. Apache-2.0."),
    "声音筛选依赖": ("音声フィルター用", "Audio filter dependency"),
    "CPU运行；同时重叠的讲话与歌声不能保证完全分离。": ("CPU で動作。重なる発話と歌声の完全分離は保証できません。", "Runs on CPU; complete separation of overlapping speech and singing is not guaranteed."),
    "专用翻译备用；体积较小。 权重许可：Apache-2.0，请保留官方 LICENSE。": ("翻訳専用の予備。比較的小さいモデルです。重みのライセンスは Apache-2.0、公式 LICENSE を保持してください。", "Specialized translation fallback with a smaller size. Weights license: Apache-2.0; retain the official LICENSE."),
    "本机质量与实时使用首选。 权重许可：Apache-2.0，请保留官方 LICENSE。": ("本 PC での品質とリアルタイム用途の第一候補。重みのライセンスは Apache-2.0、公式 LICENSE を保持してください。", "Local first choice for quality and real-time use. Weights license: Apache-2.0; retain the official LICENSE."),
    "手动质量对照；同组评测未稳定优于7B，非默认选择。 权重许可：Apache-2.0，请保留官方 LICENSE。": ("手動品質比較用。同じ評価で7Bを一貫して上回らず、既定ではありません。重みのライセンスは Apache-2.0、公式 LICENSE を保持してください。", "Manual quality comparison; did not consistently outperform 7B on the same evaluation and is not the default. Weights license: Apache-2.0; retain the official LICENSE."),
    "通用模型，作为翻译对照候选。 权重许可：Apache-2.0，请保留官方 LICENSE。": ("汎用モデル、翻訳比較の候補。重みのライセンスは Apache-2.0、公式 LICENSE を保持してください。", "General-purpose model used as a translation comparison candidate. Weights license: Apache-2.0; retain the official LICENSE."),
    "CPU 或 CUDA": ("CPU または CUDA", "CPU or CUDA"),
    "CPU 或 CUDA；速度需实测": ("CPU または CUDA。速度は実測が必要", "CPU or CUDA; speed requires testing"),
    "8 GB 显存需控制上下文，必要时 CPU + GPU 混合": ("8 GB の GPU メモリでは文脈長を抑え、必要に応じて CPU + GPU を併用", "With 8 GB VRAM, limit context and use CPU + GPU if needed"),
    "8 GB 显存需 CPU + GPU 混合；32 GB 本机试跑系统内存余量曾仅36.6 MiB": ("8 GB の GPU メモリでは CPU + GPU 併用が必要。本 PC の32 GB環境で試行時に空きメモリが36.6 MiBまで減少", "8 GB VRAM requires CPU + GPU. In a local 32 GB test, available system memory fell to 36.6 MiB"),
    "本机42条固定文本的备用方案；仍有增译和义变。 推荐指数不是准确率或通用排行榜。": ("本 PC の固定42文評価での予備。追加訳や意味変化が残ります。推奨度は精度や一般ランキングではありません。", "Fallback based on 42 fixed local texts; added content and meaning shifts remain. Recommendation is not accuracy or a general ranking."),
    "本机42条固定文本的当前综合首选；30B两轮未稳定更好且系统内存紧张。 推荐指数不是准确率或通用排行榜。": ("本 PC の固定42文評価での現在の総合第一候補。30Bは2回で安定して改善せず、メモリも逼迫しました。推奨度は精度や一般ランキングではありません。", "Current overall choice on 42 fixed local texts. 30B was not consistently better over two runs and strained system memory. Recommendation is not accuracy or a general ranking."),
    "本机42条固定文本的对照候选；出现数字条件遗漏与日文残留。 推荐指数不是准确率或通用排行榜。": ("本 PC の固定42文評価での比較候補。数値条件の欠落と日本語の残留がありました。推奨度は精度や一般ランキングではありません。", "Comparison candidate on 42 fixed local texts; omitted numerical conditions and left Japanese text. Recommendation is not accuracy or a general ranking."),
    "两轮各42条本机文本评测未一致优于7B；翻译模型单句耗时中位数1.754/1.848秒，保留手动质量对照。 推荐指数不是准确率或通用排行榜。": ("各42文の2回のローカル評価では7Bを一貫して上回りませんでした。1文の翻訳時間中央値は1.754/1.848秒、手動品質比較用として保持します。推奨度は精度や一般ランキングではありません。", "Two local evaluations of 42 texts each did not consistently outperform 7B. Median translation time per sentence was 1.754/1.848s. Retained for manual quality comparison. Recommendation is not accuracy or a general ranking."),
    "设备未知": ("デバイス不明", "Unknown device"),
    "识别": ("音声認識", "Recognition"),
    "翻译": ("翻訳", "Translation"),
    "识别语言：自动多语": ("認識言語：多言語自動認識", "Recognition language: multilingual auto"),
    "识别语言：按通道方向": ("認識言語：音声入力の翻訳方向に従う", "Recognition language: channel direction"),
    "待确认": ("未確認", "Unconfirmed"),
    "标准": ("標準", "Standard"),
    "推理：{0} · 速度：{1}": ("推論：{0} · 速度：{1}", "Reasoning: {0} · Speed: {1}"),
    "讲话筛选：关闭": ("発話フィルター：オフ", "Speech filter: off"),
    "讲话筛选：已启用": ("発話フィルター：オン", "Speech filter: enabled"),
    "自动多语识别开启时，箭头右侧决定译文语言；左侧作为通道主要语言参考。": (
        "多言語自動認識では、矢印の右側が翻訳先の言語、左側が音声入力の主な言語の目安です。",
        "With multilingual recognition enabled, the right side determines the target language; the left side indicates the main source language."),
    "首次设置": ("初回設定", "Initial setup"),
    "字幕设置": ("字幕設定", "Subtitle settings"),
    "选择声音来源和翻译方向": ("音声入力と翻訳方向を選択", "Choose audio sources and translation directions"),
    "保存设置后才会开始采集声音。两路可以分别开启，也可以分别选择翻译方向。": (
        "設定の保存後に音声の取得を開始します。各入力のオン・オフと翻訳方向を個別に設定できます。",
        "Audio capture starts after saving. Enable each source and select its translation direction separately."),
    "两路可以分别开启，也可以分别选择翻译方向。保存后应用新设置。": (
        "各入力のオン・オフと翻訳方向を個別に設定できます。保存後に設定を適用します。",
        "Enable each source and choose its translation direction separately. Save to apply the settings."),
    "翻译源": ("翻訳エンジン", "Translation source"),
    "本地模型（离线）": ("ローカルモデル（オフライン）", "Local model (offline)"),
    "Codex CLI（联网，使用 ChatGPT 订阅）": ("Codex CLI（オンライン・ChatGPT サブスクリプション）", "Codex CLI (online, ChatGPT subscription)"),
    "Codex 模型": ("Codex モデル", "Codex model"),
    "从当前账户的模型列表选择；主页显示实际生效的模型。": (
        "現在のアカウントで利用可能なモデルから選択します。メイン画面には実際に使用中のモデルを表示します。",
        "Choose from the current account's model list. The main window shows the model actually in use."),
    "账户默认（模型目录推荐）": ("アカウント既定（モデル一覧の推奨）", "Account default (model catalog recommendation)"),
    "{0} · 暂未验证": ("{0} · 未検証", "{0} · Unverified"),
    "刷新模型": ("モデル一覧を更新", "Refresh models"),
    "请刷新账户模型列表，再选择可用模型。": ("アカウントのモデル一覧を更新してから選択してください。", "Refresh the account model list, then select an available model."),
    "推理强度": ("推論の強度", "Reasoning effort"),
    "账户默认": ("アカウント既定", "Account default"),
    "选项依据当前账户返回的模型能力。低推理可作为速度对照，翻译准确率需要实测。": (
        "選択肢はアカウントから返されたモデルの機能に基づきます。低い推論設定は速度比較に使えますが、翻訳精度は実測が必要です。",
        "Options reflect the model capabilities returned by the account. Low reasoning can be compared for speed; translation accuracy requires testing."),
    "速度": ("速度", "Speed"),
    "Fast（额度消耗更快）": ("Fast（利用枠を多く消費）", "Fast (higher usage consumption)"),
    "Fast 需要账户和模型支持，订阅额度消耗通常为标准模式的 2.5 倍；不支持时会报告错误。": (
        "Fast はアカウントとモデルの対応が必要です。通常、標準の 2.5 倍の利用枠を消費します。未対応の場合はエラーを表示します。",
        "Fast requires account and model support and typically uses 2.5 times the standard subscription allowance. Unsupported requests report an error."),
    "声音先在本机识别，再将原文与最近上下文发送给 Codex 翻译。需要联网并登录 ChatGPT，持续翻译会消耗 Codex 订阅额度。Fast 需要账户及模型支持，通常消耗约 2.5 倍订阅额度。": (
        "音声をローカルで認識し、原文と直近の文脈を Codex に送信して翻訳します。インターネット接続と ChatGPT ログインが必要で、Codex の利用枠を消費します。Fast は対応が必要で、通常約 2.5 倍の利用枠を消費します。",
        "Audio is recognized locally; source text and recent context are sent to Codex for translation. Internet access and ChatGPT sign-in are required, and translation consumes Codex usage. Fast requires support and typically consumes about 2.5 times the allowance."),
    "麦克风": ("マイク", "Microphone"),
    "播放声音": ("再生音声", "Playback audio"),
    "电脑声音": ("PC の音声", "Computer audio"),
    "来自本地话筒": ("ローカルのマイクから取得", "From the local microphone"),
    "来自电脑正在播放的声音": ("PC で再生中の音声から取得", "From audio currently playing on the computer"),
    "自动多语识别（日语、中文、英语；译文语言按方向设置）": ("多言語自動認識（日本語・中国語・英語、翻訳先は方向設定に従う）", "Multilingual recognition (Japanese, Chinese, English; target follows direction)"),
    "识别不再强制单一语言，允许主播和游戏切换语言。Codex CLI 和本地 GGUF 翻译支持此模式；旧 M2M100 翻译保持固定语言并提示。不能分离同时重叠的人声。": (
        "認識言語を固定せず、配信やゲームでの言語切替に対応します。Codex CLI とローカル GGUF 翻訳が対応します。旧 M2M100 は固定言語を使用し、通知します。同時に重なる人声は分離できません。",
        "Recognition is not forced to one language, allowing language changes in streams and games. Codex CLI and local GGUF translation support this mode; older M2M100 keeps a fixed language and shows a notice. Overlapping voices cannot be separated."),
    "只翻译讲话（过滤音乐/唱歌片段）": ("発話のみ翻訳（音楽・歌唱を除外）", "Translate speech only (filter music/singing)"),
    "跳过高置信度的音乐、唱歌片段；短句和不确定片段优先保留。无法分离同时重叠的讲话与歌声。": (
        "音楽・歌唱と高い確度で判定した区間を除外します。短い発話や不確かな区間は優先して残します。重なる発話と歌声は分離できません。",
        "Skip segments confidently identified as music or singing. Keep short or uncertain segments. Overlapping speech and singing cannot be separated."),
    "建议使用耳机，避免电脑声音再次进入麦克风。": ("PC の音声がマイクに入らないよう、ヘッドホンの使用をお勧めします。", "Use headphones to prevent computer audio from entering the microphone again."),
    "保存并开始": ("保存して開始", "Save and start"),
    "保存设置": ("設定を保存", "Save settings"),
    "取消": ("キャンセル", "Cancel"),
    "退出": ("終了", "Exit"),
    "关闭": ("閉じる", "Close"),
    "正在读取已登录账户的可用模型…": ("ログイン中のアカウントのモデル一覧を取得中…", "Loading available models for the signed-in account…"),
    "读取模型失败：{0}。已保存的选择仍保留。": ("モデル一覧の取得に失敗：{0}。保存済みの選択は保持しています。", "Failed to load models: {0}. The saved selection is retained."),
    "已读取 {0} 个账户模型。综合、速度、质量标记是评测候选，实际表现需对比。": (
        "{0} 個のモデルを取得しました。総合・速度・品質の表示は評価候補を示し、実際の性能は比較が必要です。",
        "Loaded {0} account models. Overall, speed and quality labels indicate evaluation candidates; actual performance needs comparison."),
    "账户未返回可用模型，请确认 Codex 登录后重试；可保留账户默认。": (
        "利用可能なモデルが返されませんでした。Codex のログインを確認して再試行してください。アカウント既定は保持できます。",
        "No available models were returned. Check Codex sign-in and retry; you can keep the account default."),
    "输入音量": ("入力レベル", "Input level"),
    "准备中": ("準備中", "Preparing"),
    "等待声音…": ("音声を待機中…", "Waiting for audio…"),
    "方向未设置": ("翻訳方向未設定", "Direction not set"),
    "已关闭": ("オフ", "Disabled"),
    "此路声音已关闭": ("この音声入力はオフです", "This audio source is disabled"),
    "翻译中…": ("翻訳中…", "Translating…"),
    "等待后半句…": ("続きの発話を待機中…", "Waiting for the rest of the sentence…"),
    "识别复核尚未确认此段原文，请核对。": ("この認識原文は再確認されていません。内容を確認してください。", "Recognition review has not confirmed this source text. Please check it."),
    "（未返回译文）": ("（翻訳が返されませんでした）", "(No translation returned)"),
    "已暂停，此片段已取消": ("一時停止中、この区間はキャンセル済み", "Paused; this segment was cancelled"),
    "翻译姬": ("翻译姬", "翻译姬"),
    "拖动窗口标题栏可以移动位置": ("タイトルバーをドラッグして移動できます", "Drag the title bar to move the window"),
    "暂停": ("一時停止", "Pause"),
    "暂停或继续两路声音处理": ("両方の音声処理を一時停止・再開", "Pause or resume processing both audio sources"),
    "更改设置": ("設定を変更", "Settings"),
    "模型管理": ("モデル管理", "Models"),
    "置顶": ("最前面に表示", "Always on top"),
    "译文字号": ("翻訳の文字サイズ", "Font size"),
    "透明度": ("透明度", "Opacity"),
    "调整整个字幕窗口的透明度": ("字幕ウィンドウ全体の透明度を調整", "Adjust the opacity of the subtitle window"),
    "专用模型": ("専用モデル", "Specialized model"),
    "查看专用模型的本地数据、模型对照和训练准备；当前暂不开放使用": (
        "専用モデルのローカルデータ・比較・学習準備を表示します。現時点では使用できません。",
        "View local data, model comparisons and training preparation for the specialized model. Use is not enabled yet."),
    "专用模型：独立数据与训练准备 · 暂未开放使用": ("専用モデル：分離したデータと学習準備 · 使用未開放", "Specialized model: separate data and training preparation · Not enabled"),
    "模型：未加载": ("モデル：未読み込み", "Model: not loaded"),
    "待机中\n通过“更改设置”开启声音来源。": ("待機中\n「設定を変更」で音声入力を有効にしてください。", "Idle\nEnable audio sources in Change settings."),
    "请先保存首次设置。": ("初回設定を保存してください。", "Save the initial settings first."),
    "继续": ("再開", "Resume"),
    "正在准备识别和翻译…": ("音声認識と翻訳を準備中…", "Preparing recognition and translation…"),
    "正在听": ("音声を取得中", "Listening"),
    "正在听 · 翻译排队 {0} 句 · 最早等待 {1:.0f} 秒": ("音声を取得中 · 翻訳待ち {0} 文 · 最長待ち時間 {1:.0f} 秒", "Listening · {0} sentences queued · Oldest wait {1:.0f}s"),
    "已暂停": ("一時停止中", "Paused"),
    "已停止": ("停止済み", "Stopped"),
    "处理遇到问题": ("処理で問題が発生しました", "Processing problem"),
    "模型加载中…": ("モデルを読み込み中…", "Loading models…"),
    "模型切换中…": ("モデルを切替中…", "Switching models…"),
    "模型加载失败": ("モデルの読み込みに失敗", "Model loading failed"),
    "待机：两路采集均已关闭": ("待機中：両方の音声入力はオフです", "Idle: both audio sources are disabled"),
    "正在准备本地识别和 Codex 翻译…": ("ローカル音声認識と Codex 翻訳を準備中…", "Preparing local recognition and Codex translation…"),
    "正在加载本地模型…": ("ローカルモデルを読み込み中…", "Loading local models…"),
    "正在加载所选模型…": ("選択したモデルを読み込み中…", "Loading selected models…"),
    "正在听 · 本地识别 / Codex 订阅翻译": ("音声を取得中 · ローカル認識 / Codex サブスクリプション翻訳", "Listening · Local recognition / Codex subscription translation"),
    "正在听 · 完全本地": ("音声を取得中 · 完全ローカル", "Listening · Fully local"),
    "设置已保存 · 当前已暂停": ("設定を保存済み · 一時停止中", "Settings saved · Currently paused"),
    "无法开始采集": ("音声取得を開始できません", "Cannot start capture"),
    "模型管理 · 字幕已暂停": ("モデル管理 · 字幕を一時停止中", "Model management · Subtitles paused"),
    "第二路 Codex 翻译暂不可用，继续使用当前模型单路翻译。": ("2つ目の Codex 翻訳が利用できません。現在のモデルで1つずつ翻訳します。", "The second Codex translation worker is unavailable. Continuing with one worker using the current model."),
    "当前翻译模型要求固定源语言，识别沿用通道方向；选择 Codex 或支持混合语言的本地模型可启用自动多语。": (
        "現在の翻訳モデルは固定の入力言語が必要です。認識は音声入力の翻訳方向に従います。Codex または混合言語対応のローカルモデルを選ぶと多言語自動認識を有効にできます。",
        "The current translation model requires a fixed source language, so recognition follows channel direction. Choose Codex or a local model supporting mixed languages to enable multilingual recognition."),
    "字幕处理跟不上，已暂停接收新片段；当前片段未接收。已接收片段继续处理，点击暂停后再继续可恢复。": (
        "字幕処理が追いつかないため、新しい区間の受付を停止しました。今回の区間は未受付です。受付済みの区間は処理を続けます。一時停止して再開すると復帰できます。",
        "Subtitle processing cannot keep up, so new segments are paused and this segment was not accepted. Accepted segments continue processing. Pause and resume to recover."),
    "处理积压 · 已暂停采集": ("処理待ちが蓄積 · 音声取得を停止中", "Processing backlog · Capture paused"),
    "一个片段的讲话筛选或语音识别失败，已跳过；后续片段继续处理。": ("1区間の発話フィルターまたは音声認識に失敗したためスキップしました。後続の区間は処理を続けます。", "Speech filtering or recognition failed for a segment. It was skipped; later segments continue processing."),
    "识别控制标记混入原文，已跳过此片段；后续字幕继续处理。": ("原文に認識制御マーカーがあるため、この区間をスキップしました。後続の字幕は処理を続けます。", "Recognition control markers appeared in source text. This segment was skipped; later subtitles continue processing."),
    "翻译排队 {0} 句，最早已等待 {1} 秒；语音识别继续运行。": ("翻訳待ち {0} 文、最長待ち時間 {1} 秒。音声認識は継続します。", "{0} sentences queued for translation; oldest wait {1}s. Recognition continues."),
    "翻译缓存读写失败，已暂停采集；已接收的原文仍显示。请检查磁盘空间后暂停再继续；重新开始会取消当前待处理任务。": (
        "翻訳キャッシュの読み書きに失敗し、音声取得を停止しました。受付済みの原文は表示します。空き容量を確認し、一時停止して再開してください。再開すると現在の待機中処理はキャンセルします。",
        "Translation cache access failed, so capture is paused. Accepted source text remains displayed. Check disk space, then pause and resume; restarting cancels current pending jobs."),
    "翻译缓存不可用 · 已暂停采集": ("翻訳キャッシュが利用不可 · 音声取得を停止中", "Translation cache unavailable · Capture paused"),
    "译文包含识别控制标记，已阻止显示；后续字幕继续处理。": ("翻訳に認識制御マーカーがあるため、表示を阻止しました。後続の字幕は処理を続けます。", "Translation contains recognition control markers; display was blocked. Later subtitles continue processing."),
    "识别控制标记混入原文或前文，已跳过此片段；后续字幕继续处理。": ("原文または文脈に認識制御マーカーがあるため、この区間をスキップしました。後続の字幕は処理を続けます。", "Recognition control markers appeared in source text or context. This segment was skipped; later subtitles continue processing."),
    "批次译文格式未通过校验，正在逐句重译；原文顺序保持不变。": ("一括翻訳の形式検証に失敗しました。原文の順番を保ち、1文ずつ再翻訳します。", "Batch translation format validation failed. Retranslating sentence by sentence in the original order."),
    "Codex 翻译暂不可用，请检查联网、ChatGPT 登录或订阅额度；可在更改设置切回本地模型。": (
        "Codex 翻訳は現在利用できません。接続・ChatGPT ログイン・利用枠を確認してください。「設定を変更」でローカルモデルに戻せます。",
        "Codex translation is temporarily unavailable. Check internet access, ChatGPT sign-in or subscription allowance. Switch back to a local model in Change settings."),
    "专用本地模型 · 数据与训练准备": ("専用ローカルモデル · データと学習準備", "Specialized local model · Data and training preparation"),
    "专用模型尚未开放使用": ("専用モデルはまだ使用できません", "Specialized model use is not enabled yet"),
    "使用专用模型（未开放）": ("専用モデルを使用（未開放）", "Use specialized model (not enabled)"),
    "需要完成独立评测并另行开放；现有模型和训练文件不会自动启用此按钮": (
        "独立した評価後に使用を開放する必要があります。既存のモデルや学習ファイルでは自動的に有効になりません。",
        "Independent evaluation and separate activation are required. Existing models and training files do not automatically enable this button."),
    "推荐基底：Hy-MT2 1.8B；先对照本地模型，再准备 QLoRA 微调。训练环境和原始训练权重需要单独准备，已有 GGUF 用于翻译对照。本页不会切换当前正在使用的翻译模型。": (
        "推奨ベース：Hy-MT2 1.8B。まずローカルモデルを比較し、次に QLoRA 学習を準備します。学習環境と元の重みは別途必要です。既存の GGUF は翻訳比較に使用します。この画面では使用中の翻訳モデルを切り替えません。",
        "Recommended base: Hy-MT2 1.8B. Compare local models first, then prepare QLoRA fine-tuning. Training infrastructure and original weights must be prepared separately; existing GGUF files are for translation comparisons. This page does not switch the active translation model."),
    "积累专用模型数据（本地单独存放）": ("専用モデルのデータを収集（ローカルで分離保存）", "Collect specialized model data (stored separately locally)"),
    "数据目录：{0}": ("データフォルダー：{0}", "Data directory: {0}"),
    "对照本地模型（自动暂停字幕）": ("ローカルモデルを比較（字幕を自動で一時停止）", "Compare local models (automatically pauses subtitles)"),
    "按顺序加载已下载的本地翻译模型，避免与语音识别争抢显存；结束后恢复原字幕状态": (
        "音声認識と GPU メモリを競合しないよう、ダウンロード済み翻訳モデルを順番に読み込みます。終了後に元の字幕状態に戻ります。",
        "Load downloaded local translation models one at a time to avoid competing with recognition for GPU memory. Restore the previous subtitle state afterward."),
    "取消对照": ("比較をキャンセル", "Cancel comparison"),
    "导出训练集": ("学習データをエクスポート", "Export training set"),
    "准备训练方案": ("学習計画を準備", "Prepare training plan"),
    "打开数据目录": ("データフォルダーを開く", "Open data directory"),
    "Codex 纠错仅作隔离对照，不自动进入训练集。本地候选需要核对完整原话、译文和素材使用权；断句、听错和未完成字幕不能当作正确答案。": (
        "Codex の修正は分離した比較用で、学習データに自動追加しません。ローカル候補は原文全体・翻訳・素材の利用権を確認する必要があります。途中の発話、誤認識、未完成字幕は正解にできません。",
        "Codex corrections are kept separately for comparison and do not automatically enter the training set. Verify the complete source, local translation and material usage rights. Incomplete sentences, recognition errors and unfinished subtitles are not correct answers."),
    "选择样本查看原文和候选译文": ("サンプルを選択して原文と翻訳候補を表示", "Select a sample to view source text and translation candidates"),
    "识别原文（需独立听音核对）": ("認識原文（音声を独立して確認）", "Recognized source text (verify independently by listening)"),
    "合并语句可能对应多个录音片段，请逐段听音核对完整原文": (
        "結合された文には複数の録音が対応する場合があります。各録音を聞き、原文全体を確認してください。",
        "A merged sentence may correspond to multiple recordings. Listen to each segment to verify the complete source text."),
    "播放样本音频": ("サンプル音声を再生", "Play sample audio"),
    "已独立听音，确认原文完整且准确": ("音声を独立して確認し、原文の完全性と正確性を確認済み", "Independently listened and verified the complete, accurate source"),
    "已确认原始素材及本地译文允许用于训练": ("原素材とローカル翻訳の学習への利用権を確認済み", "Confirmed rights to use the source material and local translation for training"),
    "确认本地译文准确": ("ローカル翻訳の正確性を確認", "Confirm local translation accuracy"),
    "读取专用模型数据失败：{0}": ("専用モデルのデータを読み込めません：{0}", "Failed to read specialized model data: {0}"),
    "样本 {0} · 音频 {1} · 本地候选 {2} · Codex 隔离候选 {3} · 已核对 {4} · 可训练 {5}": (
        "サンプル {0} · 音声 {1} · ローカル候補 {2} · 分離した Codex 候補 {3} · 確認済み {4} · 学習可能 {5}",
        "Samples {0} · Audio {1} · Local candidates {2} · Isolated Codex candidates {3} · Reviewed {4} · Trainable {5}"),
    "对照已取消；原字幕状态将自动恢复。": ("比較をキャンセルしました。元の字幕状態に自動で戻ります。", "Comparison cancelled. The previous subtitle state will be restored automatically."),
    "数据积累": ("データ収集中", "Collecting data"),
    "{0} · 暂未开放使用": ("{0} · 使用未開放", "{0} · Use not enabled"),
    "电脑": ("PC", "Computer"),
    "话筒": ("マイク", "Microphone"),
    "未完成": ("未完了", "Incomplete"),
    "识别存疑": ("認識に疑義あり", "Uncertain recognition"),
    "翻译失败": ("翻訳失敗", "Translation failed"),
    "样本 {0} · 版本 {1}": ("サンプル {0} · リビジョン {1}", "Sample {0} · Revision {1}"),
    "Codex（隔离，仅对照）": ("Codex（分離・比較のみ）", "Codex (isolated, comparison only)"),
    "本地候选": ("ローカル候補", "Local candidate"),
    "未知来源（隔离）": ("出所不明（分離）", "Unknown origin (isolated)"),
    " · 已核对": (" · 確認済み", " · Reviewed"),
    "模型未知": ("モデル不明", "Unknown model"),
    "尚无样本；启用数据积累后会记录实际识别结果": ("サンプルはまだありません。データ収集を有効にすると実際の認識結果を記録します。", "No samples yet. Enable data collection to record actual recognition results."),
    "音频片段 {0} / {1}": ("音声区間 {0} / {1}", "Audio segment {0} / {1}"),
    "尚无候选译文；可启动本地模型对照。": ("翻訳候補はまだありません。ローカルモデル比較を開始できます。", "No translation candidates yet. Start a local model comparison."),
    "Codex 纠错保留原始来源，隔离存放，仅供对照；当前不能核准或导出用于训练。": (
        "Codex 修正は元の出所を保持して分離保存し、比較のみに使います。現時点では学習用の承認・エクスポートはできません。",
        "Codex corrections preserve their original provenance and are stored separately for comparison only. They cannot currently be approved or exported for training."),
    "候选来源未验证，当前不能核准用于训练。": ("候補の出所が未確認のため、学習用に承認できません。", "Candidate provenance is unverified; it cannot be approved for training."),
    "已完成本地候选人工核对。": ("ローカル候補の手動確認が完了しました。", "Manual review of the local candidate is complete."),
    "符合当前训练导出条件。": ("現在の学習用エクスポート条件を満たしています。", "Meets the current training export requirements."),
    "当前未选为训练译文，可重新核对后确认该候选。": ("学習用翻訳には未選択です。再確認してこの候補を承認できます。", "Not selected as the training translation; review again to confirm this candidate."),
    "本地模型结果尚未核对；请独立听音并确认本地译文准确。": ("ローカルモデルの結果は未確認です。音声を独立して確認し、翻訳の正確性を確認してください。", "The local result has not been reviewed. Listen independently and confirm the local translation is accurate."),
    "已开启本地数据积累。": ("ローカルデータ収集を有効にしました。", "Local data collection enabled."),
    "已暂停数据积累；字幕声音采集保持当前设置。": ("データ収集を停止しました。字幕用の音声取得設定は維持します。", "Data collection paused. Subtitle audio capture keeps its current settings."),
    "更改数据积累设置失败：{0}": ("データ収集設定を変更できません：{0}", "Failed to change data collection settings: {0}"),
    "本地模型对照已开始，完成后恢复原字幕状态。": ("ローカルモデル比較を開始しました。完了後に元の字幕状態に戻ります。", "Local model comparison started. The previous subtitle state will be restored when it finishes."),
    "本地模型对照未开始，请检查状态。": ("ローカルモデル比較を開始できませんでした。状態を確認してください。", "Local comparison did not start. Check the status."),
    "无法开始本地对照：{0}": ("ローカル比較を開始できません：{0}", "Cannot start local comparison: {0}"),
    "正在取消对照，等待本地模型释放资源后恢复字幕。": ("比較をキャンセル中です。モデルのリソース解放後に字幕を再開します。", "Cancelling comparison. Subtitles resume after the local model releases its resources."),
    "取消对照未完成：{0}": ("比較のキャンセルが未完了です：{0}", "Comparison cancellation is incomplete: {0}"),
    "已记录本地译文核对结果；训练导出仍会检查完整性和来源。": (
        "ローカル翻訳の確認結果を保存しました。学習用エクスポート時も完全性と出所を確認します。",
        "Local translation review recorded. Training export will still check completeness and provenance."),
    "核对结果未保存：{0}": ("確認結果を保存できません：{0}", "Review result was not saved: {0}"),
    "训练集已导出：": ("学習データをエクスポートしました：", "Training set exported: "),
    "训练集尚未导出：{0}": ("学習データは未エクスポートです：{0}", "Training set has not been exported: {0}"),
    "训练准备结果：": ("学習準備の結果：", "Training preparation result: "),
    "训练准备尚未完成：{0}": ("学習準備が未完了です：{0}", "Training preparation is incomplete: {0}"),
    "训练 {0} 条": ("学習 {0} 件", "Training rows: {0}"),
    "验证 {0} 条": ("検証 {0} 件", "Validation rows: {0}"),
    "训练已执行": ("学習実行済み", "Training executed"),
    "训练尚未执行": ("学習未実行", "Training not executed"),
    "已有训练产物": ("学習済み成果あり", "Trained artifacts exist"),
    "尚未训练": ("未学習", "Not trained yet"),
    "目录：{0}": ("フォルダー：{0}", "Directory: {0}"),
    "待准备：": ("準備が必要：", "Preparation needed: "),
    "；另有 {0} 项，详见训练说明": ("；ほか {0} 項目、学習手順を参照", "; {0} more items; see training instructions"),
    "数据目录尚未创建，开始积累数据后再打开。": ("データフォルダーがまだありません。収集を開始してから開いてください。", "The data directory does not exist yet. Start collecting data before opening it."),
    "离线模型管理": ("オフラインモデル管理", "Offline model manager"),
    "识别模型": ("音声認識モデル", "Recognition models"),
    "翻译模型": ("翻訳モデル", "Translation models"),
    "当前模型：{0}": ("使用中のモデル：{0}", "Current model: {0}"),
    "当前已加载 · {0}": ("現在読み込み済み · {0}", "Currently loaded · {0}"),
    "权重：{0} · 推荐指数：{1}": ("重み：{0} · 推奨度：{1}", "Weights: {0} · Recommendation: {1}"),
    "推荐依据：{0}": ("推奨理由：{0}", "Recommendation rationale: {0}"),
    "尚未校验": ("未検証", "Not verified yet"),
    "重新校验本地文件": ("ローカルファイルを再検証", "Reverify local files"),
    "打开时检查本地文件，未变化则复用校验记录，不自动联网下载。": (
        "開くとローカルファイルを確認します。変更がなければ検証記録を再利用します。自動ダウンロードは行いません。",
        "Checks local files on opening and reuses verification records when unchanged. Does not automatically download."),
    "模型独立存放在下方目录。联网下载一次，校验后可断网使用。下载不会自动切换当前模型；选中识别、翻译模型和基础分句都就绪后，点击“使用选中模型”启用。未改动的文件复用校验记录；切换保留其他已下载模型。": (
        "モデルは下記の別フォルダーに保存します。一度ダウンロードして検証するとオフラインで使用できます。ダウンロードでは使用中のモデルは変わりません。選択した認識・翻訳モデルと基本分句が揃ったら「選択したモデルを使用」を押してください。変更のないファイルは検証記録を再利用し、他のモデルも保持します。",
        "Models are stored separately below. Download once and verify to use offline. Downloading does not switch the active model. When the selected recognition, translation and basic segmentation models are ready, click Use selected models. Unchanged files reuse verification records; other downloaded models are retained."),
    "当前翻译源：Codex CLI（联网，使用 ChatGPT 订阅额度）。本页仅切换本地识别模型；识别、基础分句和已开启的声音筛选依赖就绪后即可启用，无需下载本地翻译权重。本地翻译模型仍可查看和下载，已保存的翻译模型设置会保留。": (
        "現在の翻訳エンジンは Codex CLI（オンライン・ChatGPT 利用枠を消費）です。この画面ではローカル認識モデルのみ切り替えます。認識・基本分句・有効な音声フィルターが揃えば使用でき、ローカル翻訳の重みは不要です。ローカル翻訳モデルの表示・ダウンロードは可能で、保存済み設定も保持します。",
        "Current translation source: Codex CLI (online, using ChatGPT subscription allowance). This page switches only local recognition models. Recognition, basic segmentation and enabled audio filter dependencies must be ready; local translation weights are not required. Local translation models remain available to view and download, and saved translation selections are retained."),
    "当前翻译源：Codex CLI（联网，使用 ChatGPT 订阅额度）。识别固定使用 SenseVoice + Silero VAD；识别、基础分句和已开启的声音筛选依赖就绪后即可启用，无需下载本地翻译权重。本地翻译模型仍可查看和下载，已保存的翻译模型设置会保留。": (
        "現在の翻訳エンジンは Codex CLI（オンライン・ChatGPT 利用枠を消費）です。認識は SenseVoice + Silero VAD に固定されています。認識・基本分句・有効な音声フィルターが揃えば使用でき、ローカル翻訳の重みは不要です。ローカル翻訳モデルの表示・ダウンロードは可能で、保存済み設定も保持します。",
        "Current translation source: Codex CLI (online, using ChatGPT subscription allowance). Recognition uses SenseVoice + Silero VAD. Recognition, basic segmentation and enabled audio filter dependencies must be ready; local translation weights are not required. Local translation models remain available to view and download, and saved translation selections are retained."),
    "点击顶部“导入本地模型”，选择文件、ZIP 或文件夹；离线识别兼容模板后填写名称。识别仅支持 SenseVoice，翻译模型导入保留。导入不联网、不覆盖已有模型，也不会自动启用。未知 ONNX / CT2 暂不支持。": (
        "上部の「ローカルモデルをインポート」からファイル・ZIP・フォルダーを選び、互換テンプレート検出後に名前を入力してください。音声認識は SenseVoice のみ対応し、翻訳モデルのインポートは引き続き利用できます。インポートはオフラインで行い、既存モデルの上書きや自動有効化は行いません。不明な ONNX / CT2 には未対応です。",
        "Use Import local model to choose a file, ZIP or folder, then name it after compatible template detection. Recognition supports SenseVoice only; translation model imports remain available. Imports are offline, do not overwrite existing models and do not activate automatically. Unknown ONNX / CT2 models are unsupported."),
    "Codex 翻译模式需要选择本地识别模型": ("Codex 翻訳にはローカル認識モデルの選択が必要です", "Codex translation requires selecting a local recognition model"),
    "使用选中模型": ("選択したモデルを使用", "Use selected models"),
    "使用选中识别模型": ("選択した認識モデルを使用", "Use selected recognition model"),
    "选择此翻译模型": ("この翻訳モデルを選択", "Select this translation model"),
    "选择此识别模型": ("この認識モデルを選択", "Select this recognition model"),
    "下载 / 续传": ("ダウンロード・再開", "Download / resume"),
    "重新下载": ("再ダウンロード", "Download again"),
    "下载直链": ("直接ダウンロードリンク", "Direct download link"),
    "官方资料": ("公式資料", "Official information"),
    "离线手动放置模型": ("モデルをオフラインで手動配置", "Place models manually offline"),
    "离线放置说明": ("オフライン配置手順", "Offline placement instructions"),
    "可在其他电脑从“下载直链”下载，再复制到下列完整路径：": (
        "別の PC で「直接ダウンロードリンク」から取得し、次のパスにコピーできます：",
        "Download on another computer using Direct download link, then copy to these full paths:"),
    "保留模型官方许可。放置后点击“重新校验本地文件”；文件名必须完全一致。": (
        "モデルの公式ライセンスを保持してください。配置後に「ローカルファイルを再検証」を押してください。ファイル名は完全一致が必要です。",
        "Retain the model's official license. After placement, click Reverify local files; filenames must match exactly."),
    "正在校验本地文件…": ("ローカルファイルを検証中…", "Verifying local files…"),
    "校验中…": ("検証中…", "Verifying…"),
    "校验失败：": ("検証失敗：", "Verification failed: "),
    "校验失败：{0}": ("検証失敗：{0}", "Verification failed: {0}"),
    "已下载 · 校验通过（文件未变时复用记录）": ("ダウンロード済み · 検証済み（変更がなければ記録を再利用）", "Downloaded · Verified (records reused when files are unchanged)"),
    "未下载": ("未ダウンロード", "Not downloaded"),
    "下载未完成，可续传": ("ダウンロード未完了・再開可能", "Download incomplete; can resume"),
    "已下载 · SHA-256 校验通过": ("ダウンロード済み · SHA-256 検証済み", "Downloaded · SHA-256 verified"),
    "校验未完成；可重新校验": ("検証未完了・再検証可能", "Verification incomplete; reverify to continue"),
    "正在下载；可取消并保留进度。": ("ダウンロード中です。キャンセルして進行状況を保持できます。", "Downloading. You can cancel and retain progress."),
    "下载中… {0}%": ("ダウンロード中… {0}%", "Downloading… {0}%"),
    "取消当前任务": ("現在の処理をキャンセル", "Cancel current task"),
    "正在取消… 当前网络读取结束后退出；.part 保留，可续传。": (
        "キャンセル中… 現在のネットワーク読み取り終了後に停止します。.part を保持して再開できます。",
        "Cancelling… Stops after the current network read. The .part file is retained for resuming."),
    "已取消，可续传": ("キャンセル済み・再開可能", "Cancelled; can resume"),
    "已取消；未完成的 .part 文件保留，可继续下载": ("キャンセルしました。未完了の .part ファイルを保持し、ダウンロードを再開できます", "Cancelled; unfinished .part files are retained and downloading can resume"),
    "任务失败，可手动重试": ("処理に失敗・手動で再試行可能", "Task failed; retry manually"),
    "操作完成。下载不会自动启用；请选择模型后点击“使用选中模型”。": (
        "処理が完了しました。ダウンロードでは自動使用しません。モデルを選択し「選択したモデルを使用」を押してください。",
        "Operation complete. Downloading does not activate models automatically. Select models and click Use selected models."),
    "操作完成。当前翻译源仍为 Codex CLI；可点击“使用选中识别模型”切换本地识别。": (
        "処理が完了しました。翻訳エンジンは Codex CLI のままです。「選択した認識モデルを使用」でローカル認識を切り替えられます。",
        "Operation complete. Translation still uses Codex CLI. Click Use selected recognition model to switch local recognition."),
    "导入本地模型": ("ローカルモデルをインポート", "Import local model"),
    "选择 ZIP / 模型文件": ("ZIP・モデルファイルを選択", "Choose ZIP / model file"),
    "选择模型文件夹": ("モデルフォルダーを選択", "Choose model folder"),
    "导入 {0}": ("{0} をインポート", "Import {0}"),
    "ZIP / 模型文件 (*.zip *.gguf *.onnx *.txt *.csv);;所有文件 (*)": (
        "ZIP・モデルファイル (*.zip *.gguf *.onnx *.txt *.csv);;すべてのファイル (*)",
        "ZIP / model files (*.zip *.gguf *.onnx *.txt *.csv);;All files (*)"),
    "请选择完整模型文件夹、ZIP 包；单文件仅支持单文件模型。": (
        "モデル一式のフォルダーまたは ZIP を選択してください。単独ファイルは単一ファイルモデルのみ対応します。",
        "Choose a complete model folder or ZIP archive. Single files are supported only for single-file models."),
    "正在导入本地模型… 原模型包保留，导入不会自动切换。": (
        "ローカルモデルをインポート中… 元のパッケージを保持し、自動でモデルを切り替えません。",
        "Importing local model… The original package is retained; importing does not switch models automatically."),
    "复制中… {0}%": ("コピー中… {0}%", "Copying… {0}%"),
    "保存导入模型… {0}%": ("インポートしたモデルを保存中… {0}%", "Saving imported model… {0}%"),
    "大小：{0} B": ("サイズ：{0} B", "Size: {0} B"),
    "本地导入 · 目录固定 SHA-256 校验通过": ("ローカルからインポート · カタログ指定の SHA-256 検証済み", "Local import · Catalog SHA-256 verified"),
    "导入完成。原模型包保留；请选择模型后点击“使用选中模型”，不会自动切换。": (
        "インポート完了。元のパッケージを保持しています。モデルを選択し「選択したモデルを使用」を押してください。自動では切り替えません。",
        "Import complete. The original package is retained. Select models and click Use selected models; switching is not automatic."),
    "正在取消导入… 原模型包与已有模型保留。": ("インポートをキャンセル中… 元のパッケージと既存のモデルを保持します。", "Cancelling import… The original package and existing models are retained."),
    "导入已取消，原模型包保留": ("インポートをキャンセルしました。元のパッケージを保持しています", "Import cancelled; original package retained"),
    "导入已取消；原模型包与已有模型保留，未完成暂存可稍后手动清理。": (
        "インポートをキャンセルしました。元のパッケージと既存モデルは保持します。未完了の一時ファイルは後で手動削除できます。",
        "Import cancelled. The original package and existing models are retained. Unfinished staging files can be cleaned manually later."),
    "模型导入失败；原包、已有模型和未完成暂存均保留。": (
        "モデルのインポートに失敗しました。元のパッケージ・既存モデル・未完了の一時ファイルを保持しています。",
        "Model import failed. The original package, existing models and unfinished staging files are retained."),
    "只能导入目录中已支持的识别、分句、声音筛选或翻译模型。": (
        "カタログ対応の音声認識・分句・音声フィルター・翻訳モデルのみインポートできます。",
        "Only catalog-supported recognition, segmentation, audio filter or translation models can be imported."),
    "未知模型架构不能导入；请使用界面支持的模型目录版本。": (
        "不明なモデル構造はインポートできません。画面のカタログで対応しているバージョンを使用してください。",
        "Unknown model architectures cannot be imported. Use a model version supported by the interface catalog."),
    "未选择要导入的本地模型包。": ("インポートするモデルパッケージを選択してください。", "No local model package was selected."),
    "选择的模型文件或目录不存在。": ("選択したモデルファイルまたはフォルダーがありません。", "The selected model file or directory does not exist."),
    "模型来源与存储目录不能相同或相互嵌套。": ("モデルのコピー元と保存先は同一または包含関係にできません。", "The model source and storage directory cannot be identical or nested."),
    "模型包包含不安全路径。": ("モデルパッケージに安全でないパスがあります。", "The model package contains an unsafe path."),
    "模型包路径重复或存在大小写冲突。": ("モデルパッケージのパスが重複または大文字・小文字で競合しています。", "The model package has duplicate paths or case conflicts."),
    "模型包缺少必要文件或存在同名歧义；请选择对应模型的完整文件夹。": (
        "必須ファイルがないか同名ファイルが曖昧です。対象モデルの完全なフォルダーを選択してください。",
        "The package is missing required files or has ambiguous duplicate names. Choose the complete folder for the selected model."),
    "模型包文件数量超过安全限制。": ("モデルパッケージのファイル数が安全上限を超えています。", "The package file count exceeds the safety limit."),
    "模型包大小超过安全限制。": ("モデルパッケージのサイズが安全上限を超えています。", "The model package exceeds the size limit."),
    "模型包包含 symlink/junction，不能导入。": ("symlink/junction を含むモデルパッケージはインポートできません。", "Model packages containing symlink/junction cannot be imported."),
    "模型包包含 hardlink 或非常规文件，不能导入。": ("hardlink または通常でないファイルを含むパッケージはインポートできません。", "Packages containing hardlinks or non-regular files cannot be imported."),
    "模型文件含 hardlink，不能导入。": ("hardlink のモデルファイルはインポートできません。", "Model files containing hardlinks cannot be imported."),
    "模型路径含 symlink/junction，不能导入。": ("symlink/junction を含むモデルパスはインポートできません。", "Model paths containing symlink/junction cannot be imported."),
    "模型目标路径超出存储目录。": ("モデルの保存パスが保存フォルダーの外にあります。", "The model destination path is outside the storage directory."),
    "模型文件大小与选中目录版本不匹配。": ("モデルのサイズが選択したカタログバージョンと一致しません。", "The model file size does not match the selected catalog version."),
    "模型文件不符合目录固定大小或 SHA-256；不支持该版本或包已损坏。": (
        "モデルがカタログ指定のサイズまたは SHA-256 と一致しません。未対応バージョンまたは破損の可能性があります。",
        "Model files do not match the catalog size or SHA-256. This version is unsupported or the package is damaged."),
    "目标模型已存在且不匹配；已保留，请手动处理后重试。": (
        "一致しないモデルが保存先に存在します。保持したため、手動で対応してから再試行してください。",
        "A different model already exists at the destination. It was retained; resolve it manually before retrying."),
    "已有导入来源记录不匹配，已保留，请手动处理后重试。": (
        "既存のインポート元記録が一致しません。保持したため、手動で対応してから再試行してください。",
        "The existing import provenance record does not match. It was retained; resolve it manually before retrying."),
    "模型来源在复制期间发生变化，导入未完成。": ("コピー中にモデルのコピー元が変更されました。インポートは未完了です。", "The model source changed during copying; import did not complete."),
    "导入暂存在校验后发生变化，未完成发布。": ("検証後に一時ファイルが変更されました。保存先への反映は未完了です。", "Import staging changed after verification; installation did not complete."),
    "ZIP 包含不支持的链接扩展信息。": ("ZIP に未対応のリンク拡張情報があります。", "The ZIP contains unsupported link extension metadata."),
    "ZIP 包含加密、链接或非常规文件，不能导入。": ("暗号化・リンク・通常でないファイルを含む ZIP はインポートできません。", "ZIP archives with encrypted, linked or non-regular files cannot be imported."),
    "ZIP 扩展信息无效。": ("ZIP の拡張情報が無効です。", "Invalid ZIP extension metadata."),
    "ZIP 文件与目录路径冲突。": ("ZIP のファイルとフォルダーパスが競合しています。", "ZIP file and directory paths conflict."),
    "ZIP 大小或压缩比例超过安全限制。": ("ZIP のサイズまたは圧縮率が安全上限を超えています。", "The ZIP size or compression ratio exceeds the safety limit."),
    "ZIP 包格式无效。": ("ZIP パッケージの形式が無効です。", "Invalid ZIP package format."),
    "ZIP 目录信息超过安全限制。": ("ZIP のディレクトリ情報が安全上限を超えています。", "The ZIP directory metadata exceeds the safety limit."),
    "不支持多卷 ZIP 模型包。": ("分割された ZIP モデルパッケージには対応していません。", "Multi-volume ZIP model packages are not supported."),
    "ZIP 压缩比例限制必须为有限正数": ("ZIP 圧縮率の上限は有限の正数である必要があります", "The ZIP compression ratio limit must be a finite positive number"),
    "导入文件数量、大小和块大小必须为正整数": ("インポートのファイル数・サイズ・ブロックサイズは正の整数である必要があります", "Import file count, size and chunk size must be positive integers"),
    "已有模型包可点击各行“导入本地模型”。支持对应目录版本的 ZIP、文件夹或单文件；导入不联网、不覆盖已有冲突文件，也不会自动启用。": (
        "モデルパッケージがある場合、各行の「ローカルモデルをインポート」を押してください。対象カタログバージョンの ZIP・フォルダー・単独ファイルに対応します。インポートはオフラインで行い、競合する既存ファイルを上書きせず、自動使用もしません。",
        "If you already have a model package, click Import local model on its row. ZIP archives, folders or single files must match the catalog version. Import is offline, preserves conflicting existing files and does not activate the model automatically."),
    "导入完成。原模型包保留，当前翻译源仍为 Codex CLI；模型不会自动切换。": (
        "インポート完了。元のパッケージを保持し、翻訳は Codex CLI のままです。モデルは自動で切り替えません。",
        "Import complete. The original package is retained, translation still uses Codex CLI, and models are not switched automatically."),
    "界面预览": ("画面プレビュー", "Interface preview"),
    "无法保存": ("保存できません", "Cannot save"),
    "无法启动": ("起動できません", "Cannot start"),
    "需要重新设置": ("再設定が必要です", "Settings need to be configured again"),
    "设置未改变，请检查输入内容和设置目录权限。": ("設定は変更されていません。入力内容と設定フォルダーの権限を確認してください。", "Settings were not changed. Check the input and settings directory permissions."),
    "无法保存设置，尚未开始采集。": ("設定を保存できません。音声取得は開始していません。", "Cannot save settings. Audio capture has not started."),
    "专用模型不可用": ("専用モデルを利用できません", "Specialized model unavailable"),
    "请检查本地数据目录是否可写。字幕功能仍可使用。": ("ローカルデータフォルダーへの書き込み権限を確認してください。字幕機能は利用できます。", "Check that the local data directory is writable. Subtitles remain available."),
    "已有一个字幕程序正在启动，请稍后再试。": ("別の字幕アプリが起動中です。しばらくしてから再試行してください。", "Another subtitle application is starting. Try again shortly."),
    "专用模型数据目录不可用；字幕仍可运行。": ("専用モデルのデータフォルダーが使えません。字幕は動作できます。", "The specialized model data directory is unavailable; subtitles can still run."),
    "本地模型对照进行中；完成后自动恢复此前的字幕状态。": ("ローカルモデルを比較中です。完了後に元の字幕状態に戻ります。", "Local comparison is running. The previous subtitle state will be restored afterward."),
    "请先完成或取消专用模型对照，再更改设置。": ("専用モデル比較を完了またはキャンセルしてから設定を変更してください。", "Finish or cancel the specialized model comparison before changing settings."),
    "请先完成或取消专用模型对照，再切换模型。": ("専用モデル比較を完了またはキャンセルしてからモデルを切り替えてください。", "Finish or cancel the specialized model comparison before switching models."),
    "专用模型对照中断；可在专用模型窗口查看状态。": ("専用モデル比較が中断しました。専用モデル画面で状態を確認できます。", "Specialized model comparison was interrupted. Check its window for status."),
    "无法读取设置文件，请重新设置": ("設定ファイルを読み込めません。再設定してください", "Cannot read the settings file. Configure settings again"),
    "设置文件版本无效": ("設定ファイルのバージョンが無効です", "Invalid settings file version"),
    "讲话筛选开关无效": ("発話フィルター設定が無効です", "Invalid speech filter setting"),
    "自动多语识别开关无效": ("多言語自動認識設定が無効です", "Invalid multilingual recognition setting"),
    "正在连接默认音频设备…": ("既定の音声デバイスに接続中…", "Connecting to the default audio device…"),
    "默认设备已变化，正在重连已启用的音频通道…": ("既定のデバイスが変わりました。有効な音声入力を再接続中…", "Default device changed. Reconnecting enabled audio sources…"),
    "正在重建默认设备连接（已启用的通道会一起重连）…": ("既定のデバイスへ再接続中（有効な入力をまとめて再接続）…", "Rebuilding default device connections (enabled channels reconnect together)…"),
    "已连接：{0}": ("接続済み：{0}", "Connected: {0}"),
    "音频采集已停止": ("音声取得を停止しました", "Audio capture stopped"),
    "本路音频设备连接已中断": ("この入力の音声デバイスが切断されました", "This source's audio device was disconnected"),
    "本路采集正在重连，另一通道继续运行…": ("この入力を再接続中です。他の入力は動作を継続します…", "Reconnecting this source; the other channel continues…"),
    "本路设备状态异常：{0}": ("この入力のデバイス状態に異常があります：{0}", "Audio device problem for this source: {0}"),
    "本路采集失败：{0}": ("この入力の音声取得に失敗しました：{0}", "Audio capture failed for this source: {0}"),
    "音频设备上下文不可用：{0}": ("音声デバイスのコンテキストを利用できません：{0}", "Audio device context unavailable: {0}"),
    "无法启动音频采集：{0}": ("音声取得を開始できません：{0}", "Cannot start audio capture: {0}"),
    "本路音频处理未及时退出，请暂停后重启软件。": ("この入力の音声処理が終了していません。一時停止してアプリを再起動してください。", "Audio processing did not stop promptly. Pause and restart the application."),
    "采集开启 · 等待完整语句": ("収集中 · 完全な文を待機中", "Collection enabled · Waiting for complete sentences"),
    "已暂停积累训练数据": ("学習データ収集を一時停止中", "Training data collection paused"),
    "正在暂停字幕并释放模型 · 每批最多30句": ("字幕を停止してモデルを解放中 · 1回最大30文", "Pausing subtitles and releasing models · Up to 30 sentences per batch"),
    "正在对照 {0} · 已保存 {1} 条候选": ("{0} を比較中 · 候補 {1} 件を保存済み", "Comparing {0} · Saved {1} candidates"),
    "对照已取消 · 本次保存 {0} 条本地候选 · 尚未训练": ("比較キャンセル済み · ローカル候補 {0} 件を保存 · 未学習", "Comparison cancelled · Saved {0} local candidates · Not trained"),
    "对照完成 · 本次保存 {0} 条本地候选 · 尚未训练": ("比較完了 · ローカル候補 {0} 件を保存 · 未学習", "Comparison complete · Saved {0} local candidates · Not trained"),
    "对照停止 · 尚未训练": ("比較停止 · 未学習", "Comparison stopped · Not trained"),
}


_MESSAGES.update({'GGUF 元数据字符串超过安全限制。': ('GGUF メタデータの文字列が安全上限を超えています。', 'GGUF metadata string exceeds the safety limit.'),
 'GGUF 元数据或张量数量无效。': ('GGUF メタデータまたはテンソル数が無効です。', 'Invalid GGUF metadata or tensor count.'),
 'GGUF 元数据数组无效或超过安全限制。': ('GGUF メタデータ配列が無効、または安全上限を超えています。',
                          'GGUF metadata array is invalid or exceeds the safety limit.'),
 'GGUF 元数据类型未受支持。': ('GGUF メタデータ型に対応していません。', 'Unsupported GGUF metadata type.'),
 'GGUF 元数据编码无效。': ('GGUF メタデータの文字コードが無効です。', 'Invalid GGUF metadata encoding.'),
 'GGUF 元数据键无效或重复。': ('GGUF メタデータキーが無効、または重複しています。', 'Invalid or duplicate GGUF metadata key.'),
 'GGUF 头部过大或文件不完整。': ('GGUF ヘッダーが大きすぎるか、ファイルが不完全です。', 'GGUF header is too large or the file is incomplete.'),
 'GGUF 张量块或偏移无效。': ('GGUF テンソルブロックまたはオフセットが無効です。', 'Invalid GGUF tensor block or offset.'),
 'GGUF 张量对齐信息无效。': ('GGUF テンソルのアライメント情報が無効です。', 'Invalid GGUF tensor alignment.'),
 'GGUF 张量形状无效。': ('GGUF テンソルの形状が無効です。', 'Invalid GGUF tensor shape.'),
 'GGUF 张量描述无效或重复。': ('GGUF テンソル記述が無効、または重複しています。', 'Invalid or duplicate GGUF tensor descriptor.'),
 'GGUF 权重范围重叠或文件不完整。': ('GGUF の重み領域が重複しているか、ファイルが不完全です。',
                        'GGUF weight ranges overlap or the file is incomplete.'),
 'GGUF 缺少有效架构元数据。': ('GGUF に有効なアーキテクチャ情報がありません。', 'GGUF is missing valid architecture metadata.'),
 'GGUF 词表元数据不完整。': ('GGUF の語彙メタデータが不完全です。', 'Incomplete GGUF vocabulary metadata.'),
 'GGUF 量化格式尚未适配。': ('この GGUF 量子化形式にはまだ対応していません。', 'This GGUF quantization format is not supported yet.'),
 'SenseVoice 模型包缺少必要文件。': ('SenseVoice モデルパッケージに必要なファイルがありません。',
                           'Required files are missing from the SenseVoice model package.'),
 'ZIP / 模型文件 (*.zip *.gguf *.onnx);;所有文件 (*)': ('ZIP / モデルファイル (*.zip *.gguf *.onnx);;すべてのファイル (*)',
                                                'ZIP / model files (*.zip *.gguf *.onnx);;All files (*)'),
 '不支持分片 GGUF；请选择完整单文件模型。': ('分割 GGUF には対応していません。完全な単一ファイルを選択してください。',
                            'Split GGUF is unsupported; select a complete single-file model.'),
 '取消': ('キャンセル', 'Cancel'),
 '已识别预设模型版本；使用目录固定 SHA-256 校验，原模型包保留。': ('プリセットのモデルを検出しました。カタログの固定 SHA-256 で検証し、元のパッケージは保持します。',
                                         'Preset model detected; verify against the catalog SHA-256 and '
                                         'retain the original package.'),
 '未识别为已支持的模型。未知 ONNX / CT2 不受支持；请提供完整预设模型包或单个兼容 GGUF。': ('対応モデルを検出できませんでした。未知の ONNX / CT2 '
                                                         'には対応していません。完全なプリセットパッケージか互換 GGUF を指定してください。',
                                                         'No supported model detected. Unknown ONNX / CT2 is '
                                                         'unsupported; select a complete preset package or '
                                                         'compatible GGUF.'),
 '本地导入 · 未评测': ('ローカルインポート · 未評価', 'Local import · Not evaluated'),
 '本地导入 · 本地 SHA-256 一致；官方来源未经验证': ('ローカルインポート · ローカル SHA-256 一致、公式提供元は未確認',
                                   'Local import · Local SHA-256 verified; official source unverified'),
 '本地模型名称或 ID 已存在，请使用其他名称。': ('モデル名または ID が既に存在します。別の名前を使用してください。',
                             'Model name or ID already exists; use a different name.'),
 '本地模型注册数量已达到 256 个上限。': ('ローカルモデルの登録数が上限の 256 件に達しました。',
                          'The local model registry has reached its limit of 256 models.'),
 '本地模型注册表 ID 或用途无效。': ('ローカルモデル登録の ID または用途が無効です。', 'Invalid local model registry ID or purpose.'),
 '本地模型注册表文件信息无效。': ('ローカルモデル登録のファイル情報が無効です。', 'Invalid local model registry file information.'),
 '本地模型注册表文件列表无效。': ('ローカルモデル登録のファイル一覧が無効です。', 'Invalid local model registry file list.'),
 '本地模型注册表文件路径重复。': ('ローカルモデル登録のファイルパスが重複しています。', 'Duplicate local model registry file path.'),
 '本地模型注册表格式无效；已有记录保留。': ('ローカルモデル登録の形式が無効です。既存の記録は保持しました。',
                         'Invalid local model registry format; existing records retained.'),
 '本地模型注册表路径超出独立模型目录。': ('ローカルモデル登録のパスが独立モデルフォルダーの外にあります。',
                        'Local model registry path is outside its independent model directory.'),
 '本地模型注册表过大或无效；已有记录保留。': ('ローカルモデル登録が大きすぎるか無効です。既存の記録は保持しました。',
                          'Local model registry is too large or invalid; existing records retained.'),
 '本地模型的预设来源未受支持。': ('このローカルモデルのプリセット提供元には対応していません。', 'Unsupported preset source for this local model.'),
 '本地模型记录与预设固定 SHA-256 不一致。': ('ローカルモデル記録がプリセットの固定 SHA-256 と一致しません。',
                              'Local model record does not match the preset SHA-256.'),
 '校验并导入': ('検証してインポート', 'Verify and import'),
 '检查中… {0}%': ('確認中… {0}%', 'Checking… {0}%'),
 '检查完成，请确认用途、模板与显示名称。': ('確認が完了しました。用途、テンプレート、表示名を確認してください。',
                         'Inspection complete; confirm the purpose, template and display name.'),
 '模型包无法读取或已损坏；原包保留。': ('モデルパッケージを読み込めないか破損しています。元のパッケージは保持しました。',
                       'Model package is unreadable or damaged; original package retained.'),
 '模型显示名称必须为 1 到 80 个字符，不能包含控制字符。': ('モデル表示名は 1～80 文字で、制御文字は使用できません。',
                                    'Model display name must contain 1–80 characters with no control '
                                    'characters.'),
 '模型显示名称必须是 1 到 80 个字符': ('モデル表示名は 1～80 文字で入力してください', 'Model display name must contain 1–80 characters'),
 '模型显示名称（1–80 字符）': ('モデル表示名（1～80 文字）', 'Model display name (1–80 characters)'),
 '模型来源在检查期间发生变化，请重新选择。': ('確認中にモデルのソースが変わりました。もう一度選択してください。',
                          'Model source changed during inspection; select it again.'),
 '模型来源：{0}': ('モデルのソース：{0}', 'Model source: {0}'),
 '模型模板文件路径与用途不一致。': ('モデルテンプレートのファイルパスと用途が一致しません。', 'Model template file path does not match its purpose.'),
 '正在离线检查模型包与兼容模板…': ('モデルパッケージと互換テンプレートをオフラインで確認中…',
                     'Inspecting model package and compatible templates offline…'),
 '点击顶部“导入本地模型”，选择文件、ZIP 或文件夹；离线识别兼容模板后填写名称。导入不联网、不覆盖已有模型，也不会自动启用。未知 ONNX / CT2 暂不支持。': ('上部の「ローカルモデルをインポート」からファイル、ZIP、フォルダーを選び、互換テンプレートの確認後に名前を入力してください。オフラインでインポートし、既存モデルの上書きや自動有効化は行いません。未知の '
                                                                                        'ONNX / CT2 は未対応です。',
                                                                                        'Use “Import local '
                                                                                        'model” at the top '
                                                                                        'to select a file, '
                                                                                        'ZIP or folder, then '
                                                                                        'name it after '
                                                                                        'offline template '
                                                                                        'detection. '
                                                                                        'Importing preserves '
                                                                                        'existing models and '
                                                                                        'requires manual '
                                                                                        'activation. Unknown '
                                                                                        'ONNX / CT2 is '
                                                                                        'unsupported.'),
 '独立命名的本地模型；导入后需要手动选择启用。': ('独立した名前のローカルモデル。インポート後に手動で選択して有効化してください。',
                            'Independently named local model; select it manually to activate after import.'),
 '自定义 GGUF：结构与本地 SHA-256 校验；官方来源、运行兼容性与翻译效果未经验证。': ('カスタム GGUF：構造とローカル SHA-256 を検証。公式提供元、実行互換性、翻訳精度は未確認です。',
                                                    'Custom GGUF: structure and local SHA-256 verified; '
                                                    'official source, runtime compatibility and translation '
                                                    'quality unverified.'),
 '自定义本地 GGUF；仅校验结构与本地摘要，官方来源未经验证。': ('カスタムのローカル GGUF。構造とローカルハッシュのみ検証し、公式提供元は未確認です。',
                                     'Custom local GGUF; structure and local hash verified, official source '
                                     'unverified.'),
 '自定义本地模型没有下载地址，请使用本地导入。': ('カスタムローカルモデルにはダウンロード先がありません。ローカルインポートを使用してください。',
                            'Custom local models have no download link; use local import.'),
 '自定义模型元数据与注册记录不一致。': ('カスタムモデルのメタデータが登録記録と一致しません。',
                       'Custom model metadata does not match its registry record.'),
 '该 GGUF 架构尚未适配；仅支持 HY / Qwen 翻译模板。': ('この GGUF アーキテクチャは未対応です。HY / Qwen 翻訳テンプレートのみ対応しています。',
                                       'Unsupported GGUF architecture; only HY / Qwen translation templates '
                                       'are supported.'),
 '该模型用途尚未支持独立导入。': ('この用途のモデルはまだ独立インポートに対応していません。',
                    'Independent import is not supported for this model purpose yet.'),
 '该自定义模型格式尚未适配。': ('このカスタムモデル形式にはまだ対応していません。', 'This custom model format is not supported yet.'),
 '请选择完整的 GGUF v2/v3 模型文件。': ('完全な GGUF v2/v3 モデルファイルを選択してください。', 'Select a complete GGUF v2/v3 model file.'),
 '请选择用途与兼容模板，并填写显示名称。导入后需要手动启用。': ('用途と互換テンプレートを選び、表示名を入力してください。インポート後に手動で有効化する必要があります。',
                                   'Select the purpose and compatible template, then enter a display name. '
                                   'Activate the model manually after import.')})


_MESSAGES.update({
    "含【】少于 11 字符时等待下一句合并；含【】和分隔空格最多 40 字符。实际账号加权上限可能低于 40，按当前允许长度拆分。": (
        "【】込みで 11 文字未満の場合は次の文を待って結合します。【】と区切りの空白を含めて最大40文字です。アカウントの実際の加重上限は40未満の場合があり、現在の許容長で分割します。",
        "If the message has fewer than 11 characters including 【】, wait to merge it with the next sentence. Allow up to 40 characters including brackets and separating spaces. The account's actual weighted limit may be lower than 40; split at its current allowed length."),
    "短句等待下一句合并": ("短い文を次の文と結合するまで待機", "Waiting to merge a short sentence with the next sentence"),
    "正文少于 8 字符时等待下一句；含【】和分隔空格最多 40 字符。实际账号加权上限可能低于 40，按当前允许长度拆分。": (
        "本文が8文字未満の場合は次の文を待ちます。【】と区切りの空白を含めて最大40文字です。アカウントの実際の加重上限は40未満の場合があり、現在の許容長で分割します。",
        "If the body has fewer than 8 characters, wait for the next sentence. Allow up to 40 characters including brackets and separating spaces. The account's actual weighted limit may be lower than 40; split at its current allowed length."),
})


_MESSAGES.update({
    "音源与语音处理": ("音声ソースと音声処理", "Audio source and speech processing"),
    "语音处理：{0}": ("音声処理：{0}", "Speech processing: {0}"),
    "语音处理": ("音声処理", "Speech processing"),
    "GTCRN 语音增强": ("GTCRN 音声強調", "GTCRN speech enhancement"),
    "UVR 伴奏分离": ("UVR 伴奏分離", "UVR accompaniment separation"),
    "原音回退": ("元の音声を使用", "Original audio fallback"),
    "指定来源只隔离其他软件的声音；主播、游戏对白和音乐仍属于同一份混音。": ("指定ソースは他のアプリの音声を除外します。配信者、ゲームの会話、音楽は同じミックス音声に含まれます。", "The selected source excludes other apps. Streamer speech, game dialogue, and music remain in the same audio mix."),
    "仅采集内置浏览器当前播放来源": ("内蔵ブラウザーの再生音声のみ取得", "Capture only the current built-in browser source"),
    "B站视频链接": ("Bilibili 動画 URL", "Bilibili video URL"),
    "粘贴 B站 HTTPS 普通视频地址": ("Bilibili の HTTPS 動画 URL を貼り付け", "Paste a Bilibili HTTPS video URL"),
    "打开视频": ("動画を開く", "Open video"),
    "普通视频也可生成字幕，来源不会被当作直播间。启动程序不会自动打开浏览器。": ("通常の動画でも字幕を生成できます。動画は配信ルームとして扱いません。アプリ起動時にブラウザーは自動で開きません。", "Regular videos can also generate subtitles and are not treated as live rooms. Starting the app does not open the browser automatically."),
    "语音处理模式": ("音声処理モード", "Speech processing mode"),
    "关闭语音处理": ("音声処理を無効化", "Speech processing off"),
    "GTCRN · 实时语音增强（推荐）": ("GTCRN · リアルタイム音声強調（推奨）", "GTCRN · Real-time speech enhancement (recommended)"),
    "UVR · 人声与伴奏分离（较慢）": ("UVR · 声と伴奏の分離（低速）", "UVR · Vocal and accompaniment separation (slower)"),
    "应用语音处理模式": ("音声処理モードを適用", "Apply speech processing mode"),
    "当前保存模式：{0}": ("保存済みモード：{0}", "Saved mode: {0}"),
    "下载处理模型": ("処理モデルをダウンロード", "Download processing model"),
    "校验本地处理模型": ("ローカル処理モデルを検証", "Verify local processing model"),
    "官方模型链接": ("公式モデルのリンク", "Official model link"),
    "模型单独下载到本机，不包含在程序包中。缺失或处理失败时回退原音，并显示状态。": ("モデルは別途ダウンロードし、配布パッケージには含めません。モデルがない場合や処理に失敗した場合は元の音声を使用し、状態を表示します。", "Models are downloaded separately and excluded from the app package. Missing models or processing failures fall back to original audio with a visible status."),
    "语音增强主要减少噪声；伴奏分离可能改善背景音乐，但仍会保留唱歌和游戏角色人声，不能只提取主播。": ("音声強調は主にノイズを減らします。伴奏分離は背景音楽を軽減できる場合がありますが、歌声やゲームの声も残り、配信者の声だけは抽出できません。", "Speech enhancement mainly reduces noise. Accompaniment separation may reduce background music, but retains singing and game voices and cannot isolate only the streamer."),
    "指定音源状态不可用": ("指定音声ソースの状態を取得できません", "Selected audio source status is unavailable"),
    "电脑声音采集": ("パソコンの再生音声を取得", "Capturing computer playback audio"),
    "指定视频音源已连接": ("指定した動画の音声に接続済み", "Selected video audio connected"),
    "指定直播音源已连接": ("指定した配信の音声に接続済み", "Selected live audio connected"),
    "等待指定播放来源": ("指定した再生ソースを待機中", "Waiting for the selected playback source"),
    "等待打开指定播放来源": ("指定した再生ソースが開くまで待機", "Waiting for the selected playback source to open"),
    "指定音源异常，请重新保存采集设置或刷新播放来源": ("指定ソースに問題があります。取得設定を保存し直すか、再生ソースを更新してください", "Selected source error; save capture settings again or refresh the playback source"),
    "采集设置已保存": ("取得設定を保存しました", "Capture settings saved"),
    "采集设置保存失败，已有设置保留": ("取得設定を保存できませんでした。既存の設定を保持します", "Capture settings could not be saved; existing settings retained"),
    "直播间格式无效或浏览器无法打开": ("配信ルームの形式が無効、またはブラウザーを開けません", "Invalid live room or browser unavailable"),
    "视频地址无效或浏览器无法打开": ("動画 URL が無効、またはブラウザーを開けません", "Invalid video URL or browser unavailable"),
    "处理已关闭": ("処理は無効です", "Processing is off"),
    "本地处理模型校验通过": ("ローカル処理モデルの検証に成功", "Local processing model verified"),
    "本地处理模型缺失或校验未通过": ("ローカル処理モデルがないか、検証に失敗しました", "Local processing model is missing or failed verification"),
    "点击校验以检查本地处理模型": ("検証ボタンでローカル処理モデルを確認してください", "Select Verify to check the local processing model"),
    "语音处理模式保存失败，已有设置保留": ("音声処理モードを保存できませんでした。既存の設定を保持します", "Speech processing mode could not be saved; existing settings retained"),
    "语音处理模式已保存，字幕重新加载后生效": ("音声処理モードを保存しました。字幕の再読み込み後に有効になります", "Speech processing mode saved; takes effect when subtitles reload"),
    "正在下载处理模型…": ("処理モデルをダウンロード中…", "Downloading processing model…"),
    "正在校验本地处理模型…": ("ローカル処理モデルを検証中…", "Verifying local processing model…"),
    "操作已取消": ("操作をキャンセルしました", "Action cancelled"),
    "处理模型操作失败，请检查本地文件或网络": ("処理モデルの操作に失敗しました。ローカルファイルまたはネットワークを確認してください", "Processing model operation failed; check local files or the network"),
    "正在取消处理模型操作…": ("処理モデルの操作をキャンセル中…", "Cancelling processing model operation…"),
    "指定音源设置读取失败，请重新保存采集设置": ("指定ソースの設定を読み込めません。取得設定を保存し直してください", "Selected source settings could not be read; save capture settings again"),
    "刷新或重新打开播放来源后重试": ("再生ソースを更新または開き直して再試行してください", "Refresh or reopen the playback source and retry"),
    "指定音源启动失败": ("指定した音声ソースを開始できません", "Selected audio source could not start"),
    "指定音源采集失败": ("指定した音声ソースの取得に失敗", "Selected audio source capture failed"),
})


@lru_cache(maxsize=1)
def _rendered_patterns():
    """Match complete, registered backend status templates, never substrings."""
    patterns = []
    for source in sorted(_MESSAGES, key=len, reverse=True):
        fields, chunks = [], []
        for literal, field, spec, conversion in Formatter().parse(source):
            chunks.append(re.escape(literal))
            if field is not None:
                if spec or conversion or not (field.isdecimal() or field.isidentifier()):
                    break
                if field in fields:
                    chunks.append(f"(?P=f{fields.index(field)})")
                else:
                    chunks.append(f"(?P<f{len(fields)}>.*?)")
                    fields.append(field)
        else:
            if fields:
                patterns.append((re.compile("".join(chunks), re.DOTALL), source, fields))
    return patterns


def set_language(language: str) -> None:
    global _language
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("Unsupported UI language")
    _language = language


def get_language() -> str:
    return _language


def recovery_action_text(action: object) -> str:
    """Render known recovery actions without exposing arbitrary backend details."""
    source = _RECOVERY_ACTION_LABELS.get(action) if isinstance(action, str) else None
    if source is None and isinstance(action, str) and action in _RECOVERY_ACTION_LABELS.values():
        source = action
    return tr(source or "自动处理当前失败消息")


def recovery_category_text(category: object) -> str:
    source = _RECOVERY_CATEGORY_LABELS.get(category) if isinstance(category, str) else None
    return tr(source or "原因未确定")


def tr(text: str, *values, **named) -> str:
    """Translate a whole UI message, preserving model IDs, paths and other values."""
    translated = _MESSAGES.get(text)
    # Backends may have formatted their status before it reaches the UI. Only
    # whole registered messages are recognized; all captured values stay intact.
    if translated is None and not values and not named and _language != "zh-CN" and len(text) <= 4096:
        for pattern, source, fields in _rendered_patterns():
            match = pattern.fullmatch(text)
            if match is not None:
                positional, keywords = {}, {}
                for index, field in enumerate(fields):
                    value = match.group(f"f{index}")
                    if field.isdecimal():
                        positional[int(field)] = value
                    else:
                        keywords[field] = value
                args = tuple(positional.get(index, "") for index in range(max(positional, default=-1) + 1))
                return tr(source, *args, **keywords)
    result = translated[0 if _language == "ja" else 1] if translated and _language != "zh-CN" else text
    return result.format(*values, **named) if values or named else result
