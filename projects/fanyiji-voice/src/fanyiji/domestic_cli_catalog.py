"""Official account CLI choices; credentials stay with the vendor's client."""
from __future__ import annotations

CLI_PROVIDERS = ("minimax", "minimax_code", "kimi", "codebuddy", "trae", "qwen")
CLI_NAMES = {"minimax": "MiniMax 文本 CLI", "minimax_code": "MiniMax Code CLI",
             "kimi": "Kimi Code CLI", "codebuddy": "腾讯 CodeBuddy CLI",
             "trae": "TraeCode CLI", "qwen": "Qwen Code CLI"}
CLI_URLS = {"minimax": "https://github.com/MiniMax-AI/cli",
            "minimax_code": "https://github.com/MiniMax-AI/minimax-code",
            "kimi": "https://www.kimi.com/code/docs/kimi-code-cli/guides/getting-started",
            "codebuddy": "https://www.codebuddy.cn/docs/cli/quickstart",
            "trae": "https://docs.trae.cn/cli_get-started-with-trae-code-cli-2",
            "qwen": "https://qwenlm.github.io/qwen-code-docs/zh/users/configuration/auth/"}
BLOCKED_PROVIDERS = {"qwen": "Qwen Code 官方云端 OAuth 已停止，当前需要 API Key；本模式仅接受账号登录。"}


def validate_provider(provider: str) -> str:
    if not isinstance(provider, str) or provider not in CLI_PROVIDERS:
        raise ValueError("国内 CLI 服务无效")
    return provider
