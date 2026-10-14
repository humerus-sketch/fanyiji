"""UI-facing account checks, official sign-in and user-triggered test translation."""
from __future__ import annotations

from .domestic_cli_catalog import CLI_URLS, validate_provider


class DomesticCliService:
    def probe(self, provider: str, *, cancellation_event=None) -> dict:
        from .domestic_cli_runtime import probe_cli
        return probe_cli(validate_provider(provider), cancellation_event=cancellation_event)

    def official_url(self, provider: str) -> str:
        return CLI_URLS[validate_provider(provider)]

    def login(self, provider: str, *, cancellation_event=None) -> None:
        from .domestic_cli_runtime import login_cli
        login_cli(validate_provider(provider), cancellation_event=cancellation_event)

    def check_translation(self, provider: str, model: str = "", text: str = "今日はゲームの音量を少し下げます。",
                          *, cancellation_event=None) -> dict:
        import time
        from .domestic_cli_engine import DomesticCliTranslationEngine
        engine = DomesticCliTranslationEngine(provider=validate_provider(provider), model=model)
        started = time.monotonic()
        try:
            engine.preload(cancellation_event=cancellation_event)
            translated = engine.translate(text, "ja", "zh", cancellation_event=cancellation_event)
            info = engine.backend_info
            return {"text": text, "translation": translated, "seconds": time.monotonic()-started,
                    "model": info.get("model_id", "") or "未确认", "provider": provider}
        finally:
            engine.close()
