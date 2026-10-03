import pytest

import models


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Isolated cache, zero spend, and a priced copy of the mock model."""
    monkeypatch.setattr(models, "CACHE_DIR", tmp_path)
    monkeypatch.setitem(models.SPENT, "usd", 0.0)
    cfg = dict(models.config())
    cfg["models"] = dict(cfg["models"], **{"mock-priced": {"provider": "mock", "model_id": "mock", "price_in": 1000.0,
                                                           "price_out": 1000.0, "supports_temperature": True}})
    monkeypatch.setattr(models, "_CFG", cfg)
    return tmp_path


def test_second_call_is_served_from_cache(sandbox):
    first = models.complete("mock", "system", "- write something", temperature=0)
    second = models.complete("mock", "system", "- write something", temperature=0)
    assert not first["cached"] and second["cached"] and first["text"] == second["text"]
    assert len(list(sandbox.rglob("*.json"))) == 1


def test_repeat_tag_makes_a_separate_cache_entry(sandbox):
    models.complete("mock", "system", "user", cache_tag="r1")
    models.complete("mock", "system", "user", cache_tag="r2")
    assert len(list(sandbox.rglob("*.json"))) == 2


def test_budget_stop(sandbox, monkeypatch):
    monkeypatch.setenv("MAX_BUDGET_USD", "0.01")
    with pytest.raises(models.BudgetExceeded):
        models.complete("mock-priced", "system", "a prompt long enough to cost money")  # goes over, but is cached
    with pytest.raises(models.BudgetExceeded):
        models.complete("mock-priced", "system", "a different prompt")  # refused before calling
    spent = models.SPENT["usd"]
    replay = models.complete("mock-priced", "system", "a prompt long enough to cost money")  # cache hit still works
    assert replay["cached"] and models.SPENT["usd"] == spent


def test_anthropic_call_sends_temperature_only_where_supported(sandbox, monkeypatch):
    """The SDK 1.x has no temperature argument, so it must travel in extra_body, and never for Opus 5."""
    import sys
    import types

    sent = []

    class FakeMessages:
        def create(self, **kwargs):
            sent.append(kwargs)
            block = types.SimpleNamespace(type="text", text="ok")
            usage = types.SimpleNamespace(input_tokens=10, output_tokens=2)
            return types.SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda: types.SimpleNamespace(messages=FakeMessages())
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    haiku = models.complete("claude-haiku-4-5", "s", "u", temperature=0)
    models.complete("claude-opus-5", "s", "u", temperature=0)
    assert "temperature" not in sent[0] and sent[0]["extra_body"] == {"temperature": 0}
    assert "temperature" not in sent[1] and "extra_body" not in sent[1]
    assert haiku["cost_usd"] == pytest.approx(10 * 1.0 / 1e6 + 2 * 5.0 / 1e6)


def test_cut_off_reply_is_not_reused_at_a_higher_limit(sandbox, monkeypatch):
    calls = []

    def fake_mock(spec, system, user, temperature, max_tokens, cache_tag):
        calls.append(max_tokens)
        return "partial", 10, max_tokens, "max_tokens" if max_tokens < 100 else "end_turn"

    monkeypatch.setattr(models, "call_mock", fake_mock)
    models.complete("mock", "s", "u", max_tokens=50)
    models.complete("mock", "s", "u", max_tokens=50)   # same limit: cached
    models.complete("mock", "s", "u", max_tokens=500)  # higher limit: asked again
    assert calls == [50, 500]
