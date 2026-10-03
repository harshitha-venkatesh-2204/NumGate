"""One entry point for every model: complete(model_name, system, user, temperature, max_tokens).

Every response is cached on disk under cache/<model>/<hash>.json, so reruns cost nothing.
A run stops with BudgetExceeded once billed spend passes MAX_BUDGET_USD (env) or config max_budget_usd.
"""
import hashlib
import json
import os
import threading
import time

import requests

import mock_model
from config import CACHE_DIR, load_config


class BudgetExceeded(Exception):
    pass


SPENT = {"usd": 0.0}  # billed spend in this process; cache hits add nothing
_LOCK = threading.Lock()  # guards SPENT when notes are generated in parallel
_CFG = {}  # loaded config, filled on first use (tests may replace it)


def config():
    if not _CFG:
        _CFG.update(load_config())
    return _CFG


def budget_limit():
    return float(os.environ.get("MAX_BUDGET_USD", config().get("max_budget_usd", 20)))


def cache_key(model_name, system, user, temperature, cache_tag=""):
    # cache_tag holds the repeat index, so repeat 2 is a fresh call rather than a replay of repeat 1
    blob = json.dumps([model_name, system, user, temperature, cache_tag], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_KEY_LOCKS = {}  # cache path -> lock, so two workers never pay for the same call


def key_lock(path):
    with _LOCK:
        return _KEY_LOCKS.setdefault(str(path), threading.Lock())


def complete(model_name, system, user, temperature=0, max_tokens=None, cache_tag=""):
    """Return {text, input_tokens, output_tokens, cost_usd, latency_s, cached, model, stop_reason}."""
    spec = config()["models"][model_name]  # provider, model_id, prices
    max_tokens = max_tokens or config().get("max_tokens", 4000)
    path = CACHE_DIR / model_name / f"{cache_key(model_name, system, user, temperature, cache_tag)}.json"
    with key_lock(path):  # a second worker asking for the same call waits, then reads the cache
        return complete_locked(spec, model_name, system, user, temperature, max_tokens, cache_tag, path)


def complete_locked(spec, model_name, system, user, temperature, max_tokens, cache_tag, path):
    if path.exists():
        resp = json.loads(path.read_text(encoding="utf-8"))
        # A reply cut off at a lower token limit than the one asked for now is not reused.
        cut_off = resp.get("stop_reason") in {"max_tokens", "length"} and max_tokens > resp.get("max_tokens", 0)
        if not cut_off:
            resp["cached"] = True
            return resp

    with _LOCK:
        if SPENT["usd"] >= budget_limit():
            raise BudgetExceeded(f"spent ${SPENT['usd']:.2f} of ${budget_limit():.2f}")
    start = time.time()
    call = {"mock": call_mock, "anthropic": call_anthropic, "openai": call_openai,
            "openrouter": call_openrouter}[spec["provider"]]
    text, tokens_in, tokens_out, stop_reason = call(spec, system, user, temperature, max_tokens, cache_tag)
    cost = tokens_in * spec["price_in"] / 1e6 + tokens_out * spec["price_out"] / 1e6
    resp = {"text": text, "input_tokens": tokens_in, "output_tokens": tokens_out, "cost_usd": cost,
            "latency_s": round(time.time() - start, 3), "model": spec["model_id"], "stop_reason": stop_reason,
            "max_tokens": max_tokens}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")  # unique per thread, renamed into place
    tmp.write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    resp["cached"] = False
    with _LOCK:
        SPENT["usd"] += cost
        over = SPENT["usd"] > budget_limit()
    if over:
        raise BudgetExceeded(f"spent ${SPENT['usd']:.2f} of ${budget_limit():.2f}")
    return resp


def call_mock(spec, system, user, temperature, max_tokens, cache_tag):
    text = mock_model.respond(system, user, cache_tag)
    return text, (len(system) + len(user)) // 4, len(text) // 4, "end_turn"  # rough 4 chars per token


def call_anthropic(spec, system, user, temperature, max_tokens, cache_tag):
    import anthropic  # imported here so the mock pipeline runs without the SDK installed
    client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY or an `ant auth login` profile
    kwargs = {"model": spec["model_id"], "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": user}]}
    if spec.get("supports_temperature", True):
        kwargs["extra_body"] = {"temperature": temperature}  # SDK 1.x dropped the typed argument; the API still takes it
    msg = client.messages.create(**kwargs)
    text = "".join(block.text for block in msg.content if block.type == "text")
    return text, msg.usage.input_tokens, msg.usage.output_tokens, msg.stop_reason


def post_chat(url, key_env, spec, system, user, temperature, max_tokens, token_field):
    """OpenAI-style chat completion over HTTP, with a few retries on rate limits and server errors."""
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"{key_env} is not set")
    body = {"model": spec["model_id"], token_field: max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if spec.get("supports_temperature", True):
        body["temperature"] = temperature
    for attempt in range(4):
        r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=300)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt * 5)
            continue
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return (choice["message"].get("content") or "", usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0), choice.get("finish_reason", ""))
    r.raise_for_status()
    raise RuntimeError(f"{url} kept failing with status {r.status_code}")


def call_openai(spec, system, user, temperature, max_tokens, cache_tag):
    return post_chat("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY", spec, system, user,
                     temperature, max_tokens, "max_completion_tokens")


def call_openrouter(spec, system, user, temperature, max_tokens, cache_tag):
    return post_chat("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", spec, system, user,
                     temperature, max_tokens, "max_tokens")
