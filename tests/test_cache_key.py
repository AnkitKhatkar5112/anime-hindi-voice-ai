"""Property-based tests for PipelineCache.cache_key.

Feature: production-dub-pipeline, Property 2: cache_key output is deterministic
and unique across distinct inputs (cache key uniqueness).

Uses Hypothesis to verify:
1. Determinism  — same inputs always produce the same key.
2. Uniqueness   — distinct input tuples produce distinct keys.
3. Format       — key is always a 64-char lowercase hex string.
4. Order        — argument order affects the key (no commutativity).
"""

import tempfile

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scripts.inference.pipeline_cache import PipelineCache

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_text_st = st.text(min_size=0, max_size=200)
_lang_st = st.text(alphabet="abcdefghijklmnopqrstuvwxyz-", min_size=2, max_size=10)
_backend_st = st.sampled_from(["helsinki", "deepl", "google", "nemo", "coqui", "gtts", "bark"])


def _make_cache() -> PipelineCache:
    """Return a PipelineCache backed by a fresh temporary directory."""
    tmp = tempfile.mkdtemp()
    return PipelineCache(cache_root=tmp)


# ---------------------------------------------------------------------------
# Property 1: Determinism
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=_text_st, src_lang=_lang_st, tgt_lang=_lang_st, backend=_backend_st)
def test_cache_key_is_deterministic(text, src_lang, tgt_lang, backend):
    """cache_key(*parts) returns the same value on repeated calls with identical parts.

    # Feature: production-dub-pipeline, Property 2: cache key determinism
    """
    cache = _make_cache()
    key_a = cache.cache_key(text, src_lang, tgt_lang, backend)
    key_b = cache.cache_key(text, src_lang, tgt_lang, backend)
    assert key_a == key_b, (
        f"cache_key is not deterministic for inputs "
        f"({text!r}, {src_lang!r}, {tgt_lang!r}, {backend!r}): "
        f"{key_a!r} != {key_b!r}"
    )


# ---------------------------------------------------------------------------
# Property 2: Uniqueness across distinct inputs
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    text_a=_text_st, src_lang_a=_lang_st, tgt_lang_a=_lang_st, backend_a=_backend_st,
    text_b=_text_st, src_lang_b=_lang_st, tgt_lang_b=_lang_st, backend_b=_backend_st,
)
def test_cache_key_is_unique_for_distinct_inputs(
    text_a, src_lang_a, tgt_lang_a, backend_a,
    text_b, src_lang_b, tgt_lang_b, backend_b,
):
    """Distinct (text, src_lang, tgt_lang, backend) tuples produce distinct keys.

    # Feature: production-dub-pipeline, Property 2: cache key uniqueness
    """
    from hypothesis import assume
    assume(
        (text_a, src_lang_a, tgt_lang_a, backend_a)
        != (text_b, src_lang_b, tgt_lang_b, backend_b)
    )
    cache = _make_cache()
    key_a = cache.cache_key(text_a, src_lang_a, tgt_lang_a, backend_a)
    key_b = cache.cache_key(text_b, src_lang_b, tgt_lang_b, backend_b)
    assert key_a != key_b, (
        f"cache_key collision detected:\n"
        f"  input A: ({text_a!r}, {src_lang_a!r}, {tgt_lang_a!r}, {backend_a!r})\n"
        f"  input B: ({text_b!r}, {src_lang_b!r}, {tgt_lang_b!r}, {backend_b!r})\n"
        f"  shared key: {key_a!r}"
    )


# ---------------------------------------------------------------------------
# Property 3: Key format — always a 64-char lowercase hex string
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(parts=st.lists(_text_st, min_size=1, max_size=5))
def test_cache_key_format(parts):
    """cache_key always returns a 64-character lowercase hex string.

    # Feature: production-dub-pipeline, Property 2: cache key format invariant
    """
    cache = _make_cache()
    key = cache.cache_key(*parts)
    assert len(key) == 64, f"Expected 64-char key, got {len(key)}: {key!r}"
    assert key == key.lower(), f"Key is not lowercase: {key!r}"
    assert all(c in "0123456789abcdef" for c in key), f"Key is not hex: {key!r}"


# ---------------------------------------------------------------------------
# Property 4: Argument order matters (no commutativity)
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(a=_text_st, b=_text_st)
def test_cache_key_order_sensitive(a, b):
    """cache_key(a, b) != cache_key(b, a) when a != b.

    # Feature: production-dub-pipeline, Property 2: cache key order sensitivity
    """
    from hypothesis import assume
    assume(a != b)
    cache = _make_cache()
    assert cache.cache_key(a, b) != cache.cache_key(b, a), (
        f"cache_key is commutative for ({a!r}, {b!r}) — "
        "argument order must affect the key"
    )
