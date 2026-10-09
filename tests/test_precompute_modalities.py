"""`reasondb.utils.precompute_modalities.parse_precompute_skip_modalities`.

Parses REASONDB_PRECOMPUTE_SKIP_MODALITIES, the env var both
`Executor._precompute_pipeline` and `PlanConfigurator.prepare` read (via the
shared `PRECOMPUTE_SKIP_MODALITIES` module attribute) to decide which
modalities' KV cache servers a `--precompute` pass can run without.
"""

from reasondb.utils.precompute_modalities import parse_precompute_skip_modalities


def test_empty_string_skips_nothing():
    assert parse_precompute_skip_modalities("") == frozenset()


def test_single_modality():
    assert parse_precompute_skip_modalities("image") == frozenset({"image"})


def test_comma_separated_strips_whitespace_and_lowercases():
    assert parse_precompute_skip_modalities(" Image, AUDIO ,text") == frozenset(
        {"image", "audio", "text"}
    )


def test_blank_entries_dropped():
    assert parse_precompute_skip_modalities("image,,  ,text") == frozenset(
        {"image", "text"}
    )
