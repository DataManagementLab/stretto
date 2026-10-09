"""What a benchmark's jobs actually require, as capability names.

The mirror of ``scheduler.CAPABILITY_PROVIDES``: that says what a worker brings, this
says what a job asks for, and ``worker_can_run`` matches the two. Derived from the
benchmark's own column declarations rather than hardcoded, because they have to agree
and only the benchmark knows.

``get_default_configurator`` builds every modality's operators regardless of which
modalities a benchmark uses, but that does not make every server *necessary*:

- ``prepare()`` loops over ``database.get_concrete_columns_by_type(<its modality>)``, so
  an operator whose modality the benchmark has no columns for prepares nothing and never
  contacts a server (unlike ``REASONDB_PRECOMPUTE_SKIP_MODALITIES``, which exists for the
  different case of a benchmark that *does* have the modality).
- A failed ``setup()`` is caught per operator and downgraded to "this operator
  will not be available", and such an operator is never a candidate for the benchmark's
  expressions anyway - a text filter cannot serve ``{artworks.image} depicts ...``.

Requiring only the modalities a benchmark has lets text and image work be split across
workers instead of requiring every worker to run ``--capability both``.
"""

from typing import List

from reasondb.coordinator.models import (
    CAP_AUDIO_KV,
    CAP_EMBEDDING,
    CAP_IMAGE_KV,
    CAP_TEXT_KV,
)


def capabilities_for_benchmark(benchmark_cls, split: str) -> List[str]:
    """The capabilities a job over *benchmark_cls* genuinely needs.

    ``embedding`` unconditionally: the similarity backends assert readiness on every run
    regardless of modality (see ``ImageSimilarityBackend.assert_ready``), which is also
    why a ``--capability simulate`` worker still starts those two servers.

    Loading here is cheap and deliberate: ``load_without_queries`` registers the table
    descriptors without generating a query set, and ``ExternalTable.__init__`` stores its
    column lists without reading a row.

    ``load_without_queries`` is a ``RandomBenchmark`` method, and it exists to stop a load
    from *sampling* a query set and making it authoritative. A fixed benchmark has nothing
    to sample - ``get_queries()`` returns a module constant - so ``load`` is both
    available and equally free of that hazard, and is used as the fallback.
    """
    load = getattr(benchmark_cls, "load_without_queries", None) or benchmark_cls.load
    database = load(split).database
    tables = list(getattr(database, "external_tables", []))
    required = [CAP_EMBEDDING]
    if any(table.text_columns for table in tables):
        required.append(CAP_TEXT_KV)
    if any(table.image_columns for table in tables):
        required.append(CAP_IMAGE_KV)
    if any(table.audio_columns for table in tables):
        required.append(CAP_AUDIO_KV)
    return required
