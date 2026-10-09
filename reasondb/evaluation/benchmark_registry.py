"""The name -> benchmark-class registry, in one place.

``ALL_BENCHMARKS`` holds every benchmark; ``RANDOM_BENCHMARKS``, the subset the KV
sweeps can run, is derived from it.

This module deliberately imports nothing but the benchmark classes themselves, so that
callers that only need to turn a ``--benchmarks`` string into a class (plotting and
analysis scripts, CLI argument parsing) do not import the whole engine.
"""

from typing import Dict, Type

from reasondb.evaluation.benchmarks.artwork import (
    Artwork,
    ArtworkLarge,
    ArtworkRandom,
    ArtworkRandomMedium,
)
from reasondb.evaluation.benchmarks.ecommerce import (
    EcommerceLarge,
    EcommerceRandom,
    EcommerceRandomLarge,
)
from reasondb.evaluation.benchmarks.curated import (
    ArtworkCurated,
    EcommerceCurated,
    EnronEmailCurated,
    MovieHugeCurated,
    RotowireCurated,
)
from reasondb.evaluation.benchmarks.email import EnronEmail, EnronEmailRandom
from reasondb.evaluation.benchmarks.movie import Movie, MovieRandom, MovieRandomHuge
from reasondb.evaluation.benchmarks.real_estate import RealEstate
from reasondb.evaluation.benchmarks.rotowire import Rotowire, RotowireRandom


def _with_hyphen_aliases(benchmarks: Dict[str, Type]) -> Dict[str, Type]:
    """Accept ``movie-random`` wherever ``movie_random`` works."""
    return {**benchmarks, **{k.replace("_", "-"): v for k, v in benchmarks.items()}}


#: Every benchmark in the repo, keyed as the CLI spells it.
_ALL: Dict[str, Type] = {
    # CAESURA
    "artwork": Artwork,
    "artwork_large": ArtworkLarge,
    "artwork_random": ArtworkRandom,
    "artwork_random_medium": ArtworkRandomMedium,
    "rotowire": Rotowire,
    "rotowire_random": RotowireRandom,
    #
    # Palimpzest
    "real_estate": RealEstate,
    "enron_email": EnronEmail,
    "email_random": EnronEmailRandom,
    #
    # Sembench
    "movie": Movie,
    "movie_random": MovieRandom,
    "movie_random_huge": MovieRandomHuge,
    "ecommerce_large": EcommerceLarge,
    "ecommerce_random": EcommerceRandom,
    "ecommerce_random_large": EcommerceRandomLarge,
    #
    # Curated: fixed query sets whose every semantic step carries per-tuple ground
    # truth, i.e. what `--human-labels` can be run against. Not in _RANDOM_NAMES below,
    # since they are fixed benchmarks with few queries.
    "artwork_curated": ArtworkCurated,
    "email_curated": EnronEmailCurated,
    "rotowire_curated": RotowireCurated,
    "ecommerce_curated": EcommerceCurated,
    "movie_huge_curated": MovieHugeCurated,
}

#: The subset the KV sweeps (storage, sample size) support: the ``RandomBenchmark``
#: variants, which generate their query set from ``OPERATOR_OPTIONS``/
#: ``QUERY_SHAPES`` rather than carrying a hand-written one. The fixed benchmarks
#: are kept out because a sweep needs many comparable queries per configuration.
_RANDOM_NAMES = (
    "artwork_random",
    "artwork_random_medium",
    "rotowire_random",
    "email_random",
    "movie_random",
    "movie_random_huge",
    "ecommerce_random",
    "ecommerce_random_large",
)

ALL_BENCHMARKS: Dict[str, Type] = _with_hyphen_aliases(_ALL)

RANDOM_BENCHMARKS: Dict[str, Type] = _with_hyphen_aliases(
    {name: _ALL[name] for name in _RANDOM_NAMES}
)
