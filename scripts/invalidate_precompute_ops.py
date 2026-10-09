"""Drop selected operators from a ``--precompute`` JSON's resume markers.

``--precompute`` records, alongside the model responses, one ``precomputed_ops``
marker per operator it has finished (see ``Executor._precompute_pipeline``).
Re-running ``--precompute`` against an existing JSON resumes: any operator whose
marker is present is skipped outright. That is what you want after an
interruption, but not when a backend records calls that an existing JSON does not
yet contain: a resumed precompute would skip the operator and ``--simulate`` would
fail on the missing entries. Dropping just that operator's markers makes the next
precompute re-run it (and only it) so the missing calls get captured.

``text_qa`` / ``vision`` responses are always left alone: they are keyed by
content, so stale entries are harmless and the phases that *are* already
recorded keep replaying.

``operator_configs`` - the pinned LLM configuration (question phrasing and
friends) per operator interface - are left alone by default, and ``--drop-configs``
is the deliberate exception. Dropping a pin lets the LLM re-derive a different
configuration on the next run, which orphans every response recorded under the
old phrasing; that is the wrong trade when the pin is merely stale. It is the
*only* option when the operator's prompt shape itself changed and the pinned
value can no longer be used at all (e.g. it fails parameter validation). The
orphaned responses are the accepted cost; the same ``--operators`` prefixes drop the resume markers in
the same pass, so the operator re-records from scratch.

Re-running an operator re-issues *all* of its model calls, not only the newly
covered ones, so the relevant servers must be up for that pass.

Examples
--------
    # What would be dropped (default: nothing is written)
    python scripts/invalidate_precompute_ops.py artwork_precompute.json \
        --operators ExtractAndQaImageFilter

    # Do it, keeping artwork_precompute.json.bak
    python scripts/invalidate_precompute_ops.py artwork_precompute.json \
        --operators ExtractAndQaImageFilter --apply

    # Also unpin the LLM configuration, because the prompt shape changed
    python scripts/invalidate_precompute_ops.py artwork_precompute.json \
        --operators ExtractAndQaImageFilter --drop-configs --apply

    # Which operators are in there at all?
    python scripts/invalidate_precompute_ops.py artwork_precompute.json --list
"""

import argparse
import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

# Marker layout:      "<operation_identifier>|<expression>|<base_tables>".
# Config key layout:  "<interface_name>|<expression>|<base_tables>".
# The two differ in the first field only: a marker names one physical variant
# (backend and model included), a config key names the interface they share.
KEY_SEPARATOR = "|"


def operator_of(key: str) -> str:
    """The operator a resume marker or a pinned-config key belongs to."""
    return key.split(KEY_SEPARATOR, 1)[0]


def matches(key: str, prefixes: Sequence[str]) -> bool:
    """Whether this key's operator starts with any of *prefixes*.

    Prefix matching is what makes ``--operators ExtractAndQaImageFilter`` cover
    every backend/compression variant, since the identifier carries the backend
    and model in full (e.g.
    ``ExtractAndQaImageFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0``).
    The interface name a config is keyed by is a prefix of exactly those
    identifiers, so one prefix list selects a marker set and its configs together.

    One consequence worth knowing: a full identifier is also a prefix of its
    ``-in-memory`` sibling, so ``--operators …-cr0.8`` drops the markers for the
    RAM-served operator as well as the disk one. Usually what you want (they answer
    identically, and a re-record fills both); name the suffix explicitly if not.
    """
    operator = operator_of(key)
    return any(operator.startswith(prefix) for prefix in prefixes)


def summarize(markers: Sequence[str]) -> List[str]:
    """One ``<count>x <operator>`` line per distinct operator, most frequent first."""
    counts = Counter(operator_of(m) for m in markers)
    return [f"{count:5d}x {operator}" for operator, count in counts.most_common()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("precompute_json", type=Path)
    parser.add_argument(
        "--operators",
        nargs="+",
        default=[],
        metavar="PREFIX",
        help="Operator identifier prefixes whose resume markers to drop, e.g. "
        "ExtractAndQaImageFilter. Matches every backend/compression variant.",
    )
    parser.add_argument(
        "--drop-configs",
        action="store_true",
        help="Also drop the pinned LLM configuration of the same operators. Only "
        "when their prompt shape changed and the pinned value can no longer be "
        "used - it orphans every response recorded under the old phrasing.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only print the operators present in the file, then exit.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the change. Without it, the file is left alone (dry run).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the <name>.bak copy that --apply writes first.",
    )
    args = parser.parse_args()

    with open(args.precompute_json) as f:
        data = json.load(f)
    markers: List[str] = data.get("precomputed_ops", [])
    configs: Dict[str, Any] = data.get("operator_configs", {})
    logger.info(
        "%s: %d resume markers, %d pinned operator configs",
        args.precompute_json,
        len(markers),
        len(configs),
    )

    if args.list:
        for line in summarize(markers):
            logger.info("  %s", line)
        return

    if not args.operators:
        parser.error("pass --operators PREFIX [...] (or --list to see what is there)")

    doomed = [m for m in markers if matches(m, args.operators)]
    doomed_configs = (
        [k for k in configs if matches(k, args.operators)] if args.drop_configs else []
    )
    if not doomed and not doomed_configs:
        logger.info(
            "Nothing matches %s. Present operators:", ", ".join(args.operators)
        )
        for line in summarize(markers):
            logger.info("  %s", line)
        return

    logger.info("Dropping %d of %d markers:", len(doomed), len(markers))
    for line in summarize(doomed):
        logger.info("  %s", line)
    if args.drop_configs:
        logger.info(
            "Dropping %d of %d pinned configs (their recorded responses are "
            "orphaned - the next precompute re-derives the configuration):",
            len(doomed_configs),
            len(configs),
        )
        for line in summarize(doomed_configs):
            logger.info("  %s", line)

    if not args.apply:
        logger.info("Dry run - nothing written. Re-run with --apply to write.")
        return

    if not args.no_backup:
        backup = args.precompute_json.with_suffix(args.precompute_json.suffix + ".bak")
        shutil.copy2(args.precompute_json, backup)
        logger.info("Backed up to %s", backup)

    data["precomputed_ops"] = [m for m in markers if m not in set(doomed)]
    if doomed_configs:
        data["operator_configs"] = {
            k: v for k, v in configs.items() if k not in set(doomed_configs)
        }
    with open(args.precompute_json, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(
        "Wrote %s (%d markers, %d pinned configs left). Re-run --precompute with "
        "the servers up to recapture these operators.",
        args.precompute_json,
        len(data["precomputed_ops"]),
        len(data.get("operator_configs", {})),
    )


if __name__ == "__main__":
    main()
