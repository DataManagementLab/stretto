#!/usr/bin/env bash
#
# Start one coordinator per experiment, one per cluster node. *Which* experiments those
# are - task id, producer, and the coordinator flags each one is launched with - lives in
# a YAML config (default: scripts/cluster.yaml), not in this script, so a new
# experiment is a config entry rather than an edit here. Pair with
# start-workers.sh, pointed at the *same* --config: it computes the same node
# assignment from it, which is how its workers find these coordinators.
#
# Each remote command runs under `bash -ic` on purpose: .bashrc is what provides the
# `conda` shell function, and a non-interactive shell usually returns from .bashrc before
# it is defined. Nothing is nohup'd because run_coordinator.py daemonizes itself (ignores
# SIGHUP, redirects stdio into <output-dir>/logging/coordinator.log).
#
#   ./scripts/start-configurators.sh --dry-run
#   ./scripts/start-configurators.sh --config scripts/my-sweep.yaml
#   ./scripts/start-configurators.sh --experiments base01 samp01
#
set -euo pipefail

CONFIG="scripts/cluster.yaml"
#: Every override is forwarded to the planner as --set <section>.<key>=<value>, and an
#: empty one is ignored there - so "not passed" cannot clobber what the config says.
O_PREFIX=""
O_PORT=""
O_REPO=""
O_BASHRC=""
O_HOME=""
O_CACHE_DIR=""
O_CONDA_ENV=""
O_USER=""
SELECT=()
DRY_RUN=0
PYTHON="${PYTHON:-python}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

  --config PATH  experiment definitions (default: ${CONFIG}); see that file for the
                 schema - it is where task ids, producers and their coordinator flags
                 are defined
  --experiments TASK_ID...
                 launch only these experiments out of the config, without moving any of
                 them to a different node
  --prefix P     node name prefix; node i is "<P>i"        (config: cluster.prefix)
  --port N       coordinator port                          (config: cluster.port)
  --repo PATH    absolute repository path on the nodes     (config: cluster.repo)
  --bashrc PATH  shell profile to source on the nodes; holds the conda init block
                                                           (config: cluster.bashrc)
  --home PATH    \$HOME to export on the nodes; the KV cache root defaults to
                 \$HOME/.reasondb/cache, and an ssh command otherwise inherits the
                 container's own HOME                      (config: cluster.home)
  --cache-dir P  export REASONDB_CACHE_DIR=P, overriding the \$HOME derivation
                                                           (config: cluster.cache_dir)
  --conda-env E  conda environment to activate             (config: cluster.conda_env)
  --user U       ssh as U@<node>                           (config: cluster.user)
  --dry-run      print the ssh commands instead of running them
  -h, --help     this message

Every option above overrides the config file for this launch only.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --config)    CONFIG="$2"; shift 2 ;;
        --experiments)
            shift
            while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do SELECT+=("$1"); shift; done ;;
        --prefix)    O_PREFIX="$2"; shift 2 ;;
        --port)      O_PORT="$2"; shift 2 ;;
        --repo)      O_REPO="$2"; shift 2 ;;
        --bashrc)    O_BASHRC="$2"; shift 2 ;;
        --home)      O_HOME="$2"; shift 2 ;;
        --cache-dir) O_CACHE_DIR="$2"; shift 2 ;;
        --conda-env) O_CONDA_ENV="$2"; shift 2 ;;
        --user)      O_USER="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --tasks)     echo "--tasks is not supported: experiments are defined in ${CONFIG} (see --config/--experiments)." >&2; exit 2 ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# One planner invocation decides everything both scripts need to agree on; this one asks
# it twice (settings, then the per-coordinator lines) rather than duplicating any of it.
plan() {
    "$PYTHON" -m reasondb.coordinator.cluster --config "$CONFIG" --emit "$1" \
        --set "cluster.prefix=${O_PREFIX}" \
        --set "cluster.port=${O_PORT}" \
        --set "cluster.repo=${O_REPO}" \
        --set "cluster.bashrc=${O_BASHRC}" \
        --set "cluster.home=${O_HOME}" \
        --set "cluster.cache_dir=${O_CACHE_DIR}" \
        --set "cluster.conda_env=${O_CONDA_ENV}" \
        --set "cluster.user=${O_USER}" \
        ${SELECT[@]+--experiments "${SELECT[@]}"}
}

# Assigned before eval'ing, so a planner that failed (bad config, unknown producer) kills
# the script here instead of eval'ing an empty string and launching nothing.
settings="$(plan settings)"
eval "$settings"
coordinators="$(plan coordinators)"

#: "user@" or "", prepended to every hostname this script hands to ssh.
SSH_AS="${SSH_USER:+${SSH_USER}@}"

launched=0
while IFS=$'\t' read -r node task producer port args; do
    [ -n "$task" ] || continue
    host="${SSH_AS}${node}"
    launch_log="/tmp/reasondb-launch-${task}.log"

    # ';' after the source, not '&&': plenty of .bashrc files end on a non-zero status,
    # which would otherwise silently swallow the whole command. The opening '(' is closed
    # below - see there for why the whole list, not just python, is grouped.
    remote="("
    if [ -n "$HOME_DIR" ]; then
        remote="${remote} export HOME=${HOME_DIR};"
    fi
    if [ -n "$CACHE_DIR" ]; then
        remote="${remote} export REASONDB_CACHE_DIR=${CACHE_DIR};"
    fi
    if [ -n "$BASHRC" ]; then
        remote="${remote} . ${BASHRC};"
    fi
    remote="${remote} conda activate ${CONDA_ENV} && cd ${REPO} && python scripts/run_coordinator.py"
    # The arguments come from the planner already escaped for this exact sandwich: no
    # quotes (which the single-quoted remote command below could not carry), spaces
    # backslashed, and any $VAR left alone so the node's own shell expands it.
    remote="${remote} ${args}"
    # The whole list is grouped and redirected as one, not just the python call: '&' binds
    # to the entire '... && ... && python ...' list, so bash backgrounds it in a subshell
    # whose own stdout/stderr are still ssh's channel - and ssh waits for every holder of
    # that pipe, not just for the shell it started. Redirecting only python leaves that
    # subshell holding it for as long as the coordinator runs, and the ssh never returns.
    #
    # A file rather than /dev/null because anything that fails *before* run_coordinator.py's
    # own daemonize() - a missing conda env, a bad --producer - only ever surfaces here.
    remote="${remote} ) > ${launch_log} 2>&1 &"

    if [ "$DRY_RUN" -eq 1 ]; then
        # Escaped so the printed line can be pasted into a shell and behave identically:
        # unescaped, the local shell would expand a config's $VAR here rather than leaving
        # it for the node, which is where it is meant to resolve. The
        # trailing '&' is part of that - it is what the launch path below does, so a
        # pasted batch overlaps its handshakes rather than paying a round trip per node.
        printf 'ssh -n %s "bash -ic %s" &\n' "$host" "'${remote//\$/\\$}'"
    else
        # Backgrounded so the handshakes overlap instead of costing a round trip each: the
        # remote list is already detached above, so this ssh only lives for the handshake
        # and the remote fork. '-n' and '</dev/null' are what make that safe - a
        # backgrounded ssh sharing this script's stdin would consume it.
        #
        # '&' binds to the whole '||' list, so the failure notice is what disappears if the
        # node is unreachable - without it the echo below (which now fires before ssh is
        # done) would report every node as launched.
        ssh -n "$host" "bash -ic '${remote}'" </dev/null \
            || echo "FAILED to launch on ${host}" >&2 &
        echo "${host}: ${task} (${producer}) on port ${port} - launch output in ${launch_log}"
    fi
    launched=$((launched + 1))
done <<<"$coordinators"

if [ "$DRY_RUN" -eq 1 ]; then
    exit 0
fi

# Nothing here waits on the coordinators themselves - they are daemonized on their nodes.
# This is only so the script does not exit (and hand its children a SIGHUP when the
# terminal goes away) while the last handshakes are still in flight.
wait

cat <<EOF

Started ${launched} coordinator(s) from ${CONFIG}. Confirm each one bound the port it was
given rather than scanning past an occupied one (find_free_port walks the next 10).
Backgrounded per endpoint, unlike the worker script's equivalent check: that one curls
localhost on the node and returns in milliseconds, while this greps a whole
coordinator.log off the shared FS, so run serially it costs seconds *per task*. The
subshell composes the whole line before printing it, so the parallel replies stay
attributable to their task rather than interleaving mid-line.

  for e in ${ENDPOINTS}; do t=\${e%%@*}; hp=\${e#*@}
    ( echo "\$t: \$(ssh -n ${SSH_AS}\${hp%%:*} "grep -h 'dashboard/API at' ${REPO}/benchmark_results/\$t/logging/coordinator.log | tail -1" </dev/null)" ) &
  done; wait
EOF
