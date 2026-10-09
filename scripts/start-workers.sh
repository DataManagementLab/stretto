#!/usr/bin/env bash
#
# Start the worker fleet described by a YAML config (default: scripts/cluster.yaml):
# `workers.nodes` nodes, `workers.per_node` workers on each, every one of them serving
# every experiment the config defines. A worker drains task 1, switches itself to task 2
# when that coordinator reports all_terminal, and exits after the last one - so this is
# launched once, not once per experiment.
#
# The coordinator for an experiment runs on the node the *same config* assigns it (see
# start-configurators.sh), and a worker on that node addresses it as localhost
# - the one hop that must *not* go through a proxy, and the only reason this script cares
# about the mapping at all. Pass both scripts the same --config and they cannot disagree.
#
# Every other hop uses the proxy environment exactly as the node's .bashrc sets it
# (no_proxy=localhost,127.0.0.1); this script does not add node names to NO_PROXY, since
# on clusters where nodes are only reachable through a forward proxy that would bypass it.
#
# Workers may be started before the coordinators exist: an unreachable coordinator is
# retried every 5s rather than skipped.
#
#   ./scripts/start-workers.sh --dry-run
#   ./scripts/start-workers.sh --nodes 4 --per-node 2
#   ./scripts/start-workers.sh --nodes 8-12          # that half of the fleet
#   ./scripts/start-workers.sh --config scripts/my-sweep.yaml --experiments samp01
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
O_NODES=""
O_PER_NODE=""
O_CAPABILITY=""
O_DEVICE=""
O_PIN_GB=""
O_ROTATE=""
SELECT=()
DRY_RUN=0
PYTHON="${PYTHON:-python}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

  --config PATH  experiment definitions (default: ${CONFIG}); the same file the
                 coordinators are launched from - it decides which tasks a worker drains
                 and on which node each one lives
  --experiments TASK_ID...
                 serve only these experiments out of the config
  --nodes SPEC   which nodes to start workers on: a bare count is the first N nodes
                 (4 = nodes 1-4), while a range or a comma list names
                 the nodes themselves - "8-12", "1,3,5-7". Node i is "<prefix>i", or the
                 i-th entry of cluster.hosts                (config: workers.nodes)
  --per-node N   how many workers per node; they share that node's backend servers
                                                           (config: workers.per_node)
  --prefix P     node name prefix; node i is "<P>i"        (config: cluster.prefix)
  --port N       coordinator port                          (config: cluster.port)
  --repo PATH    absolute repository path on the nodes     (config: cluster.repo)
  --bashrc PATH  shell profile to source on the nodes; holds conda
                                                           (config: cluster.bashrc)
  --home PATH    \$HOME to export on the nodes; the KV cache root defaults to
                 \$HOME/.reasondb/cache, and an ssh command otherwise inherits the
                 container's own HOME                      (config: cluster.home)
  --cache-dir P  export REASONDB_CACHE_DIR=P, overriding the \$HOME derivation
                                                           (config: cluster.cache_dir)
  --conda-env E  conda environment to activate             (config: cluster.conda_env)
  --capability C worker capability                         (config: workers.capability)
  --device D     torch device for each worker              (config: workers.device)
  --pin-gb GB    RAM budget for the KV caches an -in-memory operator pins, exported as
                 KV_CACHE_PIN_GB to each backend server the worker starts - per server
                 process, so a node running four of them needs four times this. 0 = no
                 -in-memory operator can be served      (config: workers.kv_cache_pin_gb)
  --user U       ssh as U@<node>                           (config: cluster.user)
  --rotate       shift each worker's task order by its index, so all coordinators get
                 work at once instead of the whole fleet queueing on task 1 (and nobody
                 idling on task 1's tail while its last jobs finish). 15 workers over 5
                 experiments is then 3 workers each, and none of them finishes early
  --no-rotate    the default; every worker drains the tasks in config order, so the whole
                 fleet finishes task 1 before starting task 2
                                                           (config: workers.rotate)
  --dry-run      print the ssh commands instead of running them
  -h, --help     this message

Every option above overrides the config file for this launch only.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --config)     CONFIG="$2"; shift 2 ;;
        --experiments)
            shift
            while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do SELECT+=("$1"); shift; done ;;
        --nodes|--servers) O_NODES="$2"; shift 2 ;;
        --per-node)   O_PER_NODE="$2"; shift 2 ;;
        --prefix)     O_PREFIX="$2"; shift 2 ;;
        --port)       O_PORT="$2"; shift 2 ;;
        --repo)       O_REPO="$2"; shift 2 ;;
        --bashrc)     O_BASHRC="$2"; shift 2 ;;
        --home)       O_HOME="$2"; shift 2 ;;
        --cache-dir)  O_CACHE_DIR="$2"; shift 2 ;;
        --conda-env)  O_CONDA_ENV="$2"; shift 2 ;;
        --capability) O_CAPABILITY="$2"; shift 2 ;;
        --device)     O_DEVICE="$2"; shift 2 ;;
        --pin-gb|--kv-cache-pin-gb) O_PIN_GB="$2"; shift 2 ;;
        --user)       O_USER="$2"; shift 2 ;;
        --rotate)     O_ROTATE="true"; shift ;;
        --no-rotate)  O_ROTATE="false"; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --tasks)      echo "--tasks is not supported: experiments are defined in ${CONFIG} (see --config/--experiments)." >&2; exit 2 ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# The planner owns everything the two scripts must agree on - node assignment, task order,
# which coordinator is reachable as localhost - so this script never recomputes any of it.
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
        --set "workers.nodes=${O_NODES}" \
        --set "workers.per_node=${O_PER_NODE}" \
        --set "workers.capability=${O_CAPABILITY}" \
        --set "workers.device=${O_DEVICE}" \
        --set "workers.kv_cache_pin_gb=${O_PIN_GB}" \
        --set "workers.rotate=${O_ROTATE}" \
        ${SELECT[@]+--experiments "${SELECT[@]}"}
}

# Assigned before eval'ing, so a planner that failed (bad config, unknown capability)
# kills the script here instead of eval'ing an empty string and launching nothing.
settings="$(plan settings)"
eval "$settings"
workers="$(plan workers)"

#: "user@" or "", prepended to every hostname this script hands to ssh. Kept apart from
#: the node name itself, which is also the --worker-id and must stay "node3", not
#: "user@node3".
SSH_AS="${SSH_USER:+${SSH_USER}@}"

started=0
while IFS=$'\t' read -r node worker_id args; do
    [ -n "$worker_id" ] || continue
    host="${SSH_AS}${node}"
    launch_log="/tmp/reasondb-launch-worker-${worker_id}.log"

    # The whole list is grouped and redirected as one, not just the python call: '&' binds
    # to the entire '... && ... && python ...' list, so bash backgrounds it in a subshell
    # whose own stdout/stderr are still ssh's channel - and ssh waits for every holder of
    # that pipe, not just for the shell it started. Redirecting only python leaves that
    # subshell holding it for as long as the worker runs, and the ssh never returns.
    #
    # A file rather than /dev/null because anything that fails *before* run_worker.py's own
    # daemonize() - a missing conda env, a malformed --tasks pair - only ever surfaces here.
    #
    # ';' after the source, not '&&': plenty of .bashrc files end on a non-zero status,
    # which would otherwise silently swallow the whole command.
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
    remote="${remote} conda activate ${CONDA_ENV} && cd ${REPO}"
    # Already escaped by the planner for this exact sandwich - see the coordinator script.
    remote="${remote} && python scripts/run_worker.py ${args}"
    remote="${remote} ) > ${launch_log} 2>&1 &"

    if [ "$DRY_RUN" -eq 1 ]; then
        # Escaped so the printed line can be pasted into a shell and behave identically.
        # The trailing '&' is part of that - it is what the launch path below does, so a
        # pasted batch overlaps its handshakes rather than paying a round trip per node.
        printf 'ssh -n %s "bash -ic %s" &\n' "$host" "'${remote//\$/\\$}'"
    else
        # Backgrounded so the fleet's handshakes overlap instead of costing a round trip
        # each: the remote list is already detached above, so this ssh only lives for the
        # handshake and the remote fork. '-n' and '</dev/null' are what make that safe -
        # a backgrounded ssh sharing this script's stdin would consume it.
        #
        # '&' binds to the whole '||' list, so the failure notice is what disappears if the
        # node is unreachable - without it the echo below (which now fires before ssh is
        # done) would report every node as launched.
        ssh -n "$host" "bash -ic '${remote}'" </dev/null \
            || echo "FAILED to launch on ${host}" >&2 &
        echo "${host}: ${worker_id} -> ${args#--tasks }"
    fi
    started=$((started + 1))
done <<<"$workers"

if [ "$DRY_RUN" -eq 1 ]; then
    exit 0
fi

# Nothing here waits on the workers themselves - they are daemonized on their nodes. This
# is only so the script does not exit (and hand its children a SIGHUP when the terminal
# goes away) while the last handshakes are still in flight.
wait

cat <<EOF

Started ${started} worker(s) on ${WORKER_NODES} node(s) (${WORKER_NODE_IDS}), from ${CONFIG}. Each writes to
${REPO}/benchmark_results/<task-id>/workers/<worker-id>/logging/worker.log, moving to the
next task's directory when it switches (the old log ends with a pointer to the new one).

Watch the queues - asked on each coordinator's own node, over localhost:

  for e in ${ENDPOINTS}; do t=\${e%%@*}; hp=\${e#*@}
    echo -n "\$t: "
    ssh -n ${SSH_AS}\${hp%%:*} "curl -s http://localhost:\${hp#*:}/api/task/\$t/summary"; echo
  done
EOF
