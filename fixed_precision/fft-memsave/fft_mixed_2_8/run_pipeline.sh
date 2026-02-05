#!/usr/bin/env bash
# Minimal + robust pipeline runner
# - Prints every step
# - Clear error reporting (trap)
# - No awk/sed needed; regex parse with BASH_REMATCH

set -Eeuo pipefail
trap 'code=$?; echo "❌ Error at line $LINENO: ${BASH_COMMAND}" >&2; exit $code' ERR

# Enable trace if requested: DEBUG=1 ./run_pipeline.sh
[[ "${DEBUG:-0}" == "1" ]] && set -x

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
CONFIG_FILE="${1:-$SCRIPT_DIR/config.sh}"

echo "➡ Using config: $CONFIG_FILE"
[[ -f "$CONFIG_FILE" ]] || { echo "❌ Config not found: $CONFIG_FILE" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${PYTHON_BIN:=python3}"
: "${PY_SCRIPT:?Set PY_SCRIPT in config.sh (e.g., PY_SCRIPT=./data_generator.py)}"
: "${MAKE_BIN:=make}"
: "${MAKE_TARGETS:=clean all run}"
LOG_DIR="${LOG_DIR:-"$SCRIPT_DIR/logs"}"
# Clear previous logs
if [[ -d "$LOG_DIR" ]]; then
  rm -f "$LOG_DIR"/*
fi
mkdir -p "$LOG_DIR"
[[ -w "$LOG_DIR" ]] || { echo "❌ LOG_DIR not writable: $LOG_DIR" >&2; exit 1; }
echo "➡ Logs will be saved to: $LOG_DIR"

command -v "$PYTHON_BIN" >/dev/null || { echo "❌ Missing: $PYTHON_BIN" >&2; exit 1; }
command -v "$MAKE_BIN" >/dev/null || { echo "❌ Missing: $MAKE_BIN" >&2; exit 1; }

# Validate RUNS
if ! declare -p RUNS >/dev/null 2>&1 || ! declare -p RUNS 2>/dev/null | grep -q 'declare \-a RUNS'; then
  echo "❌ RUNS must be an array in config.sh, e.g. RUNS=( \"PY: ... ; MAKE: ...\" )" >&2
  exit 1
fi
num_runs=${#RUNS[@]}
echo "➡ Found $num_runs run(s) in RUNS"
(( num_runs > 0 )) || { echo "❌ RUNS is empty" >&2; exit 1; }

# Echo raw runs for visibility
for i in "${!RUNS[@]}"; do
  printf '• RUNS[%d] raw: %q\n' "$i" "${RUNS[$i]}"
done

run_idx=0
overall_fail=0

for run in "${RUNS[@]}"; do
  (( ++run_idx ))         # prefix increment evaluates to 1 on first loop
  echo -e "\n=============================="
  echo "RUN #$run_idx — parsing"

  # Regex: capture PY args (group 1) and MAKE args (group 2)
  # Accept optional spaces around ';'
# Safe split without regex (works across bash variants)
delim="; MAKE:"
if [[ "$run" != *"$delim"* || "$run" != *"PY:"* ]]; then
  echo "❌ Run #$run_idx malformed. Expected 'PY: ... ; MAKE: ...' Got: $run" >&2
  overall_fail=1
  continue
fi

# Extract pieces
py_part="${run#*PY:}"               # drop up to 'PY:'
py_args="${py_part%%$delim*}"       # take everything before '; MAKE:'
make_args="${run#*"$delim"}"        # take everything after '; MAKE:'

# Trim leading/trailing whitespace (portable enough)
trim() { awk '{$1=$1; print}' <<<"$1"; }
py_args="$(trim "$py_args")"
make_args="$(trim "$make_args")"

  ts="$(date +"%Y%m%d_%H%M%S")"
  py_log="$LOG_DIR/run${run_idx}_python.log"
  mk_log="$LOG_DIR/run${run_idx}_make.log"

  echo "Parsed:"
  echo "  PY args  : $py_args"
  echo "  MAKE args: $make_args"

  echo -e "\nRUN #$run_idx — PYTHON"
  echo "Cmd: $PYTHON_BIN $PY_SCRIPT $py_args"
  set -o pipefail
  # shellcheck disable=SC2086
  "$PYTHON_BIN" "$PY_SCRIPT" $py_args 2>&1 | tee "$py_log"
  py_status=${PIPESTATUS[0]}
  echo "[Run #$run_idx] Python exit: $py_status"
  if (( py_status != 0 )); then
    echo "⚠️  Python failed. See: $py_log" >&2
    overall_fail=1
    continue
  fi

  echo -e "\nRUN #$run_idx — MAKE"
  echo "Cmd: $MAKE_BIN $MAKE_TARGETS $make_args"
  # shellcheck disable=SC2086
  $MAKE_BIN $MAKE_TARGETS $make_args 2>&1 | tee "$mk_log"
  mk_status=${PIPESTATUS[0]}
  echo "[Run #$run_idx] Make exit: $mk_status"
  if (( mk_status != 0 )); then
    echo "⚠️  Make failed. See: $mk_log" >&2
    overall_fail=1
    continue
  fi

  echo -e "\n[Run #$run_idx] ✅ Completed."
  echo "  Python log: $py_log"
  echo "  Make log  : $mk_log"
done

echo -e "\n=============================="
echo "ALL RUNS FINISHED"
if (( overall_fail != 0 )); then
  echo "Some runs failed. Check logs in: $LOG_DIR"
  exit 1
else
  echo "All runs succeeded 🎉  Logs in: $LOG_DIR"
fi
