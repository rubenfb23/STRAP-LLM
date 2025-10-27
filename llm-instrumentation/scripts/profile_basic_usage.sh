#!/usr/bin/env bash
# Profiles the basic_usage example with Nsight Systems and Nsight Compute.
set -euo pipefail

usage() {
  cat <<'EOF'
Profile the llm-instrumentation examples/basic_usage.py script with Nsight Systems
and Nsight Compute. Results are written to the chosen output directory.

Usage: profile_basic_usage.sh [options] [-- python_args...]

Options:
  --output-dir DIR   Directory where profiler outputs are written (default: profiles)
  --python BIN       Python interpreter to use (default: python3)
  --help             Show this message

Any arguments after "--" are forwarded to basic_usage.py.
EOF
}

OUTPUT_DIR="profiles"
PYTHON_BIN="${PYTHON:-python3}"
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --output-dir" >&2; exit 1; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -lt 2 ]] && { echo "Missing value for --python" >&2; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      FORWARDED_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

command -v nsys >/dev/null 2>&1 || { echo "nsys not found in PATH" >&2; exit 1; }
command -v ncu >/dev/null 2>&1 || { echo "ncu not found in PATH" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${PROJECT_ROOT}/examples/basic_usage.py"

[[ -f "${PYTHON_SCRIPT}" ]] || { echo "Cannot locate ${PYTHON_SCRIPT}" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

echo "Profiling with Nsight Systems..."
nsys profile \
  --force-overwrite true \
  --trace=cuda,nvtx,cudnn,cublas \
  --sample=none \
  --output "${OUTPUT_DIR}/basic_usage_nsys" \
  "${PYTHON_BIN}" "${PYTHON_SCRIPT}" "${FORWARDED_ARGS[@]}" || {
    echo "Nsight Systems profiling failed" >&2
    exit 1
  }

echo "Profiling with Nsight Compute..."
ncu \
  --force-overwrite \
  --set full \
  --target-processes all \
  --launch-skip 0 \
  --launch-count 1 \
  --export "${OUTPUT_DIR}/basic_usage_ncu" \
  "${PYTHON_BIN}" "${PYTHON_SCRIPT}" "${FORWARDED_ARGS[@]}" || {
    echo "Nsight Compute profiling failed" >&2
    exit 1
  }

echo "Profiles saved under ${OUTPUT_DIR}"
