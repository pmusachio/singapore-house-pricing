#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
DEFAULT_SAMPLE_ROWS=8000
SAMPLE_ROWS="${DEFAULT_SAMPLE_ROWS}"
MODE="smoke"
RUN_TESTS="false"

print_help() {
  cat <<EOF
Usage:
  ./scripts/run_project.sh
  ./scripts/run_project.sh --full
  ./scripts/run_project.sh --sample-rows 12000
  ./scripts/run_project.sh --with-tests

What this script does:
  1. Creates the virtual environment in .venv
  2. Upgrades pip, wheel, and setuptools
  3. Installs the project in editable mode
  4. Prepares the processed dataset
  5. Trains the model
  6. Prints the project summary at the end

Options:
  --full           Run training on the full dataset
  --sample-rows N  Per-split sample size for a faster smoke test
  --with-tests     Run pytest after training
  -h, --help       Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      MODE="full"
      shift
      ;;
    --sample-rows)
      SAMPLE_ROWS="${2:-}"
      if [[ -z "${SAMPLE_ROWS}" ]]; then
        echo "Missing value for --sample-rows" >&2
        exit 1
      fi
      MODE="smoke"
      shift 2
      ;;
    --with-tests)
      RUN_TESTS="true"
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use --help to see available options." >&2
      exit 1
      ;;
  esac
done

cd "${PROJECT_ROOT}"

echo "==> Creating virtual environment"
python3 -m venv "${VENV_DIR}"

echo "==> Upgrading pip tooling"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel setuptools

echo "==> Installing project dependencies"
"${VENV_DIR}/bin/pip" install -e ".[dev]"

mkdir -p "${PROJECT_ROOT}/.cache/matplotlib"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"

echo "==> Preparing engineered dataset"
"${VENV_DIR}/bin/singapore-house-pricing" prepare-data

echo "==> Training model"
if [[ "${MODE}" == "full" ]]; then
  "${VENV_DIR}/bin/singapore-house-pricing" train
else
  "${VENV_DIR}/bin/singapore-house-pricing" train --sample-rows "${SAMPLE_ROWS}"
fi

if [[ "${RUN_TESTS}" == "true" ]]; then
  echo "==> Running tests"
  "${VENV_DIR}/bin/pytest" -q
fi

echo
echo "==> Project summary"
if [[ -f "${PROJECT_ROOT}/reports/project_summary.md" ]]; then
  cat "${PROJECT_ROOT}/reports/project_summary.md"
else
  echo "Summary file not found."
fi

echo
echo "Generated files:"
echo "  - ${PROJECT_ROOT}/data/processed/"
echo "  - ${PROJECT_ROOT}/models/"
echo "  - ${PROJECT_ROOT}/reports/"
