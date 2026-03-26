#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-autodl}"
BINARY_CONFIG="${BINARY_CONFIG:-configs/binary_thesis_public.yaml}"
LANDMARKS_CONFIG="${LANDMARKS_CONFIG:-configs/landmarks_cat_dataset.yaml}"
DOWNLOAD_BINARY_DATASETS="${DOWNLOAD_BINARY_DATASETS:-1}"
DOWNLOAD_CAT_DATASET="${DOWNLOAD_CAT_DATASET:-1}"
TRAIN_BINARY="${TRAIN_BINARY:-1}"
TRAIN_LANDMARKS="${TRAIN_LANDMARKS:-1}"
RUN_CHECKS="${RUN_CHECKS:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
USE_SYSTEM_PYTHON="${USE_SYSTEM_PYTHON:-0}"
USE_PREINSTALLED_TENSORFLOW="${USE_PREINSTALLED_TENSORFLOW:-0}"
PIP_BIN=""
PYTHON_IN_VENV=""

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/autodl_one_click_train.sh [options]

Options:
  --python PATH                 Python executable to create the venv.
  --venv PATH                   Virtual environment path. Default: .venv-autodl
  --binary-config PATH          Binary training config. Default: configs/binary_thesis_public.yaml
  --landmarks-config PATH       Landmark training config. Default: configs/landmarks_cat_dataset.yaml
  --use-system-python           Use the image's built-in python/pip instead of creating a venv.
  --use-preinstalled-tensorflow Keep the image's built-in TensorFlow and skip installing tensorflow from requirements-ml.txt.
  --skip-install                Skip virtualenv creation and dependency installation.
  --skip-download-binary        Skip TFDS binary dataset download.
  --skip-download-cat-dataset   Skip Kaggle CAT Dataset download.
  --skip-binary-train           Skip binary model training.
  --skip-landmarks-train        Skip landmark model training.
  --skip-checks                 Skip environment and download checks.
  --help                        Show this message.

Environment variables:
  PYTHON_BIN
  VENV_DIR
  BINARY_CONFIG
  LANDMARKS_CONFIG
  DOWNLOAD_BINARY_DATASETS
  DOWNLOAD_CAT_DATASET
  TRAIN_BINARY
  TRAIN_LANDMARKS
  RUN_CHECKS
  INSTALL_DEPS
  USE_SYSTEM_PYTHON
  USE_PREINSTALLED_TENSORFLOW
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --binary-config)
      BINARY_CONFIG="$2"
      shift 2
      ;;
    --landmarks-config)
      LANDMARKS_CONFIG="$2"
      shift 2
      ;;
    --use-system-python)
      USE_SYSTEM_PYTHON=1
      shift
      ;;
    --use-preinstalled-tensorflow)
      USE_PREINSTALLED_TENSORFLOW=1
      shift
      ;;
    --skip-install)
      INSTALL_DEPS=0
      shift
      ;;
    --skip-download-binary)
      DOWNLOAD_BINARY_DATASETS=0
      shift
      ;;
    --skip-download-cat-dataset)
      DOWNLOAD_CAT_DATASET=0
      shift
      ;;
    --skip-binary-train)
      TRAIN_BINARY=0
      shift
      ;;
    --skip-landmarks-train)
      TRAIN_LANDMARKS=0
      shift
      ;;
    --skip-checks)
      RUN_CHECKS=0
      shift
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

log() {
  printf '[autodl] %s\n' "$*"
}

ensure_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

require_supported_python() {
  local version_output
  version_output="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  local major="${version_output%%.*}"
  local minor="${version_output##*.}"
  if (( major < 3 )) || (( major == 3 && minor < 10 )) || (( major == 3 && minor > 12 )) || (( major > 3 )); then
    echo "Current Python is ${version_output}. Full training in this project currently requires Python 3.10-3.12." >&2
    echo "Python 3.8 is too old for the current codebase, and Python 3.13+ is outside the validated TensorFlow/dependency range." >&2
    echo "Please create and activate a conda environment such as: conda create -n catalgo310 python=3.10 -y" >&2
    exit 1
  fi
}

install_python_requirements() {
  "${PIP_BIN}" install -r requirements.txt
  if [[ "${USE_PREINSTALLED_TENSORFLOW}" == "1" ]]; then
    local filtered
    filtered="$(mktemp)"
    grep -v '^tensorflow==' requirements-ml.txt > "${filtered}"
    "${PIP_BIN}" install -r "${filtered}"
    rm -f "${filtered}"
  else
    "${PIP_BIN}" install -r requirements-ml.txt
  fi
}

setup_venv() {
  ensure_command "${PYTHON_BIN}"
  require_supported_python
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  PIP_BIN="${VENV_DIR}/bin/pip"
  PYTHON_IN_VENV="${VENV_DIR}/bin/python"

  log "Upgrading pip"
  "${PIP_BIN}" install --upgrade pip
  log "Installing project dependencies"
  install_python_requirements
}

activate_existing_venv() {
  PIP_BIN="${VENV_DIR}/bin/pip"
  PYTHON_IN_VENV="${VENV_DIR}/bin/python"
  if [[ ! -x "${PYTHON_IN_VENV}" ]]; then
    echo "Virtual environment not found at ${VENV_DIR}. Remove --skip-install or provide a valid --venv." >&2
    exit 1
  fi
}

setup_system_python() {
  ensure_command "${PYTHON_BIN}"
  require_supported_python
  PYTHON_IN_VENV="${PYTHON_BIN}"
  if command -v pip3 >/dev/null 2>&1; then
    PIP_BIN="pip3"
  elif command -v pip >/dev/null 2>&1; then
    PIP_BIN="pip"
  else
    echo "pip/pip3 not found in current image." >&2
    exit 1
  fi

  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    log "Installing dependencies into image python environment"
    "${PIP_BIN}" install --upgrade pip
    install_python_requirements
  fi
}

run_checks() {
  log "Running environment check"
  PYTHONPATH=src "${PYTHON_IN_VENV}" scripts/check_env.py
}

download_binary_datasets() {
  log "Downloading TFDS binary datasets"
  PYTHONPATH=src "${PYTHON_IN_VENV}" scripts/download_public_datasets.py \
    --datasets cats_vs_dogs oxford_iiit_pet celeb_a caltech_birds2011
}

download_cat_dataset() {
  if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
    echo "Missing ${HOME}/.kaggle/kaggle.json. Upload Kaggle credentials or pass --skip-download-cat-dataset." >&2
    exit 1
  fi
  chmod 600 "${HOME}/.kaggle/kaggle.json"
  log "Downloading Kaggle CAT Dataset"
  PYTHONPATH=src "${PYTHON_IN_VENV}" scripts/download_public_datasets.py --datasets cat_dataset
}

check_downloads() {
  log "Checking dataset status"
  PYTHONPATH=src "${PYTHON_IN_VENV}" scripts/check_downloads.py
}

train_binary_model() {
  log "Training binary model with ${BINARY_CONFIG}"
  PYTHONPATH=src "${PYTHON_IN_VENV}" scripts/train_binary.py --config "${BINARY_CONFIG}"
}

train_landmark_model() {
  log "Training landmark model with ${LANDMARKS_CONFIG}"
  PYTHONPATH=src "${PYTHON_IN_VENV}" scripts/train_landmarks.py --config "${LANDMARKS_CONFIG}"
}

main() {
  mkdir -p artifacts/autodl
  if [[ "${USE_SYSTEM_PYTHON}" == "1" ]]; then
    setup_system_python
  elif [[ "${INSTALL_DEPS}" == "1" ]]; then
    setup_venv
  else
    activate_existing_venv
  fi

  if [[ "${RUN_CHECKS}" == "1" ]]; then
    run_checks
  fi

  if [[ "${DOWNLOAD_BINARY_DATASETS}" == "1" ]]; then
    download_binary_datasets
  fi

  if [[ "${DOWNLOAD_CAT_DATASET}" == "1" ]]; then
    download_cat_dataset
  fi

  if [[ "${RUN_CHECKS}" == "1" ]]; then
    check_downloads
  fi

  if [[ "${TRAIN_BINARY}" == "1" ]]; then
    train_binary_model
  fi

  if [[ "${TRAIN_LANDMARKS}" == "1" ]]; then
    train_landmark_model
  fi

  log "AutoDL one-click training flow completed."
}

main
