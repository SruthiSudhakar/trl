#!/usr/bin/env bash
set -euo pipefail  # exit on error, undefined var, or failed pipe

# --- Check for Python3 ---
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 is not installed. Please install Python3 and rerun this script."
    exit 1
fi

# --- Ensure pip is installed ---
if ! command -v pip3 &> /dev/null
then
    echo "📦 Installing python3-pip and venv..."
    apt update && apt install -y python3-pip python3-venv
fi

# --- Upgrade pip, setuptools, wheel ---
python3 -m pip install --upgrade pip setuptools wheel

# --- Enable HuggingFace Hub online mode ---
export HF_HUB_OFFLINE=0

# --- Install dependencies ---
pip3 install -e .[dev]
pip3 install \
    peft \
    deepspeed \
    math-verify \
    latex2sympy2_extended \
    vllm \
    wandb \
    qwen_vl_utils \
    matplotlib

echo "✅ Installation complete!"

# --- Hugging Face authentication ---
echo "🔑 Logging into Hugging Face..."
huggingface-cli login --token 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 --add-to-git-credential

# --- Weights & Biases authentication ---
echo "📊 Logging into Weights & Biases..."
wandb login YOUR_WANDB_TOKEN

echo "✅ Installation and authentication complete!"
