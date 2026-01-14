# Reasoning with Sampling  Setup & Mini-Experiments (MATH / HumanEval)

### [Paper](https://arxiv.org/abs/2510.14901) | [Project Page](https://aakaran.github.io/reasoning_with_sampling/)

[![rws](teaser.png)](teaser.png)

This repo allows you to evaluate Base sampling / Low-temperature sampling / Power Sampling (MCMC) on multiple benchmarks (MATH, HumanEval).

---

## Prerequisites

- Linux (Ubuntu/Onyxia)
- GPU available + CUDA drivers (optional but recommended)
- git installed
- Internet connection (to download HuggingFace models + datasets)

**Check GPU availability:**
``
nvidia-smi
``

---

## 1) Install Miniconda (if conda doesn't exist)

If conda --version returns "command not found", install Miniconda:

**Download Miniconda (Linux x86_64):**
``
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
``

**Install:**
``
bash Miniconda3-latest-Linux-x86_64.sh
``

Reload your shell (or close/reopen terminal), then verify:
``
conda --version
``

---

## 2) Accept Terms of Service (if ToS error)

If you see the error:
``
CondaToSNonInteractiveError: Terms of Service have not been accepted
``

Execute:
``
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
``

---

## 3) Create and Activate Conda Environment

Create a Python 3.11 environment (recommended as some packages break in Python 3.13):

``
conda create -n reasoning-sampling python=3.11 -y
conda activate reasoning-sampling
python --version
``

You should see Python 3.11.x.

---

## 4) Install Dependencies (pip within conda environment)

Execute from the repo root:

``
pip install --upgrade pip
pip install torch transformers accelerate datasets tqdm sentencepiece
``

**Notes:**
- If using GPU, ensure 	orch is CUDA-compatible (usually already OK on GPU platforms)

---
