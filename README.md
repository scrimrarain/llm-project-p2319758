# LLM Project — Qwen2.5-1.5B Quantized Chatbot

An end-to-end **Large Language Model (LLM) application** built for
**AI — Assignment 1B**. It loads the open-weight instruction-tuned model
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
(1.5 B parameters, Apache-2.0 — **no login or license required**), applies
**4-bit quantization** where the hardware supports it, and exposes a clean
`load_model()` / `get_response()` interface plus an interactive demo and a
unit-test suite. Any other open-weight model (e.g. the assignment's example
`google/gemma-2b-it`) can be swapped in with one environment variable.

> **Group members:** `p2319758` + `p2303357` — Final submission: **Nov 12**.

---

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Quantization & hardware notes](#quantization--hardware-notes)
- [Testing](#testing)
- [Results](#results)
- [Videos](#videos)
- [Submission checklist](#submission-checklist)

---

## Features

- Loads **Qwen2.5-1.5B (instruction-tuned, open weights)** through HuggingFace
  Transformers — downloadable without a HuggingFace account.
- **Hardware-adaptive loading** in `src/model.py`:
  - NVIDIA GPU → **4-bit NF4 quantization** (`bitsandbytes`) with
    `device_map="auto"` — the weights shrink from ~5 GB to ~1.7 GB.
  - Apple Silicon → **float16 on the MPS (Metal) backend**.
  - CPU only → full-precision fallback.
- Chat-template-aware prompting (`tokenizer.apply_chat_template`), so replies
  stay on-topic and correctly formatted for instruction-tuned models.
- Interactive `demo.py` and 4 unit tests (`unittest`/`pytest`).
- **The application**: a **Language-Learning Assistant** (`app.py`) — a Gradio
  web UI with four tools built on the same model engine (see below).

## Requirements

- Python 3.8+
- Git and a terminal
- Recommended: an NVIDIA GPU with ≥8 GB VRAM (4-bit quantized).
  On Apple Silicon or CPU the same code runs in float16/fp32, just slower.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<YourUsername>/llm-project-<YourName>.git
cd llm-project-<YourName>
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies (one command)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) HuggingFace authentication

`Qwen/Qwen2.5-1.5B-Instruct` downloads without any login. A HuggingFace token
is only needed if you switch to a gated model such as
[google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it):

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Open the model card and click **Accept license**.
3. In *Settings → Access Tokens*, create a token with **read** permission.
4. Log in once from the terminal:

```bash
huggingface-cli login
```

Then switch the model with an environment variable — no code changes:

```bash
export LLM_MODEL_ID=google/gemma-2b-it   # macOS / Linux
# Windows PowerShell: $env:LLM_MODEL_ID = "google/gemma-2b-it"
```

## Usage

Interactive chat loop (asks for input until `quit`):

```bash
python demo.py
```

```text
You: What is the capital of France?
Model: The capital of France is Paris. 🇫🇷 ...
```

Use the API from Python:

```python
from src.model import load_model, get_response

model, tokenizer = load_model()
reply = get_response("Write a haiku about AI.", model, tokenizer, max_length=100)
print(reply)
```

## The application — Language-Learning Assistant

`app.py` is the application layer required by the assignment: a
**language-learning assistant** with a Gradio web interface. The quantized
model is loaded once and shared by four tools, each of which is a specialised
prompt wrapped around the same `get_response()` engine:

```bash
python app.py        # starts http://127.0.0.1:7860 and opens the browser
```

| Tab | Tool | What it does |
|---|---|---|
| 📖 Word Helper | `word_helper(word)` | meaning, part of speech, example sentences, memory tip |
| 🌐 Translator | `translate(text, direction)` | natural translation between English and Chinese |
| 🗣️ Conversation Practice | `conversation_practice(scenario, msg, lang)` | role-plays real scenarios (café, hotel, airport) in the chosen language and ends with a one-line correction of your mistakes |
| ✍️ Writing Coach | `writing_coach(text)` | corrected version of your writing + an explanation of every change |

Every tool also works from Python, e.g.
`from app import word_helper; print(word_helper("serendipity"))`. Sample
outputs for all four tools are recorded in [`RESULTS.md`](RESULTS.md).

## Project structure

```text
llm-project-<YourName>/
├── src/
│   └── model.py          # load_model() / get_response() — main implementation
├── tests/
│   └── test_model.py     # unit tests (factual, code-gen, reasoning, sanity)
├── demo.py               # interactive demo script
├── app.py                # the application: language-learning assistant (Gradio)
├── requirements.txt      # one-command install
├── README.md
├── RESULTS.md            # the five qualitative analysis points
└── videos/               # the three required recordings (see videos/README.md)
```

## Quantization & hardware notes

| Device | Weights | Strategy in `load_model()` |
|---|---|---|
| NVIDIA GPU (CUDA) | 4-bit NF4 (~1.7 GB) | `BitsAndBytesConfig(load_in_4bit=True)` + `device_map="auto"` |
| Apple Silicon (MPS) | float16 (~5 GB) | `torch_dtype=float16` + `.to("mps")` |
| CPU only | float32 (~10 GB) | plain load, slow but functional |

*Why the difference?* `bitsandbytes` 4-bit kernels are CUDA-only, so Apple
machines fall back to half-precision weights on Metal — the smallest format
MPS can run. The code detects the best option at load time, so the same
`model.py` works on a lab GPU machine and on a MacBook.

## Testing

From the project root (with the environment active):

```bash
python -m unittest discover tests
# or
python -m pytest -v
```

| Test | What it verifies |
|---|---|
| `test_factual_query` | "Capital of France" → answer contains **Paris** |
| `test_code_generation` | factorial request → answer contains `def`, `factorial`, and a loop keyword |
| `test_reasoning` | 3 − 1 apple puzzle → answer contains **2** (digit or word) |
| `test_response_is_text` | replies are non-empty strings |

The model is loaded once in `setUpClass` so the suite only pays the load cost
a single time.

## Results

The five qualitative analysis points (factual accuracy, creativity, reasoning,
bias & safety, failure case) with verbatim model output are documented in
[`RESULTS.md`](RESULTS.md).

## Videos

Three short screen recordings demonstrate the whole workflow:

| # | File | Content |
|---|---|---|
| 1 | `p2319758_p2303357_LLM_Chatbot_1-installation.mp4` | Clone repo → venv → `pip install -r requirements.txt` → `huggingface-cli login` |
| 2 | `p2319758_p2303357_LLM_Chatbot_2-implementation.mp4` | Walk through `src/model.py` → `demo.py` three live questions → the language-learning app (`app.py`, Gradio UI + the four tools) |
| 3 | `p2319758_p2303357_LLM_Chatbot_3-testing.mp4` | `python -m unittest discover tests` → `RESULTS.md` five points live |

Recording scripts are in [`videos/README.md`](videos/README.md).

## Submission checklist

- [x] Source code (`src/model.py`, `tests/test_model.py`, `demo.py`)
- [x] `README.md` with setup instructions and video links
- [x] `RESULTS.md` with the five qualitative analysis points
- [x] `requirements.txt` for a one-command install
- [x] Videos (`*.mp4`) recorded and linked in this README
