"""Model loading and response generation for the LLM project (Assignment 1B).

This is the main implementation file. It loads an open-weight instruct model
(``google/gemma-2b-it`` by default) and answers questions with it.

Hardware-adaptive design
------------------------
Quantization lets a 5 GB model run in ~1.7 GB of VRAM, but the 4-bit
``bitsandbytes`` backend only exists for NVIDIA CUDA. This module therefore
picks the best strategy for the machine it runs on:

- **CUDA GPU**      -> 4-bit NF4 quantization + ``device_map="auto"``
- **Apple Silicon** -> half-precision (float16) weights on the MPS backend
                       (bitsandbytes has no macOS build)
- **CPU only**      -> full-precision weights as a slow fallback

The model id can be overridden with the ``LLM_MODEL_ID`` environment variable
so the same code works with any other open-weight model, e.g.::

    LLM_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct python demo.py
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#: Default model id. The assignment suggests google/gemma-2b-it; the
#: environment variable makes the choice explicit and testable.
DEFAULT_MODEL_ID = os.environ.get("LLM_MODEL_ID", "google/gemma-2b-it")


def _pick_device():
    """Return the fastest available device: ``cuda`` > ``mps`` > ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_pad_token_id(tokenizer):
    """Return a usable pad token id (some checkpoints do not define one)."""
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id


def load_model(model_id=DEFAULT_MODEL_ID):
    """Load the tokenizer and the quantized causal language model.

    Returns
    -------
    tuple[PreTrainedModel, PreTrainedTokenizer]
        ``(model, tokenizer)`` ready for :func:`get_response`.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    device = _pick_device()
    if device == "cuda":
        # 4-bit NF4 quantization: weights drop from ~5 GB (fp16) to ~1.7 GB,
        # which is what makes the model fit on an 8 GB consumer GPU.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        # Apple Silicon: no bitsandbytes support, so load fp16 weights
        # directly and move them to the Metal (MPS) backend.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model = model.to("mps")
    else:
        # CPU-only fallback: fp32 keeps every CPU op available; slow but works.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, tokenizer


def get_response(prompt, model, tokenizer, max_length=150, temperature=0.7, seed=None):
    """Generate the assistant's reply for ``prompt``.

    The prompt is wrapped with the model's *chat template* so the special role
    tokens the model was instruction-tuned with are present, then we sample
    ``max_length`` new tokens and cut everything that belongs to the prompt
    itself.

    Parameters
    ----------
    prompt : str
        The user's question.
    model, tokenizer
        The pair returned by :func:`load_model`.
    max_length : int
        Maximum number of *new* tokens to generate.
    temperature : float
        Sampling temperature; lower = more deterministic.
    seed : int or None
        Optional random seed for reproducible replies.
    """
    if seed is not None:
        torch.manual_seed(seed)

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=_resolve_pad_token_id(tokenizer),
        )

    # Decode only the freshly generated tokens: this keeps the user prompt
    # out of the reply without fragile string splitting.
    reply = tokenizer.decode(
        outputs[0][prompt_length:], skip_special_tokens=True
    ).strip()
    return reply
