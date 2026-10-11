"""Model loading and deterministic question generation for TechLIZA."""

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import INTERACTIVE_SYSTEM_PROMPT

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_techliza_model(
    adapter_path: str | Path,
    base_model_name: str = BASE_MODEL_NAME,
) -> tuple[Any, Any]:
    """Load the Qwen tokenizer, base model, and submitted LoRA adapter."""
    adapter_path = Path(adapter_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def build_interview_messages(
    main_topic: str,
    previous_question: str,
    interviewee_response: str,
) -> list[dict[str, str]]:
    """Build the prompt format used by the original Streamlit application."""
    return [
        {"role": "system", "content": INTERACTIVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Main interview topic:\n{main_topic}\n\n"
                f"Previous interviewer question:\n{previous_question}\n\n"
                f"Interviewee response:\n{interviewee_response}"
            ),
        },
    ]


def generate_with_model(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int = 80,
) -> str:
    """Generate a deterministic response from a chat-formatted model."""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def generate_question(
    tokenizer: Any,
    model: Any,
    main_topic: str,
    previous_question: str,
    interviewee_response: str,
    max_new_tokens: int = 80,
) -> str:
    """Generate the next TechLIZA interview question."""
    messages = build_interview_messages(
        main_topic,
        previous_question,
        interviewee_response,
    )
    return generate_with_model(tokenizer, model, messages, max_new_tokens)
