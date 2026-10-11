"""Training data and configuration helpers matching the original notebook."""

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig

from .prompts import TRAINING_SYSTEM_PROMPT


def build_prompt_completion(example: dict, include_earlier_turn: bool = False) -> dict:
    """Convert one row to the prompt/completion structure used for SFT."""
    context_parts = []
    if include_earlier_turn:
        context_parts.extend(
            [
                f"Earlier interviewer question:\n{example['earlier_question']}",
                f"Earlier interviewee response:\n{example['earlier_answer']}",
            ]
        )
    context_parts.extend(
        [
            f"Previous interviewer question:\n{example['previous_question']}",
            f"Interviewee response:\n{example['interviewee_answer']}",
        ]
    )
    return {
        "prompt": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(context_parts)},
        ],
        "completion": [{"role": "assistant", "content": example["next_question"]}],
    }


def dataframe_to_sft_dataset(frame, include_earlier_turn: bool = False) -> Dataset:
    """Convert a pandas frame to the original prompt/completion dataset."""
    dataset = Dataset.from_pandas(frame.reset_index(drop=True))
    return dataset.map(
        lambda example: build_prompt_completion(example, include_earlier_turn)
    )


def build_lora_config() -> LoraConfig:
    """Return the LoRA configuration used in all TechLIZA runs."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_sft_config(output_dir: str) -> SFTConfig:
    """Return the supervised fine-tuning configuration from the coursework."""
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        max_length=512,
        fp16=True,
        bf16=False,
        gradient_checkpointing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        seed=42,
        report_to="none",
    )
