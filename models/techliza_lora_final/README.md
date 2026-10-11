# TechLIZA LoRA Adapter

## Model Description

TechLIZA is a LoRA adapter for `Qwen/Qwen2.5-0.5B-Instruct`. It was trained to generate one concise technology-interview follow-up question from the previous interviewer question and the interviewee's latest response.

This adapter is coursework from COMS2702: AI in Context at Columbia University (Fall 2026). The repository is maintained as a private record of the project.

## Training Data

The adapter was trained on 436 examples derived from technology-related episodes in the NPR Interview 2P corpus. Validation and test data contained 56 and 59 examples. Splitting was performed at the episode level.

The source data is not included with the adapter.

## Training Procedure

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Method: supervised fine-tuning with LoRA
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Epochs: 3
- Learning rate: `2e-4`
- Maximum sequence length: 512
- Training precision: FP16
- Random seed: 42

The adapter contains 2,162,688 trainable parameters, approximately 0.44% of the combined model parameters reported during training.

## Evaluation Summary

The original coursework compared the base model and TechLIZA on held-out NPR examples and on the Anthropic Interviewer dataset. In a 10-case blinded cross-source comparison, TechLIZA was preferred in 7 cases and the base model in 3. This result is exploratory because the sample was small and used one human reviewer.

A controlled one-turn versus two-turn context experiment found no clear human-evaluated advantage for the two-turn model. The larger one-turn adapter was therefore selected as the final model.

## Limitations

- The training set is small and comes from one media source.
- Evaluation samples are small and were judged by one reviewer.
- The model can produce generic, repetitive, overly clarifying, or off-topic questions.
- The adapter inherits limitations and risks from the Qwen base model.
- The project is a classroom experiment, not a production interviewing system.

## Files

- `adapter_config.json`: PEFT adapter configuration
- `adapter_model.safetensors`: trained LoRA weights

Load the public base model tokenizer separately; a tokenizer copy is intentionally not duplicated in this repository.
