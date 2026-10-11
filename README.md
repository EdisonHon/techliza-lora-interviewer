# TechLIZA

TechLIZA is a technology media interviewer built by adapting `Qwen/Qwen2.5-0.5B-Instruct` with LoRA supervised fine-tuning. Given the previous interviewer question and the interviewee's latest response, it generates one concise follow-up question. The project includes the original data-processing and model-comparison workflow, a reusable Python implementation, the selected LoRA adapter, and a Streamlit interface.

## Project Information

| Field | Details |
|---|---|
| Course | COMS2702: AI in Context |
| Institution | Columbia University |
| Term | Fall 2026 |
| Project Type | Individual Coursework |
| Status | Refactored |
| Primary Language | Python |

## Overview

The project investigates whether fine-tuning a small language model on real technology interviews improves its ability to ask relevant, specific, and natural follow-up questions. NPR interview transcripts provide the training and in-domain evaluation data. The Anthropic Interviewer dataset provides a separate cross-source evaluation set and is not used for training.

The repository preserves the coursework's original methods and reported results while separating reusable code from the experiment notebook. The submitted adapter can be used without repeating model training.

## Objectives

- Construct follow-up-question examples from technology-related NPR interviews.
- Fine-tune Qwen2.5-0.5B-Instruct with LoRA.
- Compare the base model and fine-tuned model on held-out NPR examples.
- Test whether an additional context turn improves question quality.
- Evaluate transfer to interview examples from a different source.
- Provide an interactive Streamlit demonstration of the selected model.

## Methods and Technologies

The NPR corpus is indexed by episode, filtered with technology-related title patterns, and converted into host–guest–host sequences. Consecutive turns from the same role are merged before extraction. Examples are filtered by question count and word-count limits, then split at the episode level to prevent turns from the same interview appearing in different subsets.

The final adapter uses LoRA rank 16, alpha 32, dropout 0.05, and the attention projection modules `q_proj`, `k_proj`, `v_proj`, and `o_proj`. Training used three epochs, a learning rate of `2e-4`, FP16 precision, an effective batch size of 16, a maximum sequence length of 512, and random seed 42.

Main technologies include PyTorch, Transformers, PEFT, TRL, Hugging Face Datasets, ConvoKit, pandas, scikit-learn, and Streamlit.

## Results

The data pipeline retained 551 of 961 candidate NPR examples. The episode-level split contained 436 training, 56 validation, and 59 test examples.

| Evaluation | Sample Size | Result |
|---|---:|---|
| Matched one-turn vs. two-turn NPR comparison | 25 | One-turn 11, two-turn 10, tie 1, neither 3 |
| Base model vs. TechLIZA on Anthropic examples | 10 | TechLIZA 7, base model 3 |

The two-turn model achieved a lower validation loss but no clear human-evaluated advantage. The original one-turn adapter was selected because it used more than twice as many training examples and achieved comparable qualitative performance. The external 7-of-10 result is preliminary because the sample was small and used one human reviewer.

Detailed recorded metrics are available in [`results/metrics/training_metrics.csv`](results/metrics/training_metrics.csv), and the evaluation summary is in [`results/tables/evaluation_summary.csv`](results/tables/evaluation_summary.csv).

## Repository Structure

```text
.
├── app.py                         # Streamlit interview interface
├── data/README.md                 # Dataset sources and handling notes
├── models/techliza_lora_final/    # Selected adapter and model card
├── notebooks/techliza_experiment.ipynb
├── reports/techliza_report.pdf
├── results/                       # Recorded metrics and evaluation summary
├── src/techliza/                  # Reusable project logic
├── tests/                         # Data and evaluation tests
├── requirements.txt               # Application dependencies
└── requirements-notebook.txt      # Full notebook dependencies
```

## Setup

Python 3.11 and a CUDA-capable GPU are recommended. The application can fall back to CPU inference, but generation will be slower. The base Qwen model is downloaded from Hugging Face the first time the application runs.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the complete notebook workflow:

```powershell
python -m pip install -r requirements-notebook.txt
```

## Usage

Run the submitted adapter in the Streamlit interface:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

The notebook preserves the original development, training, comparison, and evaluation sequence. Expensive sections are controlled by configuration flags and are disabled by default because the submitted adapter is already included. Start Jupyter from the repository root and open `notebooks/techliza_experiment.ipynb`:

```powershell
python -m jupyter lab
```

The final notebook section can launch the same repository `app.py`; it does not generate a second copy of the application.

## Data

The project does not commit raw or processed interview text. The notebook downloads the NPR Interview 2P corpus through ConvoKit and loads `Anthropic/AnthropicInterviewer` through Hugging Face Datasets when the corresponding experiment flags are enabled. See [`data/README.md`](data/README.md) for the preparation workflow.

## Limitations

- The training data is small and comes from one media source.
- The human evaluations use small samples and one reviewer.
- Some evaluation examples from the same external transcript may be correlated.
- TechLIZA can still produce generic questions, repetition, unnecessary clarification, or topic drift.
- Exact training reproduction can depend on GPU hardware and library versions.
- The Streamlit interface is a coursework demonstration, not a production service.

## Report

The submitted project report is included at [`reports/techliza_report.pdf`](reports/techliza_report.pdf). It documents the experimental process, results, challenges, limitations, and final model selection.

## References

- Qwen Team. *Qwen2.5-0.5B-Instruct*. Hugging Face model repository.
- Cornell ConvoKit. *NPR Interview 2P Corpus*.
- Anthropic. *Anthropic Interviewer*. Hugging Face dataset repository.
- Hu, E. J., et al. “LoRA: Low-Rank Adaptation of Large Language Models.” 2021. <https://arxiv.org/abs/2106.09685>
