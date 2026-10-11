# Data

The project uses two external datasets and does not redistribute either dataset.

## NPR Interview 2P Corpus

The training, validation, and in-domain test examples are derived from the NPR Interview 2P corpus distributed through ConvoKit. The notebook downloads the corpus with `convokit.download("npr-2p-corpus")`.

The original workflow indexes episodes, filters technology-related titles, merges consecutive turns by speaker role, extracts host–guest–host sequences, and applies word-count and question-count filters. The final 551 examples are split at the episode level to prevent turns from the same interview appearing in more than one split.

## Anthropic Interviewer

The `Anthropic/AnthropicInterviewer` dataset is used only for cross-source evaluation. It is loaded through Hugging Face Datasets and is not used for training.

## Local Data Policy

Downloaded corpora, processed CSV files, and caches are ignored by Git. Run the notebook to acquire and prepare the data in the active environment. Review the source datasets' terms before redistributing any derived text.
