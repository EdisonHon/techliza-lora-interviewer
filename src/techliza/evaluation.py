"""Evaluation helpers used by the TechLIZA experiments."""

import random
import re

import pandas as pd


def parse_anthropic_transcript(text: str) -> list[dict[str, str]]:
    """Parse Anthropic Interviewer text into interviewer/interviewee turns."""
    pattern = r"(?m)^(Assistant|AI|User):\s*"
    matches = list(re.finditer(pattern, text))
    turns = []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        role = "interviewer" if match.group(1) in {"Assistant", "AI"} else "interviewee"
        turns.append({"role": role, "text": text[start:end].strip()})
    return turns


def build_anthropic_examples(dataset, split_names) -> pd.DataFrame:
    """Build follow-up examples with the same format as the NPR data."""
    records = []
    for split_name in split_names:
        for record in dataset[split_name]:
            turns = parse_anthropic_transcript(record["text"])
            for index in range(len(turns) - 2):
                first, second, third = turns[index : index + 3]
                if (
                    first["role"] == "interviewer"
                    and second["role"] == "interviewee"
                    and third["role"] == "interviewer"
                    and "?" in first["text"]
                    and "?" in third["text"]
                ):
                    records.append(
                        {
                            "split": split_name,
                            "transcript_id": record["transcript_id"],
                            "previous_question": first["text"],
                            "interviewee_answer": second["text"],
                            "next_question": third["text"],
                        }
                    )
    return pd.DataFrame(records)


def remove_interview_boundaries(examples: pd.DataFrame) -> pd.DataFrame:
    """Remove the first onboarding and final closing example per transcript."""
    groups = []
    for _, group in examples.groupby(["split", "transcript_id"], sort=False):
        if len(group) > 2:
            groups.append(group.iloc[1:-1])
    if not groups:
        return examples.iloc[0:0].copy()
    return pd.concat(groups, ignore_index=True)


def create_blind_options(
    frame: pd.DataFrame,
    first_column: str,
    second_column: str,
    first_label: str,
    second_label: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly assign two model outputs to options A and B."""
    generator = random.Random(random_state)
    blind_rows = []
    mapping_rows = []

    for _, row in frame.iterrows():
        if generator.random() < 0.5:
            option_a, option_b = row[first_column], row[second_column]
            label_a, label_b = first_label, second_label
        else:
            option_a, option_b = row[second_column], row[first_column]
            label_a, label_b = second_label, first_label

        common = {
            column: row[column]
            for column in frame.columns
            if column not in {first_column, second_column}
        }
        blind_rows.append({**common, "option_A": option_a, "option_B": option_b})
        mapping_rows.append(
            {
                "case_id": row["case_id"],
                "option_A_model": label_a,
                "option_B_model": label_b,
            }
        )

    return pd.DataFrame(blind_rows), pd.DataFrame(mapping_rows)


def reveal_winners(judgments: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Map blinded A/B judgments back to the model labels."""
    revealed = judgments.merge(mapping, on="case_id")

    def reveal(row):
        if row["winner"] == "A":
            return row["option_A_model"]
        if row["winner"] == "B":
            return row["option_B_model"]
        if row["winner"] in {"Tie", "Neither"}:
            return row["winner"]
        return "Unknown"

    revealed["actual_winner"] = revealed.apply(reveal, axis=1)
    return revealed
