import pandas as pd

from src.techliza.evaluation import (
    create_blind_options,
    parse_anthropic_transcript,
    reveal_winners,
)


def test_parse_anthropic_transcript_maps_roles():
    transcript = "Assistant: First question?\nUser: First answer.\nAI: Follow-up?"

    assert parse_anthropic_transcript(transcript) == [
        {"role": "interviewer", "text": "First question?"},
        {"role": "interviewee", "text": "First answer."},
        {"role": "interviewer", "text": "Follow-up?"},
    ]


def test_blind_mapping_can_be_revealed():
    frame = pd.DataFrame(
        [
            {
                "case_id": 1,
                "base_output": "Base question?",
                "techliza_output": "TechLIZA question?",
            }
        ]
    )
    blind, mapping = create_blind_options(
        frame,
        "base_output",
        "techliza_output",
        "Base",
        "TechLIZA",
    )
    blind["winner"] = "A"

    revealed = reveal_winners(blind, mapping)

    assert revealed.loc[0, "actual_winner"] == mapping.loc[0, "option_A_model"]
