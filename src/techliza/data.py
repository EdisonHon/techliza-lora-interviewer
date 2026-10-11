"""Data preparation helpers for the NPR interview workflow."""

import re
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

HIGH_PRECISION_PATTERNS = [
    r"\bartificial intelligence\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bAI\b",
    r"\bA\.I\.\b",
    r"\balgorithms?\b",
    r"\bsoftware\b",
    r"\bcomputers?\b",
    r"\bcomputing\b",
    r"\binternet\b",
    r"\bcyber(?:security|attack|attacks|warfare|crime)?\b",
    r"\bhackers?\b",
    r"\bhacking\b",
    r"\brobots?\b",
    r"\brobotics\b",
    r"\bsocial media\b",
    r"\bsmartphones?\b",
    r"\bmobile apps?\b",
    r"\bsilicon valley\b",
    r"\bbig tech\b",
    r"\btech companies\b",
    r"\btech company\b",
    r"\btechnology companies\b",
    r"\btechnology company\b",
    r"\bsemiconductors?\b",
    r"\bmicrochips?\b",
    r"\bencryption\b",
    r"\bdigital privacy\b",
    r"\bonline privacy\b",
]

EXCLUSION_PATTERNS = [
    r"\bcyber monday\b",
    r"\bmemes?\b",
    r"\binternet's attention\b",
    r"\bcaptures internet\b",
    r"\bweek in internet\b",
    r"\bshopping\b",
    r"\bdeals\b",
    r"\bsales records\b",
]


def group_utterances_by_episode(corpus):
    """Group ConvoKit utterances by NPR episode ID."""
    episodes = defaultdict(list)
    for utterance in corpus.iter_utterances():
        episode_id = utterance.meta.get("episode")
        if episode_id is not None:
            episodes[episode_id].append(utterance)
    return episodes


def build_episode_index(corpus, conversations_data: dict) -> tuple[pd.DataFrame, dict]:
    """Build the episode-level metadata table used by the original project."""
    episodes = group_utterances_by_episode(corpus)
    records = []

    for episode_id, utterances in episodes.items():
        utterances = sorted(utterances, key=lambda item: item.meta.get("order", 0))
        first_utterance = utterances[0]
        metadata_record = conversations_data.get(first_utterance.conversation_id, {})
        metadata = metadata_record.get("meta", {})

        hosts = sorted(
            {
                utterance.speaker.meta.get("name")
                for utterance in utterances
                if utterance.speaker.meta.get("type") == "host"
                and utterance.speaker.meta.get("name")
            }
        )
        guests = sorted(
            {
                utterance.speaker.meta.get("name")
                for utterance in utterances
                if utterance.speaker.meta.get("type") == "guest"
                and utterance.speaker.meta.get("name")
            }
        )

        records.append(
            {
                "episode_id": episode_id,
                "program": metadata.get("program"),
                "title": metadata.get("title"),
                "date": metadata.get("date"),
                "num_utterances": len(utterances),
                "host": "; ".join(hosts),
                "guest": "; ".join(guests),
                "opening_text": first_utterance.text,
            }
        )

    return pd.DataFrame(records), episodes


def filter_technology_episodes(episode_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the original high-precision title filter and exclusions."""
    include = re.compile("|".join(HIGH_PRECISION_PATTERNS), re.IGNORECASE)
    exclude = re.compile("|".join(EXCLUSION_PATTERNS), re.IGNORECASE)
    titles = episode_df["title"].fillna("")
    filtered = episode_df[titles.str.contains(include, na=False)].copy()
    filtered = filtered[
        ~filtered["title"].fillna("").str.contains(exclude, na=False)
    ].copy()
    return filtered[filtered["num_utterances"] >= 8].copy()


def merge_role_turns(utterances) -> list[dict[str, str]]:
    """Merge consecutive host or guest utterances into speaker-role turns."""
    merged = []
    for utterance in sorted(utterances, key=lambda item: item.meta.get("order", 0)):
        role = utterance.speaker.meta.get("type")
        text = utterance.text.strip()
        if role not in {"host", "guest"} or not text:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["text"] += " " + text
        else:
            merged.append({"role": role, "text": text})
    return merged


def extract_follow_up_examples(
    episode_ids,
    episodes: dict,
) -> pd.DataFrame:
    """Extract host-question, guest-answer, host-question triples."""
    records = []
    for episode_id in episode_ids:
        turns = merge_role_turns(episodes[episode_id])
        for index in range(len(turns) - 2):
            previous_turn, guest_turn, next_turn = turns[index : index + 3]
            if (
                previous_turn["role"] == "host"
                and guest_turn["role"] == "guest"
                and next_turn["role"] == "host"
                and "?" in next_turn["text"]
            ):
                records.append(
                    {
                        "episode_id": episode_id,
                        "previous_question": previous_turn["text"],
                        "interviewee_answer": guest_turn["text"],
                        "next_question": next_turn["text"],
                    }
                )
    return pd.DataFrame(records)


def clean_follow_up_examples(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the original word-count and one-question cleaning rules."""
    frame = candidate_df.copy()
    frame["previous_question_words"] = frame["previous_question"].str.split().str.len()
    frame["interviewee_answer_words"] = frame["interviewee_answer"].str.split().str.len()
    frame["next_question_words"] = frame["next_question"].str.split().str.len()
    frame["previous_question_marks"] = frame["previous_question"].str.count(r"\?")
    frame["next_question_marks"] = frame["next_question"].str.count(r"\?")

    return frame[
        (frame["next_question_marks"] == 1)
        & frame["previous_question_words"].between(3, 100)
        & frame["interviewee_answer_words"].between(5, 200)
        & frame["next_question_words"].between(5, 60)
    ].copy()


def split_by_episode(
    clean_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the original 80/10/10 episode-level split."""
    episode_ids = clean_df["episode_id"].unique()
    train_ids, temporary_ids = train_test_split(
        episode_ids,
        test_size=0.20,
        random_state=random_state,
    )
    validation_ids, test_ids = train_test_split(
        temporary_ids,
        test_size=0.50,
        random_state=random_state,
    )

    train = clean_df[clean_df["episode_id"].isin(train_ids)].copy()
    validation = clean_df[clean_df["episode_id"].isin(validation_ids)].copy()
    test = clean_df[clean_df["episode_id"].isin(test_ids)].copy()
    return train, validation, test


def normalize_text(text: str) -> str:
    """Normalize transcript spacing for deterministic example matching."""
    return " ".join(str(text).split()).strip()


def extract_two_turn_candidates(
    episode_ids,
    episodes: dict,
) -> pd.DataFrame:
    """Extract the five-turn patterns used by the context experiment."""
    records = []
    for episode_id in episode_ids:
        utterances = sorted(
            episodes[episode_id],
            key=lambda item: item.meta.get("order", 0),
        )
        merged = []
        for utterance in utterances:
            speaker = utterance.speaker.id
            text = utterance.text.strip()
            if not text:
                continue
            if merged and merged[-1]["speaker"] == speaker:
                merged[-1]["text"] += " " + text
            else:
                merged.append({"speaker": speaker, "text": text})

        for index in range(len(merged) - 4):
            first, second, third, fourth, fifth = merged[index : index + 5]
            if (
                first["speaker"] == third["speaker"] == fifth["speaker"]
                and second["speaker"] == fourth["speaker"]
                and first["speaker"] != second["speaker"]
                and all("?" in turn["text"] for turn in (first, third, fifth))
            ):
                records.append(
                    {
                        "episode_id": episode_id,
                        "earlier_question": first["text"],
                        "earlier_answer": second["text"],
                        "previous_question": third["text"],
                        "interviewee_answer": fourth["text"],
                        "next_question": fifth["text"],
                    }
                )
    return pd.DataFrame(records)


def match_two_turn_context(
    clean_df: pd.DataFrame,
    two_turn_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach an earlier turn to examples retained by the original cleaning rules."""
    key_columns = ["previous_question", "interviewee_answer", "next_question"]
    clean = clean_df.copy()
    candidates = two_turn_df.copy()
    normalized_keys = []

    for column in key_columns:
        normalized = f"{column}_norm"
        normalized_keys.append(normalized)
        clean[normalized] = clean[column].apply(normalize_text)
        candidates[normalized] = candidates[column].apply(normalize_text)

    matched = clean.merge(
        candidates[
            ["episode_id", *normalized_keys, "earlier_question", "earlier_answer"]
        ],
        on=["episode_id", *normalized_keys],
        how="left",
    )
    return matched[matched["earlier_question"].notna()].copy()
