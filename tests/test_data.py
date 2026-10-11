import pandas as pd

from src.techliza.data import clean_follow_up_examples, split_by_episode


def test_clean_follow_up_examples_applies_original_limits():
    candidates = pd.DataFrame(
        [
            {
                "episode_id": 1,
                "previous_question": "What changed in the system?",
                "interviewee_answer": "The new design reduced latency for our users.",
                "next_question": "How did the lower latency affect adoption?",
            },
            {
                "episode_id": 2,
                "previous_question": "Why?",
                "interviewee_answer": "It improved.",
                "next_question": "What changed?",
            },
        ]
    )

    cleaned = clean_follow_up_examples(candidates)

    assert cleaned["episode_id"].tolist() == [1]


def test_episode_split_has_no_episode_overlap():
    frame = pd.DataFrame(
        {
            "episode_id": [episode for episode in range(20) for _ in range(2)],
            "value": range(40),
        }
    )

    train, validation, test = split_by_episode(frame)
    train_ids = set(train["episode_id"])
    validation_ids = set(validation["episode_id"])
    test_ids = set(test["episode_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(range(20))
