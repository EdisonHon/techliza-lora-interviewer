"""Streamlit interface for the submitted TechLIZA LoRA adapter."""

from pathlib import Path

import streamlit as st

from src.techliza.model import generate_question, load_techliza_model
from src.techliza.prompts import OPENING_QUESTION

REPOSITORY_ROOT = Path(__file__).resolve().parent
ADAPTER_PATH = REPOSITORY_ROOT / "models" / "techliza_lora_final"

st.set_page_config(
    page_title="TechLIZA",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_cached_model():
    """Load and cache the base model, tokenizer, and TechLIZA adapter."""
    return load_techliza_model(ADAPTER_PATH)


try:
    tokenizer, techliza_model = load_cached_model()
except Exception as error:  # noqa: BLE001 - surface model-loading failures in the UI
    st.error(f"TechLIZA could not load its model: {error}")
    st.stop()

for key, default in {
    "messages": [],
    "main_topic": None,
    "previous_question": OPENING_QUESTION,
    "interview_started": False,
    "interview_ended": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🎙️ TechLIZA")
st.caption("AI Technology Media Interviewer")

with st.sidebar:
    st.header("TechLIZA")

    if st.button("New Interview", use_container_width=True):
        st.session_state.messages = []
        st.session_state.main_topic = None
        st.session_state.previous_question = OPENING_QUESTION
        st.session_state.interview_started = False
        st.session_state.interview_ended = False
        st.rerun()

    st.divider()
    st.subheader("Current Topic")
    if st.session_state.main_topic:
        st.write(st.session_state.main_topic)
    else:
        st.caption("No topic selected yet.")

    st.divider()
    st.subheader("About TechLIZA")
    st.markdown(
        """
        **Model:** Qwen2.5-0.5B-Instruct

        **Fine-tuning:** LoRA

        **Training Data:** NPR Technology Interviews

        **Role:** Technology Media Interviewer
        """
    )

    st.divider()
    if st.button(
        "End Interview",
        use_container_width=True,
        disabled=st.session_state.interview_ended,
    ):
        st.session_state.interview_ended = True
        st.rerun()

if not st.session_state.messages:
    st.session_state.messages.append(
        {"role": "assistant", "content": OPENING_QUESTION}
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if not st.session_state.interview_ended:
    user_input = st.chat_input("Your answer...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        if not st.session_state.interview_started:
            st.session_state.main_topic = user_input
            st.session_state.interview_started = True

        with st.spinner("TechLIZA is thinking..."):
            next_question = generate_question(
                tokenizer=tokenizer,
                model=techliza_model,
                main_topic=st.session_state.main_topic,
                previous_question=st.session_state.previous_question,
                interviewee_response=user_input,
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": next_question}
        )
        st.session_state.previous_question = next_question
        st.rerun()
else:
    st.success("Thank you for the interview.")
