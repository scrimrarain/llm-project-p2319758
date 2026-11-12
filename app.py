"""Language-Learning Assistant — the application layer of the LLM project.

Assignment 1B asks for a conversational application built on top of the
quantized model. This module provides one: four language-learning tools, each
of which is a carefully designed prompt wrapped around the same
``src.model.get_response`` engine, plus a Gradio web interface with one tab
per tool.

Tools
-----
1. Word Helper      — explain a word: meaning, part of speech, examples, tip
2. Translator       — translate between English and Chinese
3. Conversation     — role-play a real scenario (e.g. ordering at a café) and
                      get a one-line correction of your language mistakes
4. Writing Coach    — fix grammar/word choice and explain each change

Run the web UI with::

    python app.py

The model is loaded once at import time and shared by all four tools.
"""

import gradio as gr

from src.model import get_response, load_model

#: One model instance shared by every tool.
MODEL, TOKENIZER = load_model()

#: Seed fixed so the demo in the submission video is reproducible.
SEED = 42

#: Common system instruction for all tools.
ROLE = (
    "You are a friendly, patient language-learning assistant. "
    "Be clear and keep answers under 150 words."
)


def _ask(prompt, max_length=250):
    """Run one tool's prompt through the shared model."""
    return get_response(prompt, MODEL, TOKENIZER, max_length=max_length, seed=SEED)


# --------------------------------------------------------------------------- #
# Tool 1: Word Helper
# --------------------------------------------------------------------------- #
def word_helper(word):
    """Explain a word for a language learner."""
    prompt = (
        f"{ROLE}\n\n"
        f"Explain the word '{word}' for an English learner. Format as:\n"
        f"1. Meaning\n2. Part of speech\n"
        f"3. Two example sentences\n4. A short memory tip"
    )
    return _ask(prompt)


# --------------------------------------------------------------------------- #
# Tool 2: Translator
# --------------------------------------------------------------------------- #
def translate(text, direction):
    """Translate *text*; *direction* is 'EN -> 中文' or '中文 -> EN'."""
    if direction.startswith("EN"):
        prompt = f"{ROLE}\n\nTranslate this text from English to natural Chinese:\n{text}"
    else:
        prompt = f"{ROLE}\n\nTranslate this text from Chinese to natural English:\n{text}"
    return _ask(prompt)


# --------------------------------------------------------------------------- #
# Tool 3: Conversation Practice (role-play + correction)
# --------------------------------------------------------------------------- #
SCENARIOS = {
    "Café — ordering a drink": "a barista serving a customer who wants to order a cappuccino",
    "Hotel — checking in": "a hotel receptionist checking in a guest",
    "Airport — asking for directions": "an airport staff member giving directions to gate B12",
}


def conversation_practice(scenario, user_message, language):
    """Play a role in *scenario*, reply in *language*, then correct the learner."""
    role = SCENARIOS.get(scenario, "a helpful conversation partner")
    prompt = (
        f"{ROLE}\n\n"
        f"You are {role}. The learner says: \"{user_message}\"\n"
        f"Reply in {language}, staying in character. Then end with one short "
        f"'Correction:' line fixing any language mistakes in the learner's message."
    )
    return _ask(prompt, max_length=180)


# --------------------------------------------------------------------------- #
# Tool 4: Writing Coach
# --------------------------------------------------------------------------- #
def writing_coach(text):
    """Correct the grammar of *text* and explain each change."""
    prompt = (
        f"{ROLE}\n\n"
        f"Here is a learner's writing:\n\"{text}\"\n"
        f"First give the corrected version, then list each change with a "
        f"one-line reason."
    )
    return _ask(prompt, max_length=300)


# --------------------------------------------------------------------------- #
# Gradio web UI
# --------------------------------------------------------------------------- #
def build_ui():
    """Build the four-tab Gradio interface."""
    with gr.Blocks(title="Language Learning Assistant") as demo:
        gr.Markdown(
            "# 🌍 Language Learning Assistant\n"
            "Powered by **Qwen2.5-1.5B-Instruct**, quantized and running "
            "entirely on this machine."
        )

        with gr.Tab("📖 Word Helper"):
            gr.Markdown("Type a word or phrase and get meaning, examples and a memory tip.")
            word_in = gr.Textbox(label="Word or phrase", placeholder="e.g. serendipity")
            word_btn = gr.Button("Explain", variant="primary")
            word_out = gr.Textbox(label="Explanation", lines=10)
            word_btn.click(word_helper, inputs=word_in, outputs=word_out)

        with gr.Tab("🌐 Translator"):
            gr.Markdown("Translate between English and Chinese.")
            tr_in = gr.Textbox(label="Text", lines=3, placeholder="Paste a sentence…")
            tr_dir = gr.Radio(["EN -> 中文", "中文 -> EN"], value="中文 -> EN", label="Direction")
            tr_btn = gr.Button("Translate", variant="primary")
            tr_out = gr.Textbox(label="Translation", lines=6)
            tr_btn.click(translate, inputs=[tr_in, tr_dir], outputs=tr_out)

        with gr.Tab("🗣️ Conversation Practice"):
            gr.Markdown("Role-play a real scenario — the assistant stays in character and corrects your language.")
            cv_scenario = gr.Dropdown(list(SCENARIOS.keys()), label="Scenario",
                                      value=list(SCENARIOS.keys())[0])
            cv_lang = gr.Radio(["English", "Spanish", "French"], value="English", label="Reply language")
            cv_in = gr.Textbox(label="Your message", placeholder="e.g. I want one coffee please")
            cv_btn = gr.Button("Send", variant="primary")
            cv_out = gr.Textbox(label="Partner's reply", lines=8)
            cv_btn.click(conversation_practice,
                         inputs=[cv_scenario, cv_in, cv_lang], outputs=cv_out)

        with gr.Tab("✍️ Writing Coach"):
            gr.Markdown("Paste a sentence and get a corrected version plus an explanation of each change.")
            wc_in = gr.Textbox(label="Your writing", lines=3,
                               placeholder="e.g. I have went to the store yesterday")
            wc_btn = gr.Button("Correct my writing", variant="primary")
            wc_out = gr.Textbox(label="Feedback", lines=10)
            wc_btn.click(writing_coach, inputs=wc_in, outputs=wc_out)

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
