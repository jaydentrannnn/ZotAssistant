"""
Gradio web interface for the UCI Academic Assistant.

Usage:
    python -m rag_chatbot.app
    python -m rag_chatbot.app --port 8080
    python -m rag_chatbot.app --host 0.0.0.0 --port 7860

Each browser tab gets its own UUID session_id stored in gr.State.
LangChain's RunnableWithMessageHistory handles per-session history
server-side — Gradio only manages the display messages.
"""

import argparse
import uuid

import gradio as gr

from .chain import chain

# ──────────────────────────────────────────────────────────────────────────────
# Chat handler (streaming)
# ──────────────────────────────────────────────────────────────────────────────

async def _respond(
    message: str,
    history: list[dict],
    session_id: str,
):
    """
    Append the user message and a blank assistant placeholder, then stream
    the response chunk-by-chunk into the placeholder.
    Yields (cleared_input, updated_history, session_id) on every chunk.
    """
    if not message.strip():
        yield "", history, session_id
        return

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    yield "", history, session_id

    partial = ""
    async for chunk in chain.astream(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    ):
        partial += chunk
        history[-1]["content"] = partial
        yield "", history, session_id


# ──────────────────────────────────────────────────────────────────────────────
# UI layout
# ──────────────────────────────────────────────────────────────────────────────

def create_demo() -> gr.Blocks:
    with gr.Blocks(title="UCI Academic Assistant") as demo:
        gr.Markdown(
            "# UCI Academic Assistant\n"
            "Ask about UCI courses, academic policies, or major and minor requirements."
        )

        session_id = gr.State(lambda: str(uuid.uuid4()))

        chatbot = gr.Chatbot(height=520)

        with gr.Row():
            msg = gr.Textbox(
                placeholder="e.g. What are the prerequisites for CS 161?",
                scale=9,
                show_label=False,
                container=False,
                autofocus=True,
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")

        gr.Examples(
            examples=[
                "What are the prerequisites for CS 161?",
                "What courses are required for the Computer Science major?",
                "Can I take ICS 6B and ICS 6D at the same time?",
                "What is UCI's academic integrity policy?",
                "What is the deadline to add or drop a course?",
                "How many units do I need to graduate?",
            ],
            inputs=msg,
            label="Example questions",
        )

        submit_kwargs = dict(
            fn=_respond,
            inputs=[msg, chatbot, session_id],
            outputs=[msg, chatbot, session_id],
        )
        send_btn.click(**submit_kwargs)
        msg.submit(**submit_kwargs)

    return demo


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the UCI Academic Assistant Gradio web interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
