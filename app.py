"""
Gradio UI for AskDocs — wraps the hybrid RAG pipeline for HuggingFace Spaces.

On startup the BM25 index and vector DB are loaded from disk (committed to the
Space repo via Git LFS). The embedding model and cross-encoder are downloaded
from HuggingFace Hub on first launch (~2 min), then cached for all subsequent
requests.
"""

from __future__ import annotations

import gradio as gr
from loguru import logger

from src.generation.generator import generate_answer
from src.retrieval.hybrid_retriever import retrieve

# ── Constants ─────────────────────────────────────────────────────────────────

TITLE = "AskDocs — LangChain Documentation Q&A"

DESCRIPTION = """
Ask any question about LangChain and get a precise, cited answer — no link-dumping.

**How it works:** Your question is matched against ~3,000 chunks of LangChain docs
using both keyword search (BM25) and semantic vector search. Results are fused and
re-ranked by a cross-encoder, then a Groq-hosted LLaMA 3.1 8B model synthesises a
grounded answer and cites the exact source chunks it used.
"""

ABOUT_MD = """
### Why I built this

Most RAG demos stop at "retrieve → generate". This project goes further:

- **Hybrid retrieval** — BM25 catches exact keyword matches that embeddings miss;
  dense vector search catches paraphrases that BM25 misses. Fusing both with
  Reciprocal Rank Fusion consistently outperforms either alone.
- **Cross-encoder reranking** — a second-pass `ms-marco-MiniLM` model scores each
  candidate against the query jointly, producing much tighter top-K precision than
  bi-encoder similarity alone.
- **Inline citations** — every claim links back to a specific LangChain doc page so
  you can verify the answer, not just trust it.
- **Automated quality gate** — a RAGAS evaluation pipeline runs on every GitHub
  commit and blocks merges if Answer Relevancy or Context Recall drop below
  threshold. RAG quality is treated like a test suite, not a vibe check.

**Source code:** [github.com/zeciljain8197/askdocs](https://github.com/zeciljain8197/askdocs)

**Live Space:** [huggingface.co/spaces/zinu07/askdocs](https://huggingface.co/spaces/zinu07/askdocs)
"""

EXAMPLES = [
    "What is LangChain Expression Language (LCEL)?",
    "What is a retriever in LangChain?",
    "How does LangChain handle streaming responses from LLMs?",
    "What is a vector store in LangChain?",
    "How do you create a custom tool in LangChain?",
    "What are document loaders in LangChain?",
    "What packages make up the LangChain ecosystem?",
]

FALLBACK_PHRASE = "I don't have enough context"


# ── Core logic ────────────────────────────────────────────────────────────────


def ask(question: str) -> tuple[str, str]:
    """
    Run the full pipeline and return (answer_md, citations_md).
    Both strings are Markdown — Gradio renders them directly.
    """
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    try:
        chunks = retrieve(question)
        result = generate_answer(question, chunks)
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}")
        return f"Something went wrong: {exc}", ""

    answer_md = result.answer

    if not result.citations or FALLBACK_PHRASE in result.answer:
        return answer_md, "_No sources cited — the answer may be outside the indexed docs._"

    lines = ["| # | Title | Source |", "|---|-------|--------|"]
    for c in result.citations:
        title = c.title or "LangChain Docs"
        lines.append(f"| [{c.index}] | {title} | [view]({c.source}) |")

    citations_md = "\n".join(lines)
    return answer_md, citations_md


# ── UI layout ─────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title=TITLE,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    ) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=3):
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. How does LCEL work?",
                    lines=2,
                    autofocus=True,
                )
                ask_btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("**Example questions**")
                for ex in EXAMPLES:
                    gr.Button(ex, size="sm").click(
                        fn=lambda q=ex: q,
                        outputs=question_box,
                    )

        answer_box = gr.Markdown(label="Answer", value="")
        citations_box = gr.Markdown(label="Sources", value="")

        ask_btn.click(
            fn=ask,
            inputs=question_box,
            outputs=[answer_box, citations_box],
        )
        question_box.submit(
            fn=ask,
            inputs=question_box,
            outputs=[answer_box, citations_box],
        )

        with gr.Accordion("About this project", open=False):
            gr.Markdown(ABOUT_MD)

        gr.Markdown(
            "---\n"
            "Built by [Zecil Jain](https://github.com/zeciljain8197) · "
            "[Source code](https://github.com/zeciljain8197/askdocs) · "
            "Powered by Groq LLaMA 3.1 8B"
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
