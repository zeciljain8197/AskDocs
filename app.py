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

FALLBACK_PHRASE = "I don't have enough context"

EXAMPLES = [
    "What is LangChain Expression Language (LCEL)?",
    "What is a retriever in LangChain?",
    "How does LangChain handle streaming responses from LLMs?",
    "What is a vector store in LangChain?",
    "How do you create a custom tool in LangChain?",
    "What are document loaders in LangChain?",
    "What packages make up the LangChain ecosystem?",
    "What is the difference between invoke, stream, and batch in LangChain?",
]

HEADER_MD = """
<div style="text-align: center; padding: 24px 0 8px 0;">
  <h1 style="font-size: 2.4rem; font-weight: 700; margin-bottom: 6px;">
    📚 AskDocs
  </h1>
  <p style="font-size: 1.1rem; color: #555; margin-bottom: 16px;">
    Ask anything about <strong>LangChain</strong> — get a grounded answer with source citations
  </p>
  <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; font-size: 0.85rem;">
    <span style="background:#e8f4fd; color:#1a73e8; padding:4px 12px; border-radius:20px;">⚡ Hybrid BM25 + Vector Search</span>
    <span style="background:#e8f4fd; color:#1a73e8; padding:4px 12px; border-radius:20px;">🔁 Cross-Encoder Reranking</span>
    <span style="background:#e8f4fd; color:#1a73e8; padding:4px 12px; border-radius:20px;">📎 Inline Citations</span>
    <span style="background:#e8f4fd; color:#1a73e8; padding:4px 12px; border-radius:20px;">🤖 Groq LLaMA 3.1 8B</span>
  </div>
</div>
"""

HOW_IT_WORKS_MD = """
### How it works

1. **Retrieve** — your question hits a BM25 keyword index and a dense vector index simultaneously. Results from both are fused via Reciprocal Rank Fusion (RRF).
2. **Rerank** — a cross-encoder (`ms-marco-MiniLM`) scores each candidate chunk against your question jointly, surfacing the most relevant 5–8 chunks.
3. **Generate** — those chunks are sent to Groq LLaMA 3.1 8B with a prompt that requires inline citations. Every claim maps back to a source chunk.
4. **Evaluate** — a RAGAS quality gate runs on every GitHub commit. CI fails if Answer Relevancy or Context Recall drop below threshold.

**Why hybrid over pure vector search?** Keyword queries (e.g. "LCEL syntax") beat embeddings for exact terms. Semantic search beats BM25 for paraphrases. Fusing both wins across query types.

→ [Source code on GitHub](https://github.com/zeciljain8197/askdocs)
"""

FOOTER_MD = """
<div style="text-align:center; padding: 20px 0 8px 0; color: #888; font-size: 0.82rem; border-top: 1px solid #eee; margin-top: 16px;">
  Built by <a href="https://github.com/zeciljain8197" target="_blank">Zecil Jain</a> &nbsp;·&nbsp;
  <a href="https://github.com/zeciljain8197/askdocs" target="_blank">GitHub</a> &nbsp;·&nbsp;
  Powered by Groq LLaMA 3.1 8B &nbsp;·&nbsp; ~3,000 LangChain doc chunks indexed
</div>
"""


# ── Startup: pre-warm models so first user doesn't wait ──────────────────────


def _warmup():
    logger.info("Pre-warming retrieval models …")
    try:
        retrieve("what is langchain")
        logger.info("Warmup complete — models loaded and ready")
    except Exception as exc:
        logger.warning(f"Warmup failed (non-fatal): {exc}")


_warmup()


# ── Core logic ────────────────────────────────────────────────────────────────


def ask(question: str) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "_Please enter a question above._", ""

    try:
        chunks = retrieve(question)
        result = generate_answer(question, chunks)
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}")
        return f"⚠️ Something went wrong: {exc}", ""

    answer_md = result.answer

    if not result.citations or FALLBACK_PHRASE in result.answer:
        return (
            f"{answer_md}\n\n---\n_This question may be outside the indexed documentation. "
            "Try rephrasing or check a related topic._",
            "",
        )

    # Build citations table
    lines = [
        "**Sources**",
        "",
        "| Ref | Document | Link |",
        "|-----|----------|------|",
    ]
    for c in result.citations:
        if c.title and len(c.title.split()) > 3:
            title = c.title.strip()
        else:
            title = (
                c.source.split("/")[-1]
                .replace(".mdx", "")
                .replace(".md", "")
                .replace("-", " ")
                .replace("_", " ")
                .title()
            )
        lines.append(f"| [{c.index}] | {title} | [view ↗]({c.source}) |")

    citations_md = "\n".join(lines)
    return answer_md, citations_md


# ── UI layout ─────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="AskDocs — LangChain Q&A",
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        ),
        css="""
            .contain { max-width: 900px; margin: auto; }
            .answer-box { min-height: 120px; }
            .ask-btn { font-size: 1rem !important; }
            footer { display: none !important; }
        """,
    ) as demo:
        gr.HTML(HEADER_MD)

        with gr.Row():
            with gr.Column(scale=4):
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. How does LCEL work?  •  What is a retriever?  •  How do I stream LLM output?",
                    lines=3,
                    autofocus=True,
                    show_label=True,
                )
                with gr.Row():
                    ask_btn = gr.Button(
                        "Ask →",
                        variant="primary",
                        elem_classes=["ask-btn"],
                        scale=2,
                    )
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        with gr.Row():
            with gr.Column(scale=4):
                answer_box = gr.Markdown(
                    value="_Your answer will appear here._",
                    elem_classes=["answer-box"],
                    label="Answer",
                )
                citations_box = gr.Markdown(value="", label="Sources")

        gr.Markdown(
            "_⏱ Retrieval: ~3–5 s &nbsp;·&nbsp; Generation: ~3–8 s on Groq free tier_",
        )

        with gr.Accordion("💡 Example questions — click to load", open=True):
            with gr.Row():
                for ex in EXAMPLES[:4]:
                    gr.Button(ex, variant="secondary").click(
                        fn=lambda q=ex: q,
                        outputs=question_box,
                    )
            with gr.Row():
                for ex in EXAMPLES[4:]:
                    gr.Button(ex, variant="secondary").click(
                        fn=lambda q=ex: q,
                        outputs=question_box,
                    )

        with gr.Accordion("⚙️ How it works", open=False):
            gr.Markdown(HOW_IT_WORKS_MD)

        ask_btn.click(
            fn=ask,
            inputs=question_box,
            outputs=[answer_box, citations_box],
            api_name="ask",
        )
        question_box.submit(
            fn=ask,
            inputs=question_box,
            outputs=[answer_box, citations_box],
        )
        clear_btn.click(
            fn=lambda: ("_Your answer will appear here._", ""),
            outputs=[answer_box, citations_box],
        )

        gr.HTML(FOOTER_MD)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(ssr_mode=False)
