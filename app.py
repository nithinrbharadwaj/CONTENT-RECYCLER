"""
app.py
======
Main entry point for the Content Recycler tool.

CLI mode:
    python app.py --ingest
    python app.py --recycle "remote work productivity" --platform LinkedIn
    python app.py --recycle "AI for beginners" --platform Twitter --evaluate
    python app.py --stats

Streamlit UI mode:
    streamlit run app.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def cli_ingest(args: argparse.Namespace) -> None:
    from src.ingestion import ingest

    count = ingest(
        data_path=args.data,
        persist_dir=args.db,
        reset=args.reset,
    )
    print(f"\n✅  Ingestion complete — {count} chunks stored in vector DB.\n")


def cli_recycle(args: argparse.Namespace) -> None:
    from src.retrieval import retrieve_posts, print_results
    from src.generator import recycle_post
    from src.eval import evaluate, print_report

    query = args.recycle
    target_platform = args.platform or "LinkedIn"

    print(f"\n🔍  Searching for posts related to: '{query}'")
    results = retrieve_posts(
        query=query,
        top_n=3,
        platform_filter=args.source_platform,
        persist_dir=args.db,
    )

    if not results:
        print("❌  No matching posts found. Have you run --ingest first?")
        sys.exit(1)

    print_results(results)

    # Use the top result
    top = results[0]
    print(f"\n🔁  Recycling top result for {target_platform}…")

    result = recycle_post(
        original_text=top["original_text"],
        source_platform=top["platform"],
        original_date=top["date_posted"],
        target_platform=target_platform,
    )

    print("\n" + "=" * 70)
    print("  RECYCLED POST")
    print("=" * 70)
    print(result["recycled_text"])
    print("=" * 70 + "\n")

    if args.evaluate:
        report = evaluate(
            original=top["original_text"],
            recycled=result["recycled_text"],
            metadata={
                "source_platform": top["platform"],
                "target_platform": target_platform,
                "post_id": top["id"],
                "provider": result.get("provider", ""),
            },
        )
        print_report(report)

    # Token usage
    usage = result.get("usage", {})
    if usage:
        print(
            f"💰  Token usage — prompt: {usage.get('prompt_tokens',0)}, "
            f"completion: {usage.get('completion_tokens',0)}, "
            f"total: {usage.get('total_tokens',0)}, "
            f"latency: {usage.get('latency_s',0):.2f}s"
        )


def cli_stats(args: argparse.Namespace) -> None:
    from src.eval import load_eval_log

    records = load_eval_log()
    if not records:
        print("No evaluation records found. Run --recycle --evaluate first.")
        return

    scores = [r["bleu_score"] for r in records]
    avg = sum(scores) / len(scores)
    print(f"\n📊  Evaluation Stats ({len(scores)} records)")
    print(f"   Avg BLEU : {avg:.4f}")
    print(f"   Min BLEU : {min(scores):.4f}")
    print(f"   Max BLEU : {max(scores):.4f}")
    print("\n   Recent results:")
    for r in records[-5:]:
        meta = r.get("metadata", {})
        print(
            f"   [{r['timestamp'][:10]}]  BLEU={r['bleu_score']:.4f}  "
            f"{meta.get('source_platform','')} → {meta.get('target_platform','')}  "
            f"  {r['interpretation']}"
        )
    print()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def streamlit_app() -> None:
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit")
        sys.exit(1)

    st.set_page_config(
        page_title="Content Recycler 🔁",
        page_icon="🔁",
        layout="wide",
    )

    st.title("🔁 Content Recycler")
    st.caption(
        "AI-powered tool to revive and repurpose old social media posts using RAG."
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        db_path = st.text_input("Vector DB path", value="./vector_db")
        data_path = st.text_input("Dataset path", value="./data/old_posts.csv")
        provider = st.selectbox("LLM Provider", ["openai", "gemini"])
        target_platform = st.selectbox(
            "Target Platform",
            ["LinkedIn", "Twitter", "Instagram", "Facebook", "Threads"],
        )
        source_platform_filter = st.selectbox(
            "Filter source by platform",
            ["(any)", "LinkedIn", "Twitter", "Instagram", "Facebook"],
        )
        top_n = st.slider("Number of retrieved posts", 1, 10, 3)
        show_eval = st.checkbox("Show BLEU evaluation", value=True)

        st.divider()
        if st.button("🗄️ Re-index database", use_container_width=True):
            from src.ingestion import ingest

            with st.spinner("Ingesting…"):
                count = ingest(data_path=data_path, persist_dir=db_path)
            st.success(f"✅ {count} chunks indexed!")

    # Main area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 Search & Retrieve")
        query = st.text_input(
            "Enter topic or keyword", placeholder="e.g. AI productivity tips"
        )

        if st.button("🔍 Search", use_container_width=True) and query:
            from src.retrieval import retrieve_posts

            plat_filter = (
                None if source_platform_filter == "(any)" else source_platform_filter
            )
            with st.spinner("Searching vector DB…"):
                results = retrieve_posts(
                    query=query,
                    top_n=top_n,
                    platform_filter=plat_filter,
                    persist_dir=db_path,
                )
            st.session_state["results"] = results

        if "results" in st.session_state:
            results = st.session_state["results"]
            if results:
                st.success(f"Found {len(results)} results")
                for i, post in enumerate(results):
                    with st.expander(
                        f"[{i+1}] {post['platform']} | Score: {post['similarity_score']:.3f} | Engagement: {post['engagement_score']}"
                    ):
                        st.write(post["original_text"])
                        st.caption(
                            f"Date: {post['date_posted']}  |  Tone: {post['tone']}  |  Tags: {post['tags']}"
                        )
            else:
                st.warning("No results found. Have you indexed the database?")

    with col2:
        st.subheader("✨ Recycle Post")

        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            selected_idx = st.selectbox(
                "Select post to recycle",
                range(len(results)),
                format_func=lambda i: f"[{i+1}] {results[i]['platform']} — {results[i]['original_text'][:60]}…",
            )
            selected = results[selected_idx]

            st.text_area(
                "Original text (editable)",
                value=selected["original_text"],
                key="edit_original",
                height=120,
            )

            if st.button("🚀 Recycle!", use_container_width=True):
                from src.generator import recycle_post

                original = st.session_state["edit_original"]
                with st.spinner(f"Generating via {provider}…"):
                    try:
                        result = recycle_post(
                            original_text=original,
                            source_platform=selected["platform"],
                            original_date=selected["date_posted"],
                            target_platform=target_platform,
                            provider=provider,
                        )
                        st.session_state["last_result"] = result
                        st.session_state["last_original"] = original
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

            if "last_result" in st.session_state:
                result = st.session_state["last_result"]
                original = st.session_state["last_original"]

                st.text_area(
                    "♻️ Recycled Post", value=result["recycled_text"], height=150
                )

                usage = result.get("usage", {})
                if usage:
                    st.caption(
                        f"Model: {usage.get('model',provider)} | "
                        f"Tokens: {usage.get('total_tokens',0)} | "
                        f"Latency: {usage.get('latency_s',0):.2f}s"
                    )

                if show_eval:

                    report = evaluate(
                        original=original,
                        recycled=result["recycled_text"],
                        metadata={
                            "source_platform": selected["platform"],
                            "target_platform": target_platform,
                        },
                    )
                    score = report["bleu_score"]
                    interp = report["interpretation"]

                    st.divider()
                    st.subheader("📊 BLEU Evaluation")
                    st.metric("BLEU Score", f"{score:.4f}", help="0.2–0.5 is ideal")
                    st.progress(min(score, 1.0))
                    st.info(interp)
        else:
            st.info("Search for posts on the left first, then recycle here.")

    # Eval history tab
    st.divider()
    with st.expander("📈 Evaluation History"):
        from src.eval import load_eval_log

        records = load_eval_log()
        if records:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "Date": r["timestamp"][:10],
                        "BLEU": round(r["bleu_score"], 4),
                        "Assessment": r["interpretation"],
                        "Source": r.get("metadata", {}).get("source_platform", ""),
                        "Target": r.get("metadata", {}).get("target_platform", ""),
                    }
                    for r in records
                ]
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No evaluation history yet.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="content-recycler",
        description="AI Content Recycler — RAG-Driven Social Media Automation",
    )
    parser.add_argument(
        "--ingest", action="store_true", help="Ingest/index the dataset"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Wipe DB before re-ingesting"
    )
    parser.add_argument(
        "--recycle", metavar="QUERY", help="Recycle a post matching QUERY"
    )
    parser.add_argument(
        "--platform", metavar="PLATFORM", help="Target platform (default: LinkedIn)"
    )
    parser.add_argument(
        "--source-platform",
        dest="source_platform",
        default=None,
        help="Filter source posts by platform",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Show BLEU evaluation after recycling"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show evaluation statistics"
    )
    parser.add_argument(
        "--data",
        default=os.getenv("DATA_PATH", "./data/old_posts.csv"),
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("CHROMA_PERSIST_DIR", "./vector_db"),
        help="ChromaDB persist directory",
    )
    return parser


if __name__ == "__main__":
    # Detect if running via `streamlit run app.py`
    if "streamlit" in sys.modules or os.getenv("STREAMLIT_RUNTIME"):
        streamlit_app()
    else:
        import streamlit.web.bootstrap as _st_bootstrap  # noqa

        # If called as `streamlit run app.py`, Streamlit will re-import __main__
        # and the check above will catch it. For plain `python app.py` fall through:
        parser = _build_parser()
        args = parser.parse_args()

        if args.ingest:
            cli_ingest(args)
        elif args.recycle:
            cli_recycle(args)
        elif args.stats:
            cli_stats(args)
        else:
            parser.print_help()
