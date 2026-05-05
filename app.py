"""
Comparative Hallucination Detection in LLM-based Question Answering
Using Groq (LLaMA 3) and HuggingFace (Mistral 7B Instruct)

Streamlit Application — Main Entry Point
"""

import json
import os
import time
import streamlit as st
import pandas as pd

# -- page config ---------------------------------------------------------------
st.set_page_config(
    page_title="LLM Hallucination Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- local imports -------------------------------------------------------------
from models.llm_clients import (
    get_groq_response, get_huggingface_response,
    PROMPT_TEMPLATES, GROQ_MODEL, HF_MODEL
)
from utils.hallucination_detector import compute_hallucination_score, evaluate_batch
from utils.visualizations import (
    make_accuracy_bar_chart, make_composite_score_radar,
    make_verdict_distribution_pie, make_response_time_chart,
    make_question_heatmap, model_short_name
)

# -- CSS -----------------------------------------------------------------------
st.markdown("""
<style>
.verdict-CORRECT              { color: #16a34a; font-weight: 600; }
.verdict-LIKELY-CORRECT       { color: #65a30d; font-weight: 600; }
.verdict-LIKELY-HALLUCINATED  { color: #d97706; font-weight: 600; }
.verdict-HALLUCINATED         { color: #dc2626; font-weight: 600; }

.answer-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 14px;
    font-size: 0.9rem;
    color: #334155;
    line-height: 1.6;
}

.info-banner {
    border-left: 3px solid #3b82f6;
    padding: 10px 16px;
    background: #f0f7ff;
    border-radius: 0 6px 6px 0;
    color: #1e40af;
    margin: 12px 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# -- Data loader ---------------------------------------------------------------
@st.cache_data
def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "data", "finance_qa.json")
    with open(path) as f:
        return json.load(f)


# -- Session state init --------------------------------------------------------
def init_state():
    defaults = {
        "results": [],
        "summary": {},
        "single_result": None,
        "running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
dataset = load_dataset()


# -- Helpers -------------------------------------------------------------------
def verdict_badge(verdict: str) -> str:
    cls = "verdict-" + verdict.replace(" ", "-")
    return f'<span class="{cls}">{verdict}</span>'


def speed_comparison_banner(groq_result: dict | None, hf_result: dict | None) -> str:
    if not groq_result or not hf_result:
        return ""
    gt = groq_result.get("response_time", 0)
    ht = hf_result.get("response_time", 0)
    if gt == 0 or ht == 0:
        return ""

    if gt < ht:
        faster, slower = "Groq (LLaMA 3)", "HuggingFace (Mistral 7B)"
        diff = round(ht - gt, 3)
        ratio = round(ht / gt, 1)
    else:
        faster, slower = "HuggingFace (Mistral 7B)", "Groq (LLaMA 3)"
        diff = round(gt - ht, 3)
        ratio = round(gt / ht, 1)

    return f"""
    <div class="info-banner">
      <strong>{faster}</strong> was faster by <strong>{diff}s</strong>
      ({ratio}x speedup over {slower}) &nbsp;|&nbsp;
      Groq: {gt}s &nbsp; HuggingFace: {ht}s
    </div>
    """


def render_model_result(col, result: dict, label: str):
    with col:
        st.markdown(f"**{label}**")
        st.caption(result["model"])

        if result.get("error"):
            st.error(result["error"])
            return

        scores = result.get("scores", {})
        verdict = scores.get("verdict", "N/A")
        cls = "verdict-" + verdict.replace(" ", "-")

        st.markdown(
            f'Verdict: <span class="{cls}">{verdict}</span> &nbsp;|&nbsp; '
            f'Composite: **{scores.get("composite_score", 0):.3f}** &nbsp;|&nbsp; '
            f'Response time: **{result["response_time"]}s**',
            unsafe_allow_html=True
        )

        with st.expander("View Answer"):
            st.markdown(
                f"<div class='answer-box'>{result['answer']}</div>",
                unsafe_allow_html=True
            )

        if scores:
            with st.expander("Score Breakdown"):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Keyword Overlap", f"{scores.get('keyword_overlap', 0):.4f}")
                col_b.metric("BLEU Score",      f"{scores.get('bleu', 0):.4f}")
                col_c.metric("Jaccard Sim.",    f"{scores.get('jaccard', 0):.4f}")


# -- Query runners -------------------------------------------------------------
def run_single_query(
    question: str,
    ground_truth: str,
    question_id,
    groq_key: str,
    hf_key: str,
    prompt_style: str,
) -> tuple[dict | None, dict | None]:
    groq_result = hf_result = None

    if groq_key:
        with st.spinner("Querying Groq (LLaMA 3)..."):
            resp = get_groq_response(question, groq_key, prompt_style)
        if not resp["error"]:
            resp["scores"] = compute_hallucination_score(ground_truth, resp["answer"])
            resp.update({"question_id": question_id, "question": question,
                         "ground_truth": ground_truth})
        groq_result = resp

    if hf_key:
        with st.spinner("Querying HuggingFace (Mistral 7B)..."):
            resp = get_huggingface_response(question, hf_key, prompt_style)
        if not resp["error"]:
            resp["scores"] = compute_hallucination_score(ground_truth, resp["answer"])
            resp.update({"question_id": question_id, "question": question,
                         "ground_truth": ground_truth})
        hf_result = resp

    return groq_result, hf_result


def run_batch(
    questions_subset: list,
    groq_key: str,
    hf_key: str,
    prompt_styles: list,
    progress_bar,
    status_text,
) -> list:
    all_results = []
    total = len(questions_subset) * len(prompt_styles) * (
        (1 if groq_key else 0) + (1 if hf_key else 0)
    )
    done = 0

    for ps in prompt_styles:
        for item in questions_subset:
            for get_fn, key, delay in [
                (get_groq_response,        groq_key, 0.5),
                (get_huggingface_response, hf_key,   1.0),
            ]:
                if not key:
                    continue
                resp = get_fn(item["question"], key, ps)
                if not resp["error"]:
                    scores = compute_hallucination_score(
                        item["ground_truth"], resp["answer"]
                    )
                    all_results.append({
                        **resp,
                        "question_id": item["id"],
                        "question": item["question"],
                        "ground_truth": item["ground_truth"],
                        "scores": scores,
                    })
                done += 1
                progress_bar.progress(done / total)
                status_text.text(f"Processed {done}/{total} queries...")
                time.sleep(delay)

    return all_results


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### API Configuration")

    groq_api_key = st.text_input(
        "Groq API Key", type="password",
        placeholder="gsk_...",
        help="Get your key at https://console.groq.com/"
    )

    hf_api_key = st.text_input(
        "HuggingFace API Key", type="password",
        placeholder="hf_...",
        help="Get your key at https://huggingface.co/settings/tokens"
    )
    
    st.markdown("### Settings")

    prompt_style_choice = st.selectbox(
        "Prompt Style (Single Query)",
        list(PROMPT_TEMPLATES.keys()),
        format_func=lambda x: x.replace("_", " ").title()
    )

    st.markdown("### Batch Settings")

    batch_prompt_styles = st.multiselect(
        "Prompt Styles for Batch",
        list(PROMPT_TEMPLATES.keys()),
        default=["zero_shot", "few_shot"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    batch_question_count = st.slider(
        "Number of Questions",
        min_value=3, max_value=len(dataset),
        value=min(8, len(dataset)), step=1
    )

    category_filter = st.multiselect(
        "Filter by Category",
        list(set(q["category"] for q in dataset)),
        default=[]
    )



# =============================================================================
# HEADER
# =============================================================================
st.title("LLM Hallucination Detector")
st.divider()


# =============================================================================
# TABS
# =============================================================================
tab_single, tab_batch, tab_analytics, tab_dataset, tab_about = st.tabs([
    "Single Query", "Batch Evaluation",
    "Analytics", "Dataset", "About"
])


# -----------------------------------------------------------------------------
# TAB 1 — SINGLE QUERY
# -----------------------------------------------------------------------------
with tab_single:
    st.subheader("Ask a Finance Question")
    col_q, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        use_preset = st.checkbox("Use preset question", value=True)
        preset_idx = st.selectbox(
            "Preset question",
            range(len(dataset)),
            format_func=lambda i: f"Q{dataset[i]['id']}: {dataset[i]['question'][:55]}..."
        ) if use_preset else None

    with col_q:
        if use_preset:
            question_input = st.text_area(
                "Question", value=dataset[preset_idx]["question"], height=90)
            ground_truth_input = st.text_area(
                "Ground Truth", value=dataset[preset_idx]["ground_truth"], height=90)
        else:
            question_input = st.text_area(
                "Your Question",
                placeholder="e.g. What is the difference between stocks and bonds?",
                height=90)
            ground_truth_input = st.text_area(
                "Ground Truth Answer (for evaluation)",
                placeholder="Provide the correct answer to compare against...",
                height=90)

    run_single = st.button(
        "Run Detection", type="primary",
        disabled=not (groq_api_key or hf_api_key)
    )

    if not (groq_api_key or hf_api_key):
        st.info("Enter at least one API key in the sidebar to get started.")

    if run_single and question_input and ground_truth_input:
        qid = dataset[preset_idx]["id"] if use_preset else "custom"
        groq_res, hf_res = run_single_query(
            question_input, ground_truth_input, qid,
            groq_api_key, hf_api_key, prompt_style_choice
        )
        st.session_state["single_result"] = (groq_res, hf_res)

    if st.session_state["single_result"]:
        groq_res, hf_res = st.session_state["single_result"]

        st.divider()
        st.subheader("Results")

        if groq_res and hf_res and not groq_res.get("error") and not hf_res.get("error"):
            st.markdown(speed_comparison_banner(groq_res, hf_res), unsafe_allow_html=True)

        col_groq, col_hf = st.columns(2)
        if groq_res:
            render_model_result(col_groq, groq_res, "Groq · LLaMA 3")
        else:
            with col_groq:
                st.info("Groq key not provided.")

        if hf_res:
            render_model_result(col_hf, hf_res, "HuggingFace · Mistral 7B")
        else:
            with col_hf:
                st.info("HuggingFace key not provided.")

        with st.expander("Ground Truth"):
            st.markdown(
                f"<div class='answer-box'>{ground_truth_input}</div>",
                unsafe_allow_html=True
            )


# -----------------------------------------------------------------------------
# TAB 2 — BATCH EVALUATION
# -----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Evaluation")
    st.markdown("Run multiple questions through both models and prompt styles automatically.")

    if not (groq_api_key or hf_api_key):
        st.info("Enter at least one API key in the sidebar.")
    elif not batch_prompt_styles:
        st.warning("Select at least one prompt style in the sidebar.")
    else:
        filtered_ds = dataset
        if category_filter:
            filtered_ds = [q for q in dataset if q["category"] in category_filter]
        subset = filtered_ds[:batch_question_count]

        n_models = (1 if groq_api_key else 0) + (1 if hf_api_key else 0)
        total_queries = len(subset) * len(batch_prompt_styles) * n_models
        st.markdown(
            f"{len(subset)} questions × {len(batch_prompt_styles)} prompt styles × "
            f"{n_models} model(s) = **{total_queries} queries**"
        )

        if st.button("Start Batch Evaluation", type="primary"):
            prog = st.progress(0)
            status = st.empty()
            with st.spinner("Running batch evaluation..."):
                results = run_batch(
                    subset, groq_api_key, hf_api_key,
                    batch_prompt_styles, prog, status
                )
            st.session_state["results"] = results
            st.session_state["summary"] = evaluate_batch(results)
            status.success(f"Completed. Processed {len(results)} responses.")

    if st.session_state["results"]:
        results = st.session_state["results"]
        summary = st.session_state["summary"]

        st.divider()
        st.subheader("Summary")

        total   = len(results)
        correct = sum(1 for r in results if not r["scores"]["is_hallucinated"])
        hall    = total - correct

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Responses",  total)
        m2.metric("Correct",          correct)
        m3.metric("Hallucinated",     hall)
        m4.metric("Overall Accuracy", f"{correct / total * 100:.1f}%")

        if summary:
            times = {s["model"]: s["avg_response_time"]
                     for s in summary.values() if s["avg_response_time"] > 0}
            if len(times) == 2:
                models_sorted = sorted(times.items(), key=lambda x: x[1])
                f_name, f_t = models_sorted[0]
                s_name, s_t = models_sorted[1]
                ratio = round(s_t / f_t, 1) if f_t > 0 else "N/A"
                st.markdown(
                    f'<div class="info-banner">Speed: <strong>{model_short_name(f_name)}</strong> '
                    f'averaged {f_t}s per response ({ratio}x faster than '
                    f'{model_short_name(s_name)} at {s_t}s)</div>',
                    unsafe_allow_html=True
                )

        st.subheader("Model Comparison")
        table_rows = []
        for key, s in summary.items():
            table_rows.append({
                "Model":              model_short_name(s["model"]),
                "Prompt Style":       s["prompt_style"].replace("_", " ").title(),
                "Accuracy":           f"{s['accuracy'] * 100:.1f}%",
                "Hallucination Rate": f"{s['hallucination_rate'] * 100:.1f}%",
                "Avg Composite":      f"{s['avg_composite']:.4f}",
                "Avg Response Time":  f"{s['avg_response_time']}s",
                "Total Queries":      s["total"],
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.subheader("Detailed Results")
        detail_rows = []
        for r in results:
            s = r["scores"]
            detail_rows.append({
                "Q#":         r["question_id"],
                "Model":      model_short_name(r["model"]),
                "Prompt":     r["prompt_style"],
                "Verdict":    s["verdict"],
                "Composite":  f"{s['composite_score']:.4f}",
                "KW Overlap": f"{s['keyword_overlap']:.4f}",
                "BLEU":       f"{s['bleu']:.4f}",
                "Jaccard":    f"{s['jaccard']:.4f}",
                "Time (s)":   r["response_time"],
                "Question":   r["question"][:60] + "...",
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

        csv = pd.DataFrame(detail_rows).to_csv(index=False)
        st.download_button(
            "Download Results CSV", csv,
            "hallucination_results.csv", "text/csv"
        )


# -----------------------------------------------------------------------------
# TAB 3 — ANALYTICS
# -----------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Analytics Dashboard")

    if not st.session_state["results"]:
        st.info("Run a batch evaluation first to see analytics.")
    else:
        results = st.session_state["results"]
        summary = st.session_state["summary"]

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_accuracy_bar_chart(summary), use_container_width=True)
        with c2:
            st.plotly_chart(make_composite_score_radar(summary), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            models_in_results = list(set(r["model"] for r in results))
            sel_model = st.selectbox(
                "Model for Verdict Distribution", models_in_results,
                format_func=model_short_name
            )
            st.plotly_chart(
                make_verdict_distribution_pie(results, sel_model),
                use_container_width=True
            )
        with c4:
            st.plotly_chart(make_response_time_chart(summary), use_container_width=True)

        st.plotly_chart(make_question_heatmap(results), use_container_width=True)

        st.subheader("Insights")
        best    = max(summary.values(), key=lambda s: s["accuracy"])
        worst   = min(summary.values(), key=lambda s: s["accuracy"])
        fastest = min(summary.values(), key=lambda s: s["avg_response_time"])

        st.markdown(
            f"- **Best accuracy:** {model_short_name(best['model'])} with "
            f"{best['prompt_style']} prompt — {best['accuracy'] * 100:.1f}% correct\n"
            f"- **Most hallucinations:** {model_short_name(worst['model'])} with "
            f"{worst['prompt_style']} prompt — {worst['hallucination_rate'] * 100:.1f}% hallucination rate\n"
            f"- **Fastest model:** {model_short_name(fastest['model'])} — "
            f"avg {fastest['avg_response_time']}s per response\n"
            f"- **Total evaluations:** {len(results)} responses across "
            f"{len(set(r['model'] for r in results))} model(s) and "
            f"{len(set(r['prompt_style'] for r in results))} prompt style(s)"
        )


# -----------------------------------------------------------------------------
# TAB 4 — DATASET
# -----------------------------------------------------------------------------
with tab_dataset:
    st.subheader("Finance QA Dataset")
    st.markdown(
        f"{len(dataset)} questions across "
        f"{len(set(q['category'] for q in dataset))} categories"
    )

    cat_col, _ = st.columns([2, 3])
    with cat_col:
        filter_cat = st.selectbox(
            "Filter by Category",
            ["All"] + sorted(set(q["category"] for q in dataset))
        )

    display_ds = (
        dataset if filter_cat == "All"
        else [q for q in dataset if q["category"] == filter_cat]
    )

    for item in display_ds:
        with st.expander(f"Q{item['id']}: {item['question']}"):
            st.markdown(f"**Category:** `{item['category']}`")
            st.markdown(f"**Context:** {item['context']}")
            st.markdown("**Ground Truth Answer:**")
            st.markdown(
                f"<div class='answer-box'>{item['ground_truth']}</div>",
                unsafe_allow_html=True
            )


# -----------------------------------------------------------------------------
# TAB 5 — ABOUT
# -----------------------------------------------------------------------------
with tab_about:
    st.subheader("About This Project")

    st.markdown("#### Problem Statement")
    st.markdown(
        "Large Language Models often generate fluent but factually incorrect answers, "
        "known as hallucinations. This is a significant concern in finance where accuracy is critical."
    )

    st.markdown("#### Models Compared")
    st.markdown(
        "- **Groq — LLaMA 3 (llama-3.1-8b-instant):** Served via Groq's fast inference engine.\n"
        "- **HuggingFace — Mistral 7B Instruct v0.2:** Served via the HuggingFace Inference API."
    )

    st.markdown("#### Detection Methodology")
    st.markdown(
        "Generated answers are compared against ground truth using three complementary metrics:\n\n"
        "- **Keyword Overlap (35%):** Proportion of key reference terms found in the generated answer\n"
        "- **BLEU Score (35%):** N-gram precision measuring textual similarity\n"
        "- **Jaccard Similarity (30%):** Token-level set overlap between answer and reference\n\n"
        "A composite score is thresholded to classify responses as: "
        "CORRECT · LIKELY CORRECT · LIKELY HALLUCINATED · HALLUCINATED"
    )

    st.markdown("#### Prompt Strategies")
    st.markdown(
        "- **Zero-Shot:** Direct question with no examples\n"
        "- **Few-Shot:** Question with 2 finance-domain example pairs\n"
        "- **Chain-of-Thought:** Structured step-by-step reasoning prompt"
    )

    st.markdown("#### Limitations")
    st.markdown(
        "- Lexical similarity metrics may miss semantically equivalent paraphrases\n"
        "- Ground truth quality directly affects evaluation reliability\n"
        "- HuggingFace cold-start latency can inflate first-call response times\n"
        "- Response time includes network latency, not just model inference time"
    )