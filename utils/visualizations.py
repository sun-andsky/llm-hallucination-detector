"""
Visualization helpers for hallucination detection dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


COLORS = {
    "gemini": "#4285F4",       # Google Blue
    "groq_llama": "#F97316",   # Orange
    "correct": "#22C55E",      # Green
    "hallucinated": "#EF4444", # Red
    "likely_correct": "#86EFAC",
    "likely_hallucinated": "#FCA5A5",
    "bg": "#0F172A",
    "card": "#1E293B",
    "text": "#F1F5F9"
}

MODEL_COLOR_MAP = {
    "gemini-2.0-flash": COLORS["gemini"],
    "llama-3.1-8b-instant": COLORS["groq_llama"],
    "llama-3.3-70b-versatile": "#A855F7",
    "qwen/qwen3-32b": "#14B8A6"
}

VERDICT_COLOR_MAP = {
    "CORRECT": COLORS["correct"],
    "LIKELY CORRECT": COLORS["likely_correct"],
    "LIKELY HALLUCINATED": COLORS["likely_hallucinated"],
    "HALLUCINATED": COLORS["hallucinated"]
}


def model_short_name(model: str) -> str:
    mapping = {
        "gemini-2.0-flash": "Gemini 2.0 Flash",
        "llama-3.1-8b-instant": "Llama 3.1-8B (Groq)",
        "llama-3.3-70b-versatile": "Llama 3.3-70B (Groq)",
        "qwen/qwen3-32b": "Qwen3-32B (Groq)"
    }
    return mapping.get(model, model)


def make_accuracy_bar_chart(summary: dict) -> go.Figure:
    """Bar chart comparing accuracy and hallucination rate per model+prompt."""
    rows = []
    for key, s in summary.items():
        rows.append({
            "Label": f"{model_short_name(s['model'])}\n({s['prompt_style']})",
            "Accuracy": s["accuracy"] * 100,
            "Hallucination Rate": s["hallucination_rate"] * 100,
            "Model": model_short_name(s["model"]),
            "Prompt": s["prompt_style"]
        })

    df = pd.DataFrame(rows)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Accuracy (%)",
        x=df["Label"],
        y=df["Accuracy"],
        marker_color=COLORS["correct"],
        text=df["Accuracy"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name="Hallucination Rate (%)",
        x=df["Label"],
        y=df["Hallucination Rate"],
        marker_color=COLORS["hallucinated"],
        text=df["Hallucination Rate"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"
    ))

    fig.update_layout(
        barmode="group",
        title="Model Accuracy vs Hallucination Rate",
        xaxis_title="Model + Prompt Style",
        yaxis_title="Percentage (%)",
        yaxis=dict(range=[0, 110]),
        plot_bgcolor="#1E293B",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155"),
        margin=dict(t=60, b=80, l=40, r=20)
    )
    return fig


def make_composite_score_radar(summary: dict) -> go.Figure:
    """Radar chart of avg composite score, accuracy, and speed (inverted time)."""
    categories = ["Accuracy", "Avg Composite Score", "Speed (inv)", "Reliability"]
    fig = go.Figure()

    max_time = max((s["avg_response_time"] for s in summary.values()), default=1) or 1

    for key, s in summary.items():
        speed_inv = 1 - (s["avg_response_time"] / (max_time * 1.2))
        reliability = 1 - s["hallucination_rate"]
        values = [
            s["accuracy"],
            s["avg_composite"],
            round(speed_inv, 3),
            round(reliability, 3)
        ]
        values += [values[0]]  # close the radar
        cats = categories + [categories[0]]

        label = f"{model_short_name(s['model'])} ({s['prompt_style']})"
        color = MODEL_COLOR_MAP.get(s["model"], "#94A3B8")

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill="toself",
            name=label,
            line_color=color,
            fillcolor=color,
            opacity=0.4
        ))

    fig.update_layout(
        title="Model Performance Radar",
        polar=dict(
            bgcolor="#1E293B",
            radialaxis=dict(visible=True, range=[0, 1], color="#94A3B8"),
            angularaxis=dict(color="#94A3B8")
        ),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        legend=dict(bgcolor="#1E293B"),
        margin=dict(t=60)
    )
    return fig


def make_verdict_distribution_pie(results: list, model_filter: str = None) -> go.Figure:
    """Pie chart of verdict distribution for a given model."""
    filtered = [r for r in results if model_filter is None or r["model"] == model_filter]
    verdicts = [r["scores"]["verdict"] for r in filtered if "scores" in r]

    counts = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [VERDICT_COLOR_MAP.get(l, "#94A3B8") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo="label+percent"
    )])

    title = f"Verdict Distribution — {model_short_name(model_filter)}" if model_filter else "Overall Verdict Distribution"
    fig.update_layout(
        title=title,
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        legend=dict(bgcolor="#1E293B"),
        margin=dict(t=60)
    )
    return fig


def make_response_time_chart(summary: dict) -> go.Figure:
    """Bar chart of average response times."""
    labels = []
    times = []
    colors = []

    for key, s in summary.items():
        labels.append(f"{model_short_name(s['model'])}\n({s['prompt_style']})")
        times.append(s["avg_response_time"])
        colors.append(MODEL_COLOR_MAP.get(s["model"], "#94A3B8"))

    fig = go.Figure(go.Bar(
        x=labels,
        y=times,
        marker_color=colors,
        text=[f"{t:.2f}s" for t in times],
        textposition="outside"
    ))

    fig.update_layout(
        title="Average Response Time per Model",
        xaxis_title="Model + Prompt Style",
        yaxis_title="Seconds",
        plot_bgcolor="#1E293B",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        margin=dict(t=60, b=80, l=40, r=20)
    )
    return fig


def make_question_heatmap(results: list) -> go.Figure:
    """Heatmap of composite scores per question per model."""
    rows = []
    for r in results:
        rows.append({
            "Question": f"Q{r['question_id']}",
            "Model": model_short_name(r["model"]),
            "Score": r.get("scores", {}).get("composite_score", 0)
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()

    pivot = df.pivot_table(index="Model", columns="Question", values="Score", aggfunc="mean")

    fig = px.imshow(
        pivot,
        color_continuous_scale=["#EF4444", "#FCD34D", "#22C55E"],
        zmin=0, zmax=0.6,
        title="Composite Score Heatmap (per Question)",
        labels=dict(color="Score")
    )
    fig.update_layout(
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F1F5F9"),
        margin=dict(t=60)
    )
    return fig