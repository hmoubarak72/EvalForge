"""EvalForge curation dashboard (Streamlit).

Run with:  streamlit run src/evalforge/ui/dashboard.py

The dashboard talks to the same service layer as the API. All write actions
(approve / edit / reject / annotate / retire) are gated by
``EVALFORGE_AUTH__MODE``:

* ``local_dev`` (default): writes are allowed for anyone who can open the page;
  intended for local development only.
* ``token``: the reviewer must enter the shared token (``EVALFORGE_AUTH_TOKEN``)
  in the sidebar before any write control is enabled.
"""

from __future__ import annotations

import hmac
import json
import sys
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

# Allow `streamlit run` from a source checkout without `pip install -e .`.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evalforge.config import get_settings  # noqa: E402
from evalforge.container import Container, build_container  # noqa: E402
from evalforge.db.session import session_scope  # noqa: E402
from evalforge.domain.enums import Difficulty, ExpectedBehavior  # noqa: E402
from evalforge.domain.errors import EvalForgeError  # noqa: E402
from evalforge.domain.schemas import Assertion, LabelEdit  # noqa: E402
from evalforge.repositories import DatasetRepository, InteractionRepository, RunRepository  # noqa: E402

st.set_page_config(page_title="EvalForge", page_icon="🧪", layout="wide")


@st.cache_resource
def get_container() -> Container:
    """One container per Streamlit server process."""
    return build_container(get_settings())


def can_write(container: Container) -> tuple[bool, str]:
    """Sidebar authorization; returns (allowed, reviewer name)."""
    auth = container.settings.auth
    reviewer = st.sidebar.text_input("Reviewer name", value="reviewer").strip() or "reviewer"
    if auth.mode == "local_dev":
        st.sidebar.warning("Auth mode: local_dev - write actions are open. Do not expose publicly.")
        return True, reviewer
    token = st.sidebar.text_input("Write token", type="password")
    expected = auth.resolve_token() or ""
    allowed = (
        bool(token) and bool(expected) and hmac.compare_digest(token.encode(), expected.encode())
    )
    if allowed:
        st.sidebar.success("Write access granted")
    else:
        st.sidebar.info("Read-only. Enter the write token to enable review actions.")
    return allowed, reviewer


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


def _assertions_from_text(text: str) -> list[Assertion]:
    return [Assertion.model_validate(item) for item in json.loads(text or "[]")]


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def page_overview(container: Container) -> None:
    st.header("Dataset overview")
    with session_scope(container.session_factory) as session:
        quality = container.metrics.quality(session)
        growth = container.metrics.growth(session)
        freshness = container.metrics.freshness(session)
        coverage = container.metrics.coverage(session)
        interactions = InteractionRepository(session).count()

    by_source = quality["by_source"]
    cols = st.columns(6)
    cols[0].metric("Interactions ingested", interactions)
    cols[1].metric("Active test cases", quality["active_test_cases"])
    cols[2].metric("Dataset version", quality["dataset_version"])
    cols[3].metric("Auto-labeled", by_source.get("auto_labeled", 0))
    cols[4].metric("Human-reviewed", by_source.get("human_reviewed", 0))
    cols[5].metric("Freshness (30 d)", f"{freshness['freshness_pct']}%")
    cols = st.columns(4)
    cols[0].metric("Review queue", quality["review_queue_pending"])
    cols[1].metric("Duplicates skipped", quality["duplicates_skipped"])
    rate = quality["auto_accept_rate"]
    cols[2].metric("Auto-accept rate", f"{rate:.0%}" if rate is not None else "n/a")
    accuracy = quality["auto_label_accuracy_reviewed"]
    cols[3].metric(
        "Labels approved unchanged (reviewed)",
        f"{accuracy:.0%}" if accuracy is not None else "n/a",
        help="Share of human-reviewed candidates whose auto-labels were accepted as-is. "
        "Measured on the low-confidence queue only.",
    )

    st.subheader("Dataset growth over time")
    series = pd.DataFrame(growth["series"])
    if series.empty:
        st.info("No test cases yet. Run the pipeline or `python scripts/seed_demo.py --reset`.")
        return
    long = series.melt(
        id_vars="date",
        value_vars=["auto_labeled", "human_reviewed"],
        var_name="source",
        value_name="test cases",
    )
    st.altair_chart(
        alt.Chart(long)
        .mark_area(opacity=0.8)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d")),
            y=alt.Y("test cases:Q", stack=True, title="Cumulative test cases"),
            color=alt.Color("source:N", title="Label source"),
            tooltip=["date:T", "source:N", "test cases:Q"],
        )
        .properties(height=280),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("By category")
        categories = pd.DataFrame(
            sorted(quality["by_category"].items(), key=lambda kv: -kv[1]),
            columns=["category", "test cases"],
        )
        st.altair_chart(
            alt.Chart(categories)
            .mark_bar()
            .encode(
                x=alt.X("test cases:Q"),
                y=alt.Y("category:N", sort="-x", title=None),
                tooltip=["category", "test cases"],
            )
            .properties(height=max(200, 22 * len(categories))),
            width="stretch",
        )
    with right:
        st.subheader("By difficulty")
        difficulty = pd.DataFrame(
            [(d.value, quality["by_difficulty"].get(d.value, 0)) for d in Difficulty],
            columns=["difficulty", "test cases"],
        )
        st.altair_chart(
            alt.Chart(difficulty)
            .mark_bar()
            .encode(
                x=alt.X("difficulty:N", sort=[d.value for d in Difficulty], title=None),
                y="test cases:Q",
                tooltip=["difficulty", "test cases"],
            )
            .properties(height=280),
            width="stretch",
        )
        st.caption(f"Expected behaviour: {quality['by_expected_behavior']}")

    st.subheader("Coverage heatmap (cluster x difficulty)")
    rows = [
        {"cluster": c["cluster"], "difficulty": d, "test cases": n}
        for c in coverage["clusters"]
        for d, n in c["by_difficulty"].items()
    ]
    if rows:
        heat = pd.DataFrame(rows)
        st.altair_chart(
            alt.Chart(heat)
            .mark_rect()
            .encode(
                x=alt.X("difficulty:N", sort=coverage["difficulties"], title=None),
                y=alt.Y("cluster:N", title=None),
                color=alt.Color("test cases:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["cluster", "difficulty", "test cases"],
            )
            .properties(height=max(220, 24 * heat["cluster"].nunique())),
            width="stretch",
        )
        table = pd.DataFrame(coverage["clusters"])[
            [
                "cluster",
                "traffic_size",
                "traffic_share",
                "test_cases",
                "dataset_share",
                "deficit",
                "status",
            ]
        ]
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption(
            "`sparse` clusters are under-represented relative to their production traffic and "
            "are prioritised by the next pipeline run."
        )


# ---------------------------------------------------------------------------
# Dataset explorer
# ---------------------------------------------------------------------------
def page_explorer(container: Container, writable: bool, reviewer: str) -> None:
    st.header("Dataset explorer")
    with session_scope(container.session_factory) as session:
        categories = sorted(container.metrics.quality(session)["by_category"])
    c1, c2, c3, c4 = st.columns(4)
    category = c1.selectbox("Category", ["(all)", *categories])
    difficulty = c2.selectbox("Difficulty", ["(all)", *[d.value for d in Difficulty]])
    source = c3.selectbox("Review status", ["(all)", "auto_labeled", "human_reviewed"])
    status = c4.selectbox("Status", ["active", "retired", "(all)"])
    c5, c6, c7, c8 = st.columns(4)
    quality_range = c5.slider("Expected quality", 1, 5, (1, 5))
    feature_search = c6.text_input("Prompt contains")
    default_since = datetime.now(UTC).date() - timedelta(days=365)
    since = c7.date_input("Created since", value=default_since)
    until = c8.date_input("Created until", value=datetime.now(UTC).date() + timedelta(days=1))

    with session_scope(container.session_factory) as session:
        page = DatasetRepository(session).list_test_cases(
            category=None if category == "(all)" else category,
            difficulty=None if difficulty == "(all)" else difficulty,
            source=None if source == "(all)" else source,
            status=None if status == "(all)" else status,
            min_quality=quality_range[0],
            max_quality=quality_range[1],
            created_since=datetime.combine(since, time.min, tzinfo=UTC),
            created_until=datetime.combine(until, time.max, tzinfo=UTC),
            search=feature_search or None,
            limit=500,
        )
        rows = []
        for case in page.items:
            label = case.current_label
            rows.append(
                {
                    "id": case.id,
                    "created": case.created_at.date().isoformat(),
                    "category": case.category,
                    "difficulty": case.difficulty,
                    "type": case.case_type,
                    "source": case.source,
                    "behavior": label.expected_behavior if label else "",
                    "quality": label.expected_quality_score if label else None,
                    "confidence": label.confidence if label else None,
                    "prompt": case.user_prompt[:110],
                }
            )
    st.caption(f"{page.total} matching test case(s); showing up to 500.")
    if not rows:
        return
    frame = pd.DataFrame(rows)
    st.dataframe(frame.drop(columns=["id"]), width="stretch", hide_index=True, height=300)

    chosen = st.selectbox(
        "Inspect a test case",
        frame["id"],
        format_func=lambda i: frame.loc[frame["id"] == i, "prompt"].iloc[0],
    )
    with session_scope(container.session_factory) as session:
        case = DatasetRepository(session).get_test_case(chosen)
        if case is None:
            return
        label = case.current_label
        classification = (
            InteractionRepository(session).classification_for(case.source_interaction_id)
            if case.source_interaction_id
            else None
        )
        interaction = (
            InteractionRepository(session).get(case.source_interaction_id)
            if case.source_interaction_id
            else None
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**System prompt**")
            st.code(case.system_prompt or "(none)", language=None, wrap_lines=True)
            st.markdown("**User prompt**")
            st.code(case.user_prompt, language=None, wrap_lines=True)
            st.markdown("**Production response**")
            st.code(case.production_response or "(empty)", language=None, wrap_lines=True)
            if interaction is not None:
                st.markdown("**Interaction metadata**")
                st.json(
                    {
                        "timestamp": interaction.timestamp.isoformat(),
                        "feature": interaction.application_feature,
                        "model": interaction.model_name,
                        "feedback": interaction.user_feedback,
                        "retry_count": interaction.retry_count,
                        "latency_ms": interaction.latency_ms,
                        "confidence_score": interaction.confidence_score,
                        "content_filter_triggered": interaction.content_filter_triggered,
                        "redaction_status": interaction.redaction_status,
                        "metadata": interaction.metadata_,
                    },
                    expanded=False,
                )
        with right:
            st.markdown(
                f"**Cluster / category:** {case.category}  \n**Difficulty:** {case.difficulty}  \n"
                f"**Case type:** {case.case_type}  \n**Source:** {case.source} "
                f"(dataset v{case.added_in_version})"
            )
            if label is not None:
                st.markdown(
                    f"**Expected behaviour:** {label.expected_behavior}  \n"
                    f"**Expected quality:** {label.expected_quality_score}/5  \n"
                    f"**Label confidence:** {label.confidence:.2f}"
                    + (
                        f"  (agreement {label.agreement:.2f})"
                        if label.agreement is not None
                        else ""
                    )
                    + f"  \n**Label version:** {label.version} by {label.created_by}"
                )
                st.markdown("**Reference answer**")
                st.code(
                    label.reference_answer or "(rubric-based case)", language=None, wrap_lines=True
                )
                st.markdown("**Required assertions**")
                st.json(label.required_assertions, expanded=False)
                st.markdown("**Forbidden assertions**")
                st.json(label.forbidden_assertions, expanded=False)
                st.markdown("**Rubric**")
                st.json(label.rubric or [], expanded=False)
            if classification is not None:
                st.markdown(
                    f"**Production quality (judge):** {classification.quality_score}/5 - "
                    f"{classification.judge_rationale}  \n"
                    f"**Outlier reasons:** {', '.join(classification.outlier_reasons) or 'none'}"
                )
            if case.notes:
                st.markdown("**Notes**")
                st.text(case.notes)

    if not writable:
        st.info("Read-only mode: label editing is disabled.")
        return
    with st.expander("Edit labels / add note / retire"):
        with st.form(f"edit-case-{chosen}"):
            behavior = st.selectbox(
                "Expected behaviour",
                [b.value for b in ExpectedBehavior],
                index=[b.value for b in ExpectedBehavior].index(label.expected_behavior)
                if label
                else 0,
            )
            quality = st.slider(
                "Expected quality", 1, 5, label.expected_quality_score if label else 4
            )
            reference = st.text_area(
                "Reference answer", label.reference_answer or "" if label else ""
            )
            required = st.text_area(
                "Required assertions (JSON)", _json(label.required_assertions if label else [])
            )
            forbidden = st.text_area(
                "Forbidden assertions (JSON)", _json(label.forbidden_assertions if label else [])
            )
            note = st.text_input("Note")
            save = st.form_submit_button("Save new label version")
        if save:
            try:
                edit = LabelEdit(
                    expected_behavior=ExpectedBehavior(behavior),
                    expected_quality_score=quality,
                    reference_answer=reference or None,
                    required_assertions=_assertions_from_text(required),
                    forbidden_assertions=_assertions_from_text(forbidden),
                )
                with session_scope(container.session_factory) as session:
                    container.review.edit_test_case(session, chosen, reviewer, edit, note)
                st.success("Saved a new label version.")
                st.rerun()
            except (EvalForgeError, ValueError) as exc:
                st.error(f"Could not save: {exc}")
        note_only = st.text_input("Add a note only", key=f"note-{chosen}")
        b1, b2 = st.columns(2)
        if b1.button("Add note", disabled=not note_only):
            with session_scope(container.session_factory) as session:
                container.review.add_note(session, chosen, reviewer, note_only)
            st.rerun()
        if b2.button("Retire this test case"):
            try:
                with session_scope(container.session_factory) as session:
                    container.review.retire_test_case(session, chosen, reviewer, note_only)
                st.rerun()
            except EvalForgeError as exc:
                st.error(str(exc))


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------
def page_review(container: Container, writable: bool, reviewer: str) -> None:
    st.header("Review queue")
    with session_scope(container.session_factory) as session:
        items = DatasetRepository(session).pending_review_items()
        options = {
            item.id: f"[{item.priority:.1f}] {item.candidate.interaction.user_prompt[:90]}"
            for item in items
        }
    st.caption(f"{len(options)} low-confidence or conflicting candidate(s) awaiting review.")
    if not options:
        st.success("The review queue is empty.")
        return
    chosen = st.selectbox("Candidate", list(options), format_func=lambda i: options[i])

    with session_scope(container.session_factory) as session:
        item = DatasetRepository(session).get_review_item(chosen)
        if item is None:
            return
        candidate = item.candidate
        interaction, classification = candidate.interaction, candidate.classification
        labels = dict(candidate.proposed_labels)
        cluster_name = candidate.cluster.name if candidate.cluster else "Novel / Unclustered"
        st.warning(f"Why it needs review: {item.reason}")
        left, right = st.columns(2)
        with left:
            st.markdown("**User prompt**")
            st.code(interaction.user_prompt, language=None, wrap_lines=True)
            st.markdown("**Production response**")
            st.code(interaction.raw_response or "(empty)", language=None, wrap_lines=True)
            st.markdown(
                f"**Cluster:** {cluster_name}  \n**Difficulty:** {classification.difficulty}  \n"
                f"**Case type:** {candidate.case_type}  \n"
                f"**Judge:** {classification.quality_score}/5 - {classification.judge_rationale}  \n"
                f"**Selected because:** {', '.join(candidate.selection_reasons)}"
            )
            st.markdown("**Metadata**")
            st.json(
                {
                    "feedback": interaction.user_feedback,
                    "confidence_score": interaction.confidence_score,
                    "retry_count": interaction.retry_count,
                    **interaction.metadata_,
                },
                expanded=False,
            )
        with right:
            m1, m2, m3 = st.columns(3)
            m1.metric("Label confidence", f"{candidate.confidence or 0:.2f}")
            m2.metric(
                "Pass agreement",
                f"{candidate.agreement:.2f}" if candidate.agreement is not None else "n/a",
            )
            m3.metric(
                "Nearest existing case",
                f"{candidate.max_similarity:.2f}" if candidate.max_similarity else "n/a",
            )
            st.markdown("**Auto-generated labels**")
            st.json(labels, expanded=False)
            if candidate.conflicts:
                st.markdown(f"**Conflicts:** {', '.join(candidate.conflicts)}")
            st.markdown("**Similar existing test cases**")
            similar = container.review.similar_cases(session, candidate, 5)
            if similar:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "similarity": score,
                                "category": case.category,
                                "difficulty": case.difficulty,
                                "prompt": case.user_prompt[:100],
                            }
                            for case, score in similar
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("No existing test cases yet.")
            notes_so_far = [d for d in item.decisions]
            if notes_so_far:
                st.markdown("**Earlier decisions / notes**")
                for decision in notes_so_far:
                    st.caption(f"{decision.reviewer}: {decision.action} - {decision.notes}")

    if not writable:
        st.info("Read-only mode: review actions are disabled.")
        return
    notes = st.text_input("Notes", key=f"notes-{chosen}")
    b1, b2, b3 = st.columns(3)
    try:
        if b1.button("Approve as-is", type="primary"):
            with session_scope(container.session_factory) as session:
                container.review.approve(session, chosen, reviewer, notes)
            st.rerun()
        if b2.button("Reject"):
            with session_scope(container.session_factory) as session:
                container.review.reject(session, chosen, reviewer, notes)
            st.rerun()
        if b3.button("Add note only", disabled=not notes):
            with session_scope(container.session_factory) as session:
                container.review.annotate(session, chosen, reviewer, notes)
            st.rerun()
    except EvalForgeError as exc:
        st.error(str(exc))

    with st.expander("Edit labels, then approve"):
        with st.form(f"edit-{chosen}"):
            behaviors = [b.value for b in ExpectedBehavior]
            behavior = st.selectbox(
                "Expected behaviour",
                behaviors,
                index=behaviors.index(labels.get("expected_behavior", "answer")),
            )
            quality = st.slider(
                "Expected quality", 1, 5, int(labels.get("expected_quality_score", 4))
            )
            reference = st.text_area("Reference answer", labels.get("reference_answer") or "")
            required = st.text_area(
                "Required assertions (JSON)", _json(labels.get("required_assertions", []))
            )
            forbidden = st.text_area(
                "Forbidden assertions (JSON)", _json(labels.get("forbidden_assertions", []))
            )
            submitted = st.form_submit_button("Save edits and approve")
        if submitted:
            try:
                edit = LabelEdit(
                    expected_behavior=ExpectedBehavior(behavior),
                    expected_quality_score=quality,
                    reference_answer=reference or None,
                    required_assertions=_assertions_from_text(required),
                    forbidden_assertions=_assertions_from_text(forbidden),
                )
                with session_scope(container.session_factory) as session:
                    container.review.edit(session, chosen, reviewer, edit, notes)
                st.rerun()
            except (EvalForgeError, ValueError) as exc:
                st.error(f"Could not save: {exc}")


# ---------------------------------------------------------------------------
# Evaluations and regressions
# ---------------------------------------------------------------------------
def page_evaluations(container: Container) -> None:
    st.header("Evaluation runs and regressions")
    with session_scope(container.session_factory) as session:
        runs = RunRepository(session).list_evaluation_runs(limit=200).items
        rows = [
            {
                "id": r.id,
                "started": r.started_at.strftime("%Y-%m-%d %H:%M"),
                "name": r.name,
                "model": r.model_name,
                "dataset": f"v{r.dataset_version}",
                "cases": r.dataset_size,
                "pass_rate": r.pass_rate,
                "avg_score": r.average_quality_score,
                "cost_usd": r.estimated_cost_usd,
                "status": r.status,
            }
            for r in runs
        ]
    if not rows:
        st.info("No evaluation runs yet. Use `evalforge evaluate` or POST /api/v1/evaluations.")
        return
    frame = pd.DataFrame(rows)
    st.altair_chart(
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("started:T", title="Run"),
            y=alt.Y("pass_rate:Q", title="Pass rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["name", "model", "dataset", "pass_rate", "avg_score"],
        )
        .properties(height=220),
        width="stretch",
    )
    st.dataframe(frame.drop(columns=["id"]), width="stretch", hide_index=True)
    labels = {r["id"]: f"{r['started']} - {r['name']} ({r['model']})" for r in rows}
    chosen = st.selectbox("Run", list(labels), format_func=lambda i: labels[i])

    with session_scope(container.session_factory) as session:
        repo = RunRepository(session)
        run = repo.get_evaluation_run(chosen)
        if run is None:
            return
        report = dict(run.report)
        regression = repo.regression_report_for(chosen)
        regression_body = dict(regression.report) if regression else None

    overall, usage = report.get("overall", {}), report.get("usage", {})
    cols = st.columns(5)
    cols[0].metric("Pass rate", f"{overall.get('pass_rate', 0):.1%}")
    cols[1].metric("Avg quality", overall.get("average_quality_score"))
    cols[2].metric("Failed", overall.get("failed"))
    cols[3].metric("Tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
    cols[4].metric("Est. cost (USD)", usage.get("estimated_cost_usd"))

    if regression_body:
        severity = regression_body["severity"]
        message = f"Regression check vs. previous run: {regression_body['summary']}"
        (st.error if severity == "major" else st.warning if severity == "minor" else st.success)(
            message
        )
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**New failures**")
            st.dataframe(
                pd.DataFrame(regression_body["new_failures"]),
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Category shifts**")
            st.dataframe(
                pd.DataFrame(regression_body["category_shifts"]),
                width="stretch",
                hide_index=True,
            )
        with r2:
            st.markdown("**Newly passing**")
            st.dataframe(
                pd.DataFrame(regression_body["new_passes"]),
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Difficulty shifts**")
            st.dataframe(
                pd.DataFrame(regression_body["difficulty_shifts"]),
                width="stretch",
                hide_index=True,
            )
    else:
        st.caption("No earlier comparable run: this run is the baseline.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**By category**")
        st.dataframe(pd.DataFrame(report.get("by_category", {})).T, width="stretch")
    with c2:
        st.markdown("**By difficulty**")
        st.dataframe(pd.DataFrame(report.get("by_difficulty", {})).T, width="stretch")
        st.markdown("**By metric**")
        st.dataframe(pd.DataFrame(report.get("by_metric", {})).T, width="stretch")
    st.markdown("**Failed cases**")
    failed = pd.DataFrame(report.get("failed_cases", []))
    if failed.empty:
        st.caption("No failures.")
    else:
        failed["failure_reasons"] = failed["failure_reasons"].map("; ".join)
        st.dataframe(failed.drop(columns=["test_case_id"]), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Clusters and reviewers
# ---------------------------------------------------------------------------
def page_clusters(container: Container) -> None:
    st.header("Traffic clusters and edge cases")
    with session_scope(container.session_factory) as session:
        repo = InteractionRepository(session)
        clusters = [
            {
                "cluster": c.name,
                "traffic": c.size,
                "keywords": ", ".join(c.keywords),
                "named by": c.naming_method,
            }
            for c in repo.list_clusters()
        ]
        outliers = [
            {
                "difficulty": c.difficulty,
                "quality": c.quality_score,
                "reasons": ", ".join(c.outlier_reasons),
                "prompt": c.interaction.user_prompt[:120],
            }
            for c in repo.list_classifications(outliers_only=True, limit=200).items
        ]
    if clusters:
        frame = pd.DataFrame(clusters)
        st.altair_chart(
            alt.Chart(frame)
            .mark_bar()
            .encode(
                x="traffic:Q",
                y=alt.Y("cluster:N", sort="-x", title=None),
                tooltip=["cluster", "traffic", "keywords"],
            )
            .properties(height=max(220, 22 * len(frame))),
            width="stretch",
        )
        st.dataframe(frame, width="stretch", hide_index=True)
    st.subheader("Recent outliers (most valuable eval candidates)")
    st.dataframe(pd.DataFrame(outliers), width="stretch", hide_index=True)


def page_reviewers(container: Container) -> None:
    st.header("Reviewer throughput and agreement")
    with session_scope(container.session_factory) as session:
        stats = container.metrics.reviewer_stats(session)
    cols = st.columns(3)
    cols[0].metric("Double-reviewed items", stats["double_reviewed_items"])
    agreement = stats["percent_agreement"]
    cols[1].metric("Percent agreement", f"{agreement:.0%}" if agreement is not None else "n/a")
    kappa = stats["cohen_kappa"]
    cols[2].metric("Cohen's kappa", kappa if kappa is not None else "n/a")
    if stats["reviewers"]:
        st.dataframe(pd.DataFrame(stats["reviewers"]).fillna(0), width="stretch", hide_index=True)
        st.altair_chart(
            alt.Chart(pd.DataFrame(stats["throughput_by_day"]))
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d")),
                y=alt.Y("decisions:Q", title="Decisions"),
                color="reviewer:N",
                tooltip=["date:T", "reviewer", "decisions"],
            )
            .properties(height=260),
            width="stretch",
        )
    else:
        st.info("No review decisions recorded yet.")


def main() -> None:
    container = get_container()
    st.sidebar.title("🧪 EvalForge")
    st.sidebar.caption("Production logs → self-growing eval dataset")
    page = st.sidebar.radio(
        "View",
        ["Overview", "Dataset explorer", "Review queue", "Evaluations", "Clusters", "Reviewers"],
    )
    writable, reviewer = can_write(container)
    st.sidebar.caption(
        f"Offline mode: {container.settings.offline_mode} · DB: "
        f"{container.settings.database_url.split('://')[0]}"
    )
    if page == "Overview":
        page_overview(container)
    elif page == "Dataset explorer":
        page_explorer(container, writable, reviewer)
    elif page == "Review queue":
        page_review(container, writable, reviewer)
    elif page == "Evaluations":
        page_evaluations(container)
    elif page == "Clusters":
        page_clusters(container)
    else:
        page_reviewers(container)


main()
