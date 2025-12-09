# modules/app.py

import os
import time
from typing import List, Dict

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import random
import numpy as np
from streamlit.components.v1 import html
import networkx as nx

# Local modules
from pre_processing import clean_text, preprocess_series
from ner_relation import extract_entities_and_triples, safe_load_spacy_model
from graphbuilder import build_graph_from_triples, build_pyvis_html
from semantic_search import build_sentence_index, semantic_search_query

# Optional import for similarity histogram in admin
try:
    from sentence_transformers import util as st_util
except Exception:
    st_util = None

# ------------------ Streamlit config & CSS ------------------
st.set_page_config(page_title="Infosys Knowledge-Map", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #020617; }
    .main { background-color: #020617; }
    .stepper-container {
        background: #020617;
        padding: 1rem;
        border-right: 1px solid #1f2937;
    }
    .step {
        padding: 0.6rem 0.4rem;
        margin-bottom: 0.3rem;
        border-radius: 0.5rem;
        cursor: pointer;
    }
    .step-active {
        background: #1d4ed8;
        color: white;
        box-shadow: 0 0 10px rgba(59,130,246,0.8);
    }
    .step-completed { background: #047857; color: white; }
    .step-locked { background: #111827; color: #6b7280; }
    .step-label { font-size: 0.9rem; font-weight: 500; }
    .step-caption { font-size: 0.75rem; color: #9ca3af; }
    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1f2937;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Session state ------------------
ss = st.session_state
ss.setdefault("uploaded_df", None)
ss.setdefault("preprocessed_df", None)
ss.setdefault("triples_df", None)
ss.setdefault("nlp_model", None)
ss.setdefault("graph", None)
ss.setdefault("sentence_texts", [])
ss.setdefault("sentence_embeddings", None)
ss.setdefault("sentence_model", None)
ss.setdefault("pipeline_logs", [])
ss.setdefault("last_stats", {})
ss.setdefault("current_step", 1)        # 1..5
ss.setdefault("view_mode", "pipeline")  # "pipeline" | "admin"
ss.setdefault("step_completed", {1: False, 2: False, 3: False, 4: False, 5: False})

STEPS: Dict[int, Dict[str, str]] = {
    1: {"label": "Upload Data", "desc": "Load CSV dataset"},
    2: {"label": "Preprocessing", "desc": "Clean & normalize text"},
    3: {"label": "NER + Relation Extraction", "desc": "Extract entities & triples"},
    4: {"label": "Knowledge Graph", "desc": "Visualize entities as graph"},
    5: {"label": "Semantic Search", "desc": "Search semantically across text"},
}

# ------------------ Cached semantic index ------------------
@st.cache_data(show_spinner=False)
def cached_build_sentence_index(texts: List[str], model_name: str):
    return build_sentence_index(texts, model_name)

# ------------------ Header ------------------
header_col1, header_col2 = st.columns([4, 1], vertical_alignment="center")
with header_col1:
    st.markdown(
        '<div class="header-bar"><h2 style="color:#e5e7eb;margin:0;">Infosys Knowledge-Map</h2></div>',
        unsafe_allow_html=True,
    )
with header_col2:
    if st.button("🔒 Admin Dashboard", key="admin_button"):
        ss.view_mode = "admin"

if ss.view_mode == "admin":
    if st.button("⬅ Back to Pipeline", key="back_to_pipeline"):
        ss.view_mode = "pipeline"

left_col, right_col = st.columns([1.2, 4], gap="large")

# ------------------ LEFT: Stepper ------------------
with left_col:
    st.markdown('<div class="stepper-container">', unsafe_allow_html=True)

    # auto-compute completion
    ss.step_completed[1] = ss.uploaded_df is not None
    ss.step_completed[2] = ss.preprocessed_df is not None
    ss.step_completed[3] = ss.triples_df is not None and not (ss.triples_df is None or ss.triples_df.empty)
    ss.step_completed[4] = (
        ss.graph is not None and isinstance(ss.graph, nx.Graph) and ss.graph.number_of_nodes() > 0
    )
    ss.step_completed[5] = ss.sentence_embeddings is not None

    for step_id, info in STEPS.items():
        completed = ss.step_completed.get(step_id, False)
        enabled = True if step_id == 1 else ss.step_completed.get(step_id - 1, False)
        if ss.view_mode != "pipeline":
            enabled = False

        base_class = "step"
        if ss.current_step == step_id and ss.view_mode == "pipeline":
            base_class += " step-active"
        elif completed:
            base_class += " step-completed"
        elif not enabled:
            base_class += " step-locked"

        icon = "✔️" if completed else ("▶️" if enabled else "🔒")

        st.markdown(
            f"""
            <div class="{base_class}">
              <div class="step-label">{icon} {step_id}. {info['label']}</div>
              <div class="step-caption">{info['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if enabled and ss.view_mode == "pipeline":
            if st.button(f"Go to: {info['label']}", key=f"goto_step_{step_id}"):
                ss.current_step = step_id

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ RIGHT: Main Content ------------------
with right_col:

    # ===== ADMIN VIEW =====
    if ss.view_mode == "admin":
        st.subheader("Admin Dashboard & Pipeline Report")

        # Logs
        st.markdown("### Run logs (latest first)")
        for log in reversed(ss.pipeline_logs[-15:]):
            st.write("•", log)

        # Preprocessing summary
        st.markdown("### Preprocessing summary")
        preproc = ss.preprocessed_df
        if preproc is None:
            st.write("No preprocessing run yet.")
        else:
            st.metric("Rows processed", len(preproc))
            if "clean_text" in preproc.columns:
                missing = preproc["clean_text"].isna().sum()
                st.metric("Missing after cleaning", missing)

        # NER & RE stats
        st.markdown("### NER & RE coverage")
        stats = ss.get("last_stats", {})
        triples_df = ss.triples_df
        if stats:
            rows_total = stats["rows_total"]
            st.metric("Rows total", rows_total)
            st.metric(
                "Rows with ≥1 entity",
                f"{stats['rows_with_entity']} ({stats['rows_with_entity']/rows_total*100:.1f}%)",
            )
            st.metric(
                "Rows with ≥1 triple",
                f"{stats['rows_with_triple']} ({stats['rows_with_triple']/rows_total*100:.1f}%)",
            )
            st.metric("Total triples", stats["total_triples"])

            ent_df = (
                pd.DataFrame.from_dict(stats["entity_counts"], orient="index", columns=["count"])
                .sort_values("count", ascending=False)
                .reset_index()
                .rename(columns={"index": "label"})
            )
            if not ent_df.empty:
                st.markdown("#### Entity distribution (top 20)")
                st.bar_chart(ent_df.head(20).set_index("label"))
                with st.expander("Entity counts table"):
                    st.dataframe(ent_df)

            st.markdown("#### Sample extracted relations")
            if triples_df is not None and not triples_df.empty:
                st.dataframe(triples_df.head(200))
        else:
            st.write("No NER/RE statistics available yet.")

        # Graph summary
        st.markdown("### Knowledge graph summary")
        graph = ss.graph
        if graph is None or graph.number_of_nodes() == 0:
            st.write("Graph not built yet.")
        else:
            st.metric("Nodes", graph.number_of_nodes())
            st.metric("Edges", graph.number_of_edges())
            degs = dict(graph.degree())
            deg_series = pd.Series(list(degs.values()))
            fig, ax = plt.subplots()
            ax.hist(deg_series, bins=20)
            ax.set_xlabel("Degree")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        # Semantic similarity distribution
        st.markdown("### Semantic search similarity distribution")
        if ss.sentence_embeddings is not None and st_util is not None:
            embs = ss.sentence_embeddings
            n = min(200, embs.shape[0])
            if n >= 2:
                indices = random.sample(range(embs.shape[0]), n)
                sample_embs = embs[indices]
                sims = st_util.cos_sim(sample_embs, sample_embs).cpu().numpy()
                triu = sims[np.triu_indices(n, k=1)]
                fig2, ax2 = plt.subplots()
                ax2.hist(triu, bins=30)
                ax2.set_title("Cosine similarity (sample)")
                ax2.set_xlabel("Similarity")
                ax2.set_ylabel("Frequency")
                st.pyplot(fig2)
            else:
                st.write("Not enough embeddings to compute similarity distribution.")
        else:
            st.write("Semantic index not built or sentence-transformers missing.")

        # Admin feedback
        st.markdown("### Admin feedback")
        fb = st.text_area("Enter feedback / notes", key="admin_feedback_text")
        if st.button("Submit feedback"):
            out = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "feedback": fb}
            fb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "admin_feedback.csv"))
            if os.path.exists(fb_path):
                df_fb = pd.read_csv(fb_path)
                df_fb = pd.concat([df_fb, pd.DataFrame([out])], ignore_index=True)
            else:
                df_fb = pd.DataFrame([out])
            df_fb.to_csv(fb_path, index=False)
            st.success("Feedback saved.")
            ss.pipeline_logs.append("Admin feedback submitted.")

    # ===== PIPELINE VIEW =====
    else:
        step = ss.current_step

        # STEP 1 — Upload
        if step == 1:
            st.subheader("Step 1 — Upload Data")
            uploaded_file = st.file_uploader("Upload CSV (.csv)", type=["csv"], key="upload_csv_main")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    ss.uploaded_df = df
                    ss.pipeline_logs.append(f"Uploaded dataset with {len(df)} rows.")
                    st.success(f"Loaded {len(df)} rows.")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")

            st.markdown("---")
            st.caption("Or use local sample file if available.")
            if st.button("Load sample dataset"):
                sample_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "cross_domain_knowledge_mapping_500.csv")
                )
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                    ss.uploaded_df = df
                    ss.pipeline_logs.append("Loaded sample dataset.")
                    st.success(f"Sample loaded with {len(df)} rows.")
                    st.dataframe(df.head())
                else:
                    st.warning(f"Sample file not found at: {sample_path}")

        # STEP 2 — Preprocessing
        elif step == 2:
            st.subheader("Step 2 — Preprocessing")
            if ss.uploaded_df is None:
                st.warning("Please complete Step 1 (Upload Data) first.")
            else:
                df = ss.uploaded_df
                text_cols = [c for c in df.columns if "text" in c.lower()] or list(df.columns)
                text_col = st.selectbox(
                    "Select text column for processing",
                    options=text_cols,
                    index=0,
                    key="preprocess_text_col_main",
                )
                if st.button("Run Preprocessing"):
                    with st.spinner("Running preprocessing..."):
                        t0 = time.time()
                        df_proc = df.copy()
                        df_proc["clean_text"] = df_proc[text_col].apply(clean_text)
                        df_proc["processed_text"] = preprocess_series(df_proc["clean_text"])
                        elapsed = time.time() - t0
                    ss.preprocessed_df = df_proc
                    ss.pipeline_logs.append(f"Preprocessing completed in {elapsed:.1f}s ({len(df_proc)} rows).")
                    st.success(f"Preprocessing finished in {elapsed:.1f}s.")
                    st.dataframe(df_proc[[text_col, "processed_text"]].head(20))

        # STEP 3 — NER + RE
        elif step == 3:
            st.subheader("Step 3 — NER + Relation Extraction")
            if ss.preprocessed_df is None and ss.uploaded_df is None:
                st.warning("Please complete previous steps first.")
            else:
                if st.button("Load spaCy model (en_core_web_sm)"):
                    try:
                        nlp = safe_load_spacy_model("en_core_web_sm")
                        ss.nlp_model = nlp
                        ss.pipeline_logs.append("spaCy model en_core_web_sm loaded.")
                        st.success("spaCy model loaded.")
                    except Exception as e:
                        st.error(f"Failed to load spaCy model: {e}")

                if ss.nlp_model is None:
                    st.info("Load the spaCy model to run NER & Relation Extraction.")
                else:
                    input_df = ss.preprocessed_df if ss.preprocessed_df is not None else ss.uploaded_df
                    proc_text_col = "processed_text" if ss.preprocessed_df is not None else (
                        [c for c in input_df.columns if "text" in c.lower()][0]
                        if any("text" in c.lower() for c in input_df.columns)
                        else input_df.columns[0]
                    )
                    if st.button("Run NER & RE"):
                        with st.spinner("Running NER & RE..."):
                            t0 = time.time()
                            triples_df, stats = extract_entities_and_triples(
                                input_df, proc_text_col, ss.nlp_model
                            )
                            elapsed = time.time() - t0
                        ss.triples_df = triples_df
                        ss.last_stats = stats
                        ss.pipeline_logs.append(
                            f"NER & RE completed in {elapsed:.1f}s: {len(triples_df)} triples."
                        )
                        st.success(f"Extraction finished in {elapsed:.1f}s — {len(triples_df)} triples found.")
                        if not triples_df.empty:
                            st.markdown("#### Sample triples")
                            st.dataframe(triples_df.head(50))

                        rows_total = stats["rows_total"]
                        st.markdown("#### Coverage metrics")
                        st.markdown(f"- Rows total: **{rows_total}**")
                        st.markdown(
                            f"- Rows with ≥1 entity: **{stats['rows_with_entity']}** "
                            f"({stats['rows_with_entity']/rows_total*100:.1f}%)"
                        )
                        st.markdown(
                            f"- Rows with ≥1 triple: **{stats['rows_with_triple']}** "
                            f"({stats['rows_with_triple']/rows_total*100:.1f}%)"
                        )
                        st.markdown(f"- Total triples: **{stats['total_triples']}**")

        # STEP 4 — Knowledge Graph
        elif step == 4:
            st.subheader("Step 4 — Knowledge Graph")

            if ss.triples_df is None or ss.triples_df.empty:
                st.warning("Please complete Step 3 (NER + RE) to generate triples.")
            else:
                if st.button("Build Knowledge Graph", key="btn_build_graph_main"):
                    with st.spinner("Building graph from triples..."):
                        G = build_graph_from_triples(ss.triples_df)
                    ss.graph = G
                    ss.pipeline_logs.append(
                        f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
                    )
                    st.success(
                        f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
                    )

                if ss.graph is not None and ss.graph.number_of_nodes() > 0:
                    G = ss.graph

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nodes", G.number_of_nodes())
                    with col2:
                        st.metric("Edges", G.number_of_edges())
                    with col3:
                        degrees = dict(G.degree())
                        if degrees:
                            topk = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                            st.write("Top nodes by degree:")
                            st.dataframe(pd.DataFrame(topk, columns=["node", "degree"]))

                    st.markdown("#### Interactive Knowledge Graph")
                    graph_html = build_pyvis_html(G)
                    html(graph_html, height=600, scrolling=True)

                    # Subgraph Explorer (same behavior as your old app)
                    with st.expander("Subgraph Explorer", expanded=False):
                        min_degree = st.slider(
                            "Minimum degree filter",
                            0,
                            10,
                            1,
                            key="slider_min_degree_main",
                        )
                        sub_nodes = [n for n, d in G.degree() if d >= min_degree]
                        subG = G.subgraph(sub_nodes).copy()
                        st.write(
                            f"Subgraph nodes: {subG.number_of_nodes()} | "
                            f"edges: {subG.number_of_edges()}"
                        )
                        if subG.number_of_nodes() > 0:
                            sub_html = build_pyvis_html(subG)
                            html(sub_html, height=500, scrolling=True)

        # STEP 5 — Semantic Search
        elif step == 5:
            st.subheader("Step 5 — Semantic Search")

            base_df = ss.preprocessed_df if ss.preprocessed_df is not None else ss.uploaded_df
            if base_df is None:
                st.warning("Please upload and preprocess data first.")
            else:
                model_choice = st.selectbox(
                    "Embedding model",
                    ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                    key="embed_model_choice_main",
                )

                candidate_cols = [
                    c for c in base_df.columns if "process" in c.lower() or "text" in c.lower()
                ] or list(base_df.columns)

                search_col = st.selectbox(
                    "Text column to index",
                    candidate_cols,
                    index=0,
                    key="search_text_column_main",
                )

                if st.button("Build semantic index"):
                    texts = base_df[search_col].dropna().astype(str).tolist()
                    ss.sentence_texts = texts
                    with st.spinner("Building embeddings..."):
                        model, embeddings = cached_build_sentence_index(texts, model_choice)
                    if model is None or embeddings is None:
                        st.error("Sentence-transformers not installed or failed to load.")
                    else:
                        ss.sentence_model = model
                        ss.sentence_embeddings = embeddings
                        ss.pipeline_logs.append(f"Semantic index built with {len(texts)} texts.")
                        st.success(f"Indexed {len(texts)} texts.")

                if ss.sentence_model is not None and ss.sentence_embeddings is not None:
                    predefined_queries = [
                        "What are the key applications of machine learning?",
                        "Explain the benefits of using cloud computing.",
                        "What is the importance of data preprocessing?",
                        "How does NLP help analyze text data?",
                        "What are the common cybersecurity threats?",
                        "Explain the role of AI in automation.",
                        "Best practices for cloud migration.",
                        "What is relation extraction in NLP?",
                        "How do knowledge graphs work?",
                        "Challenges faced in data analytics.",
                    ]

                    st.markdown("#### Select predefined query or type manually")
                    query_mode = st.selectbox(
                        "Query Mode",
                        ["Type manually", "Choose from examples"],
                        key="semantic_query_mode",
                    )

                    if query_mode == "Choose from examples":
                        query = st.selectbox(
                            "Example Queries",
                            predefined_queries,
                            key="semantic_query_example_selector",
                        )
                    else:
                        query = st.text_input(
                            "Enter your query",
                            placeholder="Type something like: 'What is relation extraction?'",
                            key="semantic_query_manual_input",
                        )

                    top_k = st.slider(
                        "Number of results (Top K)",
                        min_value=1,
                        max_value=10,
                        value=5,
                        key="semantic_topk_slider",
                    )

                    if st.button("🔍 Run Semantic Search", key="semantic_search_button"):
                        if not query:
                            st.warning("Please enter or select a query.")
                        else:
                            with st.spinner("Running semantic search..."):
                                results = semantic_search_query(
                                    ss.sentence_model,
                                    ss.sentence_embeddings,
                                    ss.sentence_texts,
                                    query,
                                    top_k=top_k,
                                )
                            st.markdown("### 🔎 Search Results")
                            if not results:
                                st.info("No results found.")
                            else:
                                for idx, (matched_sentence, score) in enumerate(results, start=1):
                                    st.markdown(f"**{idx}. Score: {score:.4f}**")
                                    st.write(matched_sentence)

# ------------------ Sidebar summary ------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline Progress")
for sid, info in STEPS.items():
    status_icon = "✅" if ss.step_completed.get(sid, False) else ("🟢" if ss.current_step == sid else "⚪")
    st.sidebar.write(f"{status_icon} {sid}. {info['label']}")

st.sidebar.markdown("---")
st.sidebar.markdown("Dark Theme • spaCy • NLTK • PyVis • Sentence-Transformers • Streamlit")
