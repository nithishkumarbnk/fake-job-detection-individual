# modules/ner_relation.py

from typing import Tuple, Dict
import pandas as pd
import streamlit as st

try:
    import spacy
except Exception:
    spacy = None


@st.cache_resource(show_spinner=False)
def safe_load_spacy_model(model_name: str = "en_core_web_sm"):
    """Safely load a spaCy model with caching (best for Streamlit Cloud)."""
    global spacy
    if spacy is None:
        raise ImportError("spaCy is not installed. Please add 'spacy' to requirements.txt")

    try:
        return spacy.load(model_name)
    except Exception:
        # Download only ONCE; Streamlit cache ensures no repeated downloads
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)


def extract_entities_and_triples(
    df: pd.DataFrame,
    text_col: str,
    nlp_model
) -> Tuple[pd.DataFrame, Dict]:

    triples_rows = []
    entity_counts: Dict[str, int] = {}
    rows_with_entity = 0
    rows_with_triple = 0

    for idx, row in df.iterrows():
        text = str(row.get(text_col, "")).strip()

        if not text:
            continue

        doc = nlp_model(text)

        # ---------- Extract Entities ----------
        ents_in_row = []
        for ent in doc.ents:
            label = ent.label_
            entity_counts[label] = entity_counts.get(label, 0) + 1
            ents_in_row.append((ent.text, ent.label_))

        if ents_in_row:
            rows_with_entity += 1

        # ---------- Extract Triples ----------
        triples_for_row = []
        for sent in doc.sents:
            subject = None
            relation = None
            object_ = None

            for token in sent:
                dep = token.dep_.lower()

                if "subj" in dep:
                    subject = token.text
                if "obj" in dep:
                    object_ = token.text
                if token.dep_ == "ROOT":
                    relation = token.lemma_

            if subject and relation and object_:
                triples_for_row.append((subject, relation, object_))

        if triples_for_row:
            rows_with_triple += 1
            for (s, r, o) in triples_for_row:
                triples_rows.append({
                    "subject": s,
                    "relation": r,
                    "object": o,
                    "source_text": text,
                    "row_id": idx
                })

    # ---------- Final Data ----------
    triples_df = pd.DataFrame(triples_rows)

    stats = {
        "entity_counts": entity_counts,
        "rows_total": len(df),
        "rows_with_entity": rows_with_entity,
        "rows_with_triple": rows_with_triple,
        "avg_entities_per_row": (sum(entity_counts.values()) / len(df)) if len(df) > 0 else 0,
        "total_triples": len(triples_df),
    }

    return triples_df, stats
