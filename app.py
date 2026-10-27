"""Obsidian — legal citation outcome classifier."""
from html import escape
from pathlib import Path
from typing import Any
import re

import joblib
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_legal_citation_pipeline.joblib"


@st.cache_resource
def load_pipeline(path: str) -> Any:
    return joblib.load(path)


def clean_text(title: str, text: str) -> str:
    return re.sub(r"\s+", " ", f"{title} {text}").strip()


st.set_page_config(page_title="Obsidian | Legal Citation Intelligence", page_icon="⚖", layout="wide")

dark = st.session_state.get("theme", "dark") == "dark"
if dark:
    bg, panel, fg, muted, line, accent = "#101110", "#171917", "#f2f0e9", "#a5a69d", "#30332e", "#c5a66f"
else:
    bg, panel, fg, muted, line, accent = "#f5f3ee", "#ffffff", "#20221e", "#666960", "#deded6", "#8c6b37"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');
:root {{ color-scheme: {'dark' if dark else 'light'}; }}
.stApp {{ background:{bg}; color:{fg}; font-family:'DM Sans',sans-serif; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stMainBlockContainer"] {{ max-width:1260px; padding-top:1.2rem; padding-bottom:4rem; }}
h1,h2,h3 {{ color:{fg}; }}
.brand {{font-size:13px;letter-spacing:.22em;text-transform:uppercase;color:{accent};font-weight:700}}
.hero {{ padding:3.2rem 0 2.4rem; border-bottom:1px solid {line}; }}
.eyebrow {{ color:{accent};letter-spacing:.18em;text-transform:uppercase;font-size:11px;font-weight:700;margin-bottom:18px; }}
.hero h1 {{ font-family:'Playfair Display',Georgia,serif;font-weight:500;letter-spacing:-.035em;font-size:clamp(42px,6vw,72px);line-height:1.03;margin:0 0 18px;max-width:760px; }}
.hero p {{color:{muted};font-size:17px;line-height:1.7;max-width:720px;margin:0;}}
.section-label {{color:{accent};font-size:11px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;margin:2.3rem 0 1rem;}}
.info-card {{background:{panel};border:1px solid {line};border-radius:12px;padding:22px 24px;min-height:118px;}}
.info-card small {{display:block;color:{muted};text-transform:uppercase;letter-spacing:.12em;font-size:10px;margin-bottom:12px}}
.info-card strong {{font-size:16px;color:{fg};font-weight:600}}
.stTextInput input,.stTextArea textarea {{background:{panel};color:{fg};border-color:{line};border-radius:8px}}
.stTextInput label,.stTextArea label {{color:{fg}!important;font-weight:600}}
.stButton button[kind="primary"],.stFormSubmitButton button {{background:{accent};color:#171714;border:0;border-radius:6px;font-weight:700;padding:.65rem 1.25rem}}
.stButton button {{border-radius:6px}}
.result {{background:{panel};border:1px solid {line};border-left:3px solid {accent};border-radius:10px;padding:22px 24px;margin:1rem 0}}
.result-kicker {{color:{muted};font-size:11px;text-transform:uppercase;letter-spacing:.16em}}
.result-title {{font-family:'Playfair Display',Georgia,serif;color:{accent};font-size:30px;margin:8px 0 0}}
.footnote {{color:{muted};font-size:12px;line-height:1.65}}
div[data-testid="stProgress"] > div > div {{background-color:{accent}}}
</style>""", unsafe_allow_html=True)

top_left, top_right = st.columns([5, 1])
with top_left:
    st.markdown('<div class="brand">◈ &nbsp; Obsidian / Citation intelligence</div>', unsafe_allow_html=True)
with top_right:
    if st.button("☼  Light" if dark else "◐  Dark", use_container_width=True):
        st.session_state.theme = "light" if dark else "dark"
        st.rerun()

st.markdown("""
<section class="hero">
  <div class="eyebrow">A focused research instrument</div>
  <h1>Read the signal<br>inside legal citations.</h1>
  <p>Classify the treatment of a cited case from its title and surrounding text. Enter an excerpt to see the model’s predicted outcome and its class probability distribution.</p>
</section>
""", unsafe_allow_html=True)

pipeline = None
model_error = None
classes: list[str] = []
classifier_name = "Unavailable"
ngram = "Unavailable"
if MODEL_PATH.is_file():
    try:
        pipeline = load_pipeline(str(MODEL_PATH))
        classes = [str(x) for x in pipeline.classes_]
        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["clf"]
        classifier_name = type(classifier.estimator).__name__
        ngram = f"{vectorizer.ngram_range[0]}–{vectorizer.ngram_range[1]} word n-grams"
    except Exception as exc:
        model_error = f"The model could not be loaded ({type(exc).__name__})."
else:
    model_error = f"Model file not found: {MODEL_PATH.name}"

if model_error:
    st.error(model_error)

st.markdown('<div class="section-label">01 &nbsp; Analyze a passage</div>', unsafe_allow_html=True)
with st.form("classify", clear_on_submit=False):
    title = st.text_input("Case title", placeholder="Add the cited case title (optional)", value=st.session_state.get("case_title", ""))
    text = st.text_area("Citation context", placeholder="Paste the sentence or passage describing how the cited case is treated…", height=190, value=st.session_state.get("case_text", ""))
    submitted = st.form_submit_button("Classify passage  →", type="primary")

if submitted:
    st.session_state.case_title, st.session_state.case_text = title, text
    combined = clean_text(title, text)
    if not combined:
        st.warning("Add a case title or citation passage before requesting a classification.")
    elif pipeline is not None:
        try:
            prediction = str(pipeline.predict([combined])[0])
            probabilities = pipeline.predict_proba([combined])[0]
            ordered = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
            st.session_state.last_result = (prediction, [(label, float(value)) for label, value in ordered])
        except Exception as exc:
            st.error(f"Prediction failed ({type(exc).__name__}). Please check the text and try again.")

if st.session_state.get("last_result"):
    prediction, probabilities = st.session_state.last_result
    st.markdown('<div class="section-label">02 &nbsp; Model assessment</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result"><div class="result-kicker">Predicted treatment</div><div class="result-title">{escape(prediction)}</div></div>', unsafe_allow_html=True)
    with st.expander("View class probabilities", expanded=True):
        for label, value in probabilities:
            st.markdown(f"**{escape(label)}** · {value:.1%}")
            st.progress(value)
    st.caption("Probabilities describe this model’s relative scores. Review the source passage and cited authority before relying on a classification.")

st.markdown('<div class="section-label">03 &nbsp; About this model</div>', unsafe_allow_html=True)
cols = st.columns(3)
for col, label, value in zip(cols, ("Classifier", "Text representation", "Outcome classes"), (classifier_name, ngram, str(len(classes)) if classes else "Unavailable")):
    with col:
        st.markdown(f'<div class="info-card"><small>{label}</small><strong>{escape(value)}</strong></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="footnote">Built from the project’s trained scikit-learn pipeline. The accompanying notebook describes a Kaggle legal citation dataset and model development; the raw dataset is not present in this app folder, so no dataset statistics or performance claims are shown here. This tool supports legal research and does not replace review by a qualified professional.</p>', unsafe_allow_html=True)
