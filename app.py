import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import features  # noqa: F401  (needed so the saved model can find add_features)
from features import RAW_FEATURES

# ---- Change these two lines to match your school ----------------------------
UNIVERSITY = "University Academic Services"
OFFICE = "Office of Student Success · Early Academic Alert System"
# -----------------------------------------------------------------------------

BASE = Path(__file__).parent
MAROON, MAROON_DARK, GOLD = "#6C1606", "#4E0F04", "#C9A227"

st.set_page_config(page_title="Early Academic Alert System", page_icon="📘", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(BASE / "model.joblib")
    meta = json.loads((BASE / "model_meta.json").read_text())
    return model, meta


model, meta = load_model()

# ---- Styling -----------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700;900&family=Source+Sans+3:wght@400;600;700&display=swap');
html, body, .stApp {{font-family:'Source Sans 3', Arial, sans-serif;}}
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"], .stDeployButton {{display:none !important;}}
.block-container {{padding-top:0 !important; max-width:1180px;}}
[data-testid="stMainBlockContainer"] {{padding-top:0 !important;}}
[data-testid="stExpander"] details {{background:#fff; border-color:#E2DAD0;}}
.stApp {{background:#F4F1EC;}}

/* top identity bar */
.topbar, .subnav {{width:100vw; margin-left:calc(50% - 50vw) !important; box-sizing:border-box;
          padding-left:max(2rem, calc(50vw - 560px)) !important; padding-right:max(2rem, calc(50vw - 560px)) !important;}}
.topbar {{background:{MAROON}; border-bottom:4px solid {GOLD}; padding-top:14px; padding-bottom:14px;
          display:flex; align-items:center; gap:16px;}}
.seal {{width:52px; height:52px; border-radius:50%; background:#fff; border:3px solid {GOLD};
        display:flex; align-items:center; justify-content:center; font-family:Merriweather,serif;
        font-weight:900; color:{MAROON}; font-size:22px; flex-shrink:0;}}
.topbar .uni {{color:#fff; font-family:Merriweather,Georgia,serif; font-weight:900; font-size:21px; line-height:1.15; letter-spacing:.3px;}}
.topbar .office {{color:#EBD9B4; font-size:14px; letter-spacing:.4px;}}
.subnav {{background:#fff; margin-bottom:22px; padding-top:9px; padding-bottom:9px; border-bottom:1px solid #E2DAD0;
          font-size:14px; color:#6b5d55;}}
.subnav b {{color:{MAROON};}}

h2.page {{font-family:Merriweather,Georgia,serif; color:#2B1A17; font-size:26px; margin:0 0 2px 0;}}
.lead {{color:#5d514b; margin-bottom:14px;}}

/* section headers inside the form */
.sec {{font-weight:700; color:{MAROON}; text-transform:uppercase; letter-spacing:1px; font-size:12.5px;
       border-bottom:1px solid #E6DED4; padding-bottom:4px; margin:14px 0 6px 0;}}
[data-testid="stForm"] {{background:#fff; border:1px solid #E2DAD0; border-radius:6px; padding:8px 22px 18px;}}
div.stFormSubmitButton > button {{background:{MAROON}; color:#fff; border:none; border-radius:4px;
       font-weight:700; letter-spacing:.5px; padding:.6rem 0;}}
div.stFormSubmitButton > button:hover {{background:{MAROON_DARK}; color:#fff;}}

/* result panel */
.panel {{background:#fff; border:1px solid #E2DAD0; border-radius:6px; overflow:hidden;}}
.panel .ph {{background:#FAF6F0; border-bottom:1px solid #E2DAD0; padding:10px 18px; font-weight:700;
             color:#2B1A17; font-size:14px; text-transform:uppercase; letter-spacing:1px;}}
.panel .pb {{padding:18px;}}
.status {{display:inline-block; padding:5px 12px; border-radius:3px; font-weight:700; font-size:13px; letter-spacing:.8px;}}
.ok {{background:#E3F1E6; color:#1D5E2E; border:1px solid #B9DCC1;}}
.watch {{background:#FBF1D9; color:#7A5A06; border:1px solid #EBD69C;}}
.risk {{background:#F8E1DD; color:{MAROON}; border:1px solid #E9B9B0;}}
.verdict {{font-family:Merriweather,Georgia,serif; font-size:24px; color:#2B1A17; margin:10px 0 2px;}}
.prob {{font-size:42px; font-weight:700; color:{MAROON}; line-height:1;}}
.meter {{height:10px; background:#EFE7DD; border-radius:2px; margin:8px 0 4px; position:relative;}}
.meter span {{display:block; height:100%; background:{MAROON}; border-radius:2px;}}
.meter i {{position:absolute; left:50%; top:-4px; height:18px; border-left:2px dashed #8a7d74;}}
.scale {{display:flex; justify-content:space-between; font-size:12px; color:#8a7d74;}}
.rec {{background:#FAF6F0; border-left:4px solid {GOLD}; padding:10px 14px; margin-top:14px; font-size:14.5px; color:#3d302a;}}
.kv {{font-size:13.5px; color:#5d514b; margin-top:12px; line-height:1.7;}}
.kv b {{color:#2B1A17;}}
.empty {{color:#8a7d74; text-align:center; padding:40px 10px; font-size:15px;}}
.stat {{display:flex; gap:10px; margin:12px 0 14px;}}
.stat div {{flex:1; background:#fff; border:1px solid #E2DAD0; border-radius:6px; padding:10px 12px;}}
.stat small {{color:#6b5d55; font-size:12px; text-transform:uppercase; letter-spacing:.8px;}}
.stat p {{margin:0; font-size:22px; font-weight:700; color:{MAROON};}}
.foot {{text-align:center; color:#8a7d74; font-size:12.5px; border-top:1px solid #E2DAD0; margin-top:30px; padding:14px 0 6px;}}
</style>
""", unsafe_allow_html=True)

# ---- Header ------------------------------------------------------------------
initials = "".join(w[0] for w in UNIVERSITY.split() if w[0].isupper())[:2]
st.markdown(f"""
<div class="topbar">
  <div class="seal">{initials}</div>
  <div><div class="uni">{UNIVERSITY}</div><div class="office">{OFFICE}</div></div>
</div>
<div class="subnav">Advising Tools &nbsp;›&nbsp; <b>Pass/Fail Risk Assessment</b>
  <span style="float:right">{date.today():%B %d, %Y}</span></div>
""", unsafe_allow_html=True)

st.markdown('<h2 class="page">Student Pass/Fail Risk Assessment</h2>'
            '<div class="lead">Enter the student\'s learning management system (LMS) activity for the term. '
            'The system estimates whether the student is likely to pass or is at risk of failing.</div>',
            unsafe_allow_html=True)

LABELS = {
    "Submission.Actions": ("Submissions", "Number of assignment submission actions"),
    "Quiz.Actions": ("Quiz activity", "Quiz attempts and quiz interactions"),
    "Forum.Actions": ("Forum activity", "Posts, replies and views in discussion forums"),
    "Create.Actions": ("Content created", "Times the student created content"),
    "Read.Actions": ("Materials viewed", "Times the student opened or read course materials"),
    "Update.Actions": ("Content edited", "Times the student edited content"),
    "Delete.Actions": ("Content deleted", "Times the student deleted content"),
    "Days.Logged.In": ("Days logged in", "Distinct days the student logged in to the LMS"),
}
SECTIONS = [
    ("Coursework", ["Submission.Actions", "Quiz.Actions"]),
    ("Class participation", ["Forum.Actions", "Read.Actions"]),
    ("Content activity", ["Create.Actions", "Update.Actions", "Delete.Actions"]),
    ("Attendance", ["Days.Logged.In"]),
]

left, right = st.columns([1.35, 1], gap="large")

with left:
    with st.form("assessment"):
        st.markdown('<div class="sec">Student information</div>', unsafe_allow_html=True)
        a, b = st.columns([1, 2])
        sid = a.text_input("Student ID", placeholder="e.g. 2023-00123")
        name = b.text_input("Student name", placeholder="Last name, First name")
        values = {}
        for title, cols in SECTIONS:
            st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)
            grid = st.columns(3)
            for i, c in enumerate(cols):
                label, help_text = LABELS[c]
                values[c] = grid[i].number_input(label, min_value=0, step=1,
                                                 value=int(meta["medians"][c]), help=help_text)
        submitted = st.form_submit_button("Run assessment", use_container_width=True)

with right:
    if submitted:
        X = pd.DataFrame([values])[RAW_FEATURES]
        p = float(model.predict_proba(X)[0][1])
        if p >= 0.70:
            cls, tag, verdict = "ok", "ON TRACK", "Likely to pass"
            rec = "No intervention needed. Continue regular monitoring through the term."
        elif p >= 0.50:
            cls, tag, verdict = "watch", "MONITOR", "Likely to pass, with low confidence"
            rec = "Recommend a brief check-in with the class adviser and a follow-up review mid-term."
        else:
            cls, tag, verdict = "risk", "AT RISK", "At risk of failing"
            rec = ("Refer the student to their academic adviser for a consultation, and consider "
                   "tutoring or a learning support plan.")
        who = (f"<b>{name}</b>" if name else "Unnamed student") + (f" &nbsp;·&nbsp; {sid}" if sid else "")
        st.markdown(f"""
<div class="panel"><div class="ph">Assessment result</div><div class="pb">
  <div style="color:#5d514b;font-size:14px">{who}</div>
  <div class="verdict">{verdict}</div>
  <span class="status {cls}">{tag}</span>
  <div style="margin-top:16px;color:#6b5d55;font-size:13px;text-transform:uppercase;letter-spacing:.8px">Estimated probability of passing</div>
  <div class="prob">{p:.0%}</div>
  <div class="meter"><span style="width:{p*100:.1f}%"></span><i></i></div>
  <div class="scale"><span>0%</span><span>50% cut-off</span><span>100%</span></div>
  <div class="rec"><b>Recommended action:</b> {rec}</div>
  <div class="kv">Assessed on <b>{date.today():%B %d, %Y}</b> &nbsp;·&nbsp; Days logged in: <b>{values['Days.Logged.In']}</b>
  &nbsp;·&nbsp; Submissions: <b>{values['Submission.Actions']}</b></div>
</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel"><div class="ph">Assessment result</div>'
                    '<div class="empty">Fill in the student\'s activity and click <b>Run assessment</b>.'
                    '<br>The result will appear here.</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
<div class="stat">
  <div><small>Test accuracy</small><p>{meta['test_accuracy']:.2%}</p></div>
  <div><small>Validation accuracy</small><p>{meta['cv_accuracy']:.2%}</p></div>
</div>""", unsafe_allow_html=True)

    with st.expander("About this model"):
        st.markdown(
            f"""
- **Algorithm:** {meta['model']} classifier with engineered features (activity per day logged in,
  total activity, share of reading). It beat Logistic Regression, Gradient Boosting and Random Forest.
- **Training data:** {meta['n_rows']:,} student records (50% High / 50% Low rating).
- **Validation accuracy (5-fold cross-validation):** {meta['cv_accuracy']:.2%}
- **Test accuracy (80/20 split):** {meta['test_accuracy']:.2%}
- **Accuracy on students never seen in training:** {meta['unseen_accuracy']:.1%}
  (the dataset has duplicate rows, so this is the stricter figure).
"""
        )
        imp = pd.Series(meta["importances"]).rename(lambda c: LABELS[c][0]).sort_values()
        st.caption("Which inputs matter most")
        st.bar_chart(imp, horizontal=True, color=MAROON, height=260)

st.markdown(f'<div class="foot">{UNIVERSITY} · {OFFICE}<br>'
            'Predictions support, and do not replace, the judgment of advisers and faculty.</div>',
            unsafe_allow_html=True)