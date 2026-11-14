"""MilkWise Touch — tablet-style farm milk monitoring demo.

This interface uses historical production models, not a veterinary classifier.
Color only indicates how an ACTUAL measured milk yield differs from a model estimate.
"""
from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from milk_core import (
    BLOOD, CATEGORICAL, FEED, FEATURES, ID, MILK_OUTPUTS, OUTPUT_DETAILS,
    PHYSIOLOGY, TARGET, DataProblem, available_features, classify_deviation,
    load_study_workbooks, norm_genetics, norm_id, norm_thi, predict_one,
    read_farm_upload, template_csv_bytes, validate_and_train,
)

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="MilkWise · Simple cow checks", page_icon="🐄", layout="centered", initial_sidebar_state="collapsed")

HELP = {
    "THI": "Temperature-Humidity Index. T0 is ≤75, T1 is 75–80, and T2 is ≥80. Use the weather condition recorded for the cow's measurement period.",
    "Genetic group": "The recorded Holstein-Friesian genetic group, not a health rating.",
    "Rectal temp (F)": "Cow's internal body temperature in degrees Fahrenheit, from a measured reading.",
    "Pulse (bpm)": "Measured heartbeats per minute.",
    "Respiration (bpm)": "Measured breaths per minute.",
    "DMI (kg)": "Dry matter intake: feed consumed excluding water, in kilograms per day.",
    "Glucose (mmol/L)": "Measured blood glucose level in millimoles per liter.",
    "Total Protein (g/dL)": "Total serum protein measured in grams per deciliter.",
    "Uric Acid (mg/dL)": "Measured serum uric acid in milligrams per deciliter.",
    "Cholesterol (mg/dL)": "Measured total serum cholesterol in milligrams per deciliter.",
    "Calcium (mg/dL)": "Measured serum calcium in milligrams per deciliter.",
    "HDL (mg/dL)": "Measured high-density lipoprotein in milligrams per deciliter.",
    "AST (U/I)": "AST enzyme activity, expressed in units per liter in the study.",
    "ALT (U/I)": "ALT enzyme activity, expressed in units per liter in the study.",
    "Cortisol (µg/dL)": "Measured cortisol in micrograms per deciliter.",
}
GROUPS = {
    "Body readings": PHYSIOLOGY,
    "Feed intake": FEED,
    "Blood tests": BLOOD,
}
STATUS_COLORS = {
    "GREEN": ("#14824b", "LOOKS EXPECTED", "Production is within the usual prediction variation.", "✓"),
    "YELLOW": ("#eab54b", "CHECK THE RECORD", "This production differs more than usual. Check the measurements and feeding records.", "!"),
    "RED": ("#bd3440", "REVIEW THIS COW", "Milk production is much lower than expected. Prioritize a production check.", "!"),
    "NEUTRAL": ("#295b91", "ESTIMATE READY", "Add the actual milk volume later to see a production check.", "✓"),
}

if "page" not in st.session_state:
    st.session_state.page = "Check cow"
if "source" not in st.session_state:
    st.session_state.source = "Demo"
if "farm_upload" not in st.session_state:
    st.session_state.farm_upload = None
if "approved" not in st.session_state:
    st.session_state.approved = {}
if "pending" not in st.session_state:
    st.session_state.pending = {}
if "result" not in st.session_state:
    st.session_state.result = None
if "algorithm" not in st.session_state:
    st.session_state.algorithm = "Ridge"


@st.cache_data(show_spinner=False)
def load_demo():
    return load_study_workbooks(HERE / "data")


@st.cache_data(show_spinner=False)
def load_uploaded(name: str, content: bytes):
    return read_farm_upload(name, content)


@st.cache_resource(show_spinner=False)
def fit_model(fingerprint: str, input_features: tuple, outcome: str, algo: str, _df):
    return validate_and_train(_df, input_features, algorithm=algo, target=outcome)


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False, float_format="%.12g").encode()).hexdigest()


def train_for(df, features, outcome):
    return fit_model(frame_hash(df), tuple(features), outcome, st.session_state.algorithm, df)


def set_page(name):
    st.session_state.page = name
    st.session_state.result = None


def switch_to_form():
    st.session_state.result = None


def read_source():
    metadata = None
    if st.session_state.source == "Demo":
        df, metadata = load_demo()
        key = "demo_v2"
    else:
        upload = st.session_state.farm_upload
        if upload is None:
            return None, None, "farm_empty"
        key = "farm_" + hashlib.sha256(upload["content"]).hexdigest()
        df = load_uploaded(upload["name"], upload["content"])
    appended = st.session_state.approved.get(key, [])
    if appended:
        df = pd.concat([df, pd.DataFrame(appended)], ignore_index=True)
    return df, metadata, key


def render_css(status: str | None):
    # The entire application background changes color, but never calls this a disease diagnosis.
    if status == "RED":
        bg = "#bd3440"
    elif status == "YELLOW":
        bg = "#eab54b"
    elif status == "GREEN":
        bg = "#14824b"
    elif status == "NEUTRAL":
        bg = "#295b91"
    else:
        bg = "#f6f8f4"
    is_result = status is not None
    fg = "#2b2a17" if status == "YELLOW" else "#ffffff"
    st.markdown(f"""
    <style>
       :root {{ --milk-bg: {bg}; --milk-accent:#155c46; }}
       [data-testid="stAppViewContainer"] {{ background:{bg} !important; transition: background .28s ease; }}
       [data-testid="stHeader"] {{ background:transparent !important; }}
       #MainMenu, footer, [data-testid="stToolbar"] {{visibility:hidden;}}
       .block-container {{max-width:850px; padding-top:1rem; padding-bottom:4rem;}}
       h1,h2,h3 {{letter-spacing:-.04em;}}
       h1 {{font-size:clamp(2rem,5vw,3.4rem) !important; font-weight:800 !important;}}
       .tagline {{color:#6c8275; font-size:1.05rem; margin-top:-6px;}}
       .pill {{display:inline-block; padding:.35rem .8rem; background:#eaf3ec; border-radius:999px; color:#155c46; font-size:.85rem; font-weight:700;}}
       .mini-card {{background:white; border:1px solid #e7eee8; border-radius:24px; padding:22px; margin:10px 0; box-shadow:0px 8px 20px rgba(19,53,42,.045);}}
       .mini-card h3 {{font-size:1.15rem !important; margin:0 0 .4rem;}}
       .mini-card p {{color:#52645b; margin:0;}}
       .quiet {{color:#6f8276; font-size:.92rem;}}
       .touch-intro {{font-size:1.17rem; color:#415749; line-height:1.55;}}
       [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button,
       [data-testid="stDownloadButton"] button {{border-radius:18px !important; min-height:3.4rem; font-size:1.04rem; font-weight:650;}}
       [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{border-radius:15px; font-size:1.07rem; min-height:3rem;}}
       div[data-baseweb="select"] > div {{border-radius:15px; min-height:3rem;}}
       [data-testid="stMetric"] {{background:white; border:1px solid #e8eee9; border-radius:20px; padding:15px 18px;}}
       [data-testid="stMetricValue"] {{font-size:1.8rem;}}
       [data-testid="stExpander"] {{border-radius:18px; border:1px solid #dae6dc; overflow:hidden; background:rgba(255,255,255,.75);}}
       .result-title {{font-size:clamp(2.4rem,8vw,4.3rem); line-height:1.06; font-weight:900; color:{fg}; margin:0.4rem 0 0.8rem;}}
       .result-eyebrow {{text-transform:uppercase;letter-spacing:.16em; font-weight:750; font-size:.9rem; color:{fg};}}
       .result-desc {{font-size:1.25rem; line-height:1.4; color:{fg}; max-width:630px;}}
       .result-number {{font-size:clamp(2.6rem,9vw,5.4rem);font-weight:850;color:{fg};line-height:1.06;}}
       .result-box {{border-radius:28px;background:rgba(255,255,255,.17); padding:23px 26px; border:1px solid rgba(255,255,255,.34);margin:1.5rem 0;}}
       .result-box .note {{color:{fg};font-size:1.02rem;}}
       .result-foot {{color:{fg};font-size:.9rem;}}
       .app-mark {{font-size:1.35rem; font-weight:900; letter-spacing:-.04em; color:{fg if is_result else '#193c2d'};}}
       .app-sub {{font-size:.8rem;color:{fg if is_result else '#637769'};}}
       {'[data-testid="stAppViewContainer"] .stCaption {color:'+fg+' !important;}' if is_result else ''}
       @media (max-width:650px) {{
         .block-container {{padding-left:1rem;padding-right:1rem;}}
         .mini-card {{padding:16px;}}
         [data-testid="stButton"] button {{padding:.55rem;}}
       }}
    </style>""", unsafe_allow_html=True)


def app_header(result_mode: bool):
    st.markdown('<div class="app-mark">🐄 MilkWise <span class="app-sub">/ Simple farm checks</span></div>', unsafe_allow_html=True)
    if not result_mode:
        cols = st.columns(3, gap="small")
        for col, page, text in zip(cols, ["Check cow", "Sensors", "Farm data"], ["🐄 Check", "📊 Sensors", "📁 Data"]):
            with col:
                if st.button(text, use_container_width=True, type="primary" if st.session_state.page == page else "secondary", key=f"nav_{page}"):
                    set_page(page)
                    st.rerun()
        st.write("")


def status_for(actual: float | None, estimate: float, fit):
    if actual is None:
        return "NEUTRAL", None
    code, _, difference = classify_deviation(actual, estimate, fit)
    # An unusually HIGH production value warrants a record check, not a red welfare alert.
    if difference > fit.q95:
        return "YELLOW", difference
    return code, difference


def quality_cards(predictions: dict, errors: dict):
    if len(predictions) <= 1:
        return
    st.markdown("### Milk details")
    st.caption("Estimated composition, not lab measurements or complete nutritional information.")
    preferred = ["Fat (%)", "Protein (%)", "Lactose (%)", "Solids-not-fat (%)"]
    cols = st.columns(2, gap="small")
    for i, key in enumerate(preferred):
        if key in predictions:
            with cols[i % 2]:
                st.metric(OUTPUT_DETAILS[key][0], f"{predictions[key]:.2f}%", help=OUTPUT_DETAILS[key][2])
    with st.expander("More milk details  ❔"):
        for key in ("Salt (%)", "Milk pH", "Somatic cell count (cells/mL)"):
            if key not in predictions:
                continue
            if key == "Somatic cell count (cells/mL)":
                value = f"{predictions[key]:,.0f} cells/mL"
            elif key == "Milk pH":
                value = f"{predictions[key]:.2f} pH"
            else:
                value = f"{predictions[key]:.2f}%"
            st.metric(OUTPUT_DETAILS[key][0], value, help=OUTPUT_DETAILS[key][2])
        st.caption("Milk composition and quality measurements have their own uncertainty. See model limitations in Farm data.")


def result_screen():
    report = st.session_state.result
    status = report["status"]
    _, title, msg, symbol = STATUS_COLORS[status]
    app_header(True)
    cow_label = escape(report.get("cow") or "Cow check")
    st.markdown(f'<div style="height:10px"></div><div class="result-eyebrow">{symbol} &nbsp; {cow_label} · milk production</div>'
                f'<div class="result-title">{title}</div><div class="result-desc">{msg}</div>', unsafe_allow_html=True)
    actual = report.get("actual")
    pred = report["pred"]
    width = report["fit"].q90
    low, high = max(0.0, pred - width), pred + width
    st.markdown(f'''<div class="result-box">
        <div class="result-eyebrow">Estimated daily milk yield</div>
        <div class="result-number">{pred:.2f} <span style="font-size:.38em">L/day</span></div>
        <div class="note" style="margin-top:12px">Typical historical-error range: <b>{low:.2f}–{high:.2f} L/day</b></div>
        {f'<div class="note" style="margin-top:8px">Measured milk: <b>{actual:.2f} L/day</b> · Difference: <b>{report["delta"]:+.2f} L/day</b></div>' if actual is not None else ''}
    </div>''', unsafe_allow_html=True)
    st.markdown('<div class="result-foot">This color compares milk output with a statistical estimate. It does NOT diagnose disease or guarantee a cow is healthy. Historical-error ranges are not guaranteed individual prediction intervals.</div>', unsafe_allow_html=True)
    if st.button("← Check another cow", use_container_width=True, type="primary"):
        switch_to_form()
        st.rerun()
    if report["source"] == "My farm" and actual is not None and report.get("cow"):
        if st.button("＋ Save measurement for review", use_container_width=True):
            candidate = {ID: norm_id(report["cow"]), "Observation date": report["date"], TARGET: float(actual)}
            candidate.update(report["measurements"])
            st.session_state.pending.setdefault(report["key"], []).append(candidate)
            st.session_state.result = None
            st.session_state.page = "Farm data"
            st.rerun()
        st.caption("Review is required before new records are used to retrain the model.")
    if report.get("predictions"):
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        with st.expander("🥛 Milk fat, protein and more  ❔", expanded=False):
            quality_cards(report["predictions"], report["errors"])
    if report.get("failures"):
        with st.expander("Other predictions not available"):
            for name, why in report["failures"].items():
                st.write(f"{OUTPUT_DETAILS[name][0]}: {why}")


def check_page(frame, key, metadata):
    st.markdown("# Check a cow")
    st.markdown('<div class="touch-intro">Enter what you know. We’ll estimate today’s recorded-period milk production and flag unusual differences <b>only if you also know the actual yield</b>.</div>', unsafe_allow_html=True)
    if frame is None:
        st.info("Upload a farm history to begin, or switch back to the hackathon example.")
        if st.button("Go to farm data", type="primary", use_container_width=True):
            set_page("Farm data"); st.rerun()
        return
    feats = available_features(frame)
    groups = frame["Genetic group"].dropna().astype(str).unique().tolist() if "Genetic group" in frame else []
    thi_values = frame["THI"].dropna().astype(str).unique().tolist() if "THI" in frame else []
    st.markdown("<div class='pill'>Step 1 · Cow details</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1], gap="medium")
    with col_a:
        cow_id = st.text_input("Cow ID (optional)", key="cow_id_entry", placeholder="e.g. 102", help="Identifies the cow in a farm's records. It is NOT used as a predictor.")
    with col_b:
        sampled = st.date_input("Measurement date", value=date.today(), key="measure_date", help="Used only for saved history records, not for the prediction.")
    readings = {}
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        if "THI" in feats:
            opts = sorted(thi_values, key=lambda x: ["T0", "T1", "T2"].index(x) if x in ["T0", "T1", "T2"] else 99)
            readings["THI"] = st.selectbox("Heat level ❔", opts, help=HELP["THI"])
    with col_b:
        if "Genetic group" in feats:
            readings["Genetic group"] = st.selectbox("Cow breed group ❔", sorted(groups), help=HELP["Genetic group"])
    st.markdown("<div class='pill' style='margin-top:12px'>Step 2 · Optional sensor readings</div>", unsafe_allow_html=True)
    st.caption("You do NOT need every sensor. Only select measurements you actually collected.")
    with st.expander("＋ Add sensor readings", expanded=False):
        valid_extras = [f for f in FEATURES if f in feats and f not in CATEGORICAL]
        extras = st.multiselect("Which other measurements do you have?", valid_extras, default=[],
                                help="Choose one, several, or all available readings. The app fits a model for exactly this set of measurements.", key="extra_readings")
        if extras:
            c1, c2 = st.columns(2, gap="medium")
            for i, f in enumerate(extras):
                with [c1,c2][i%2]:
                    readings[f] = st.number_input(f, value=None, min_value=0.0, format="%.3f", placeholder="Measured value", help=HELP.get(f), key="reading_"+f)
        else:
            st.caption("That's fine. We'll use the basic information above.")
    st.markdown("<div class='pill' style='margin-top:12px'>Step 3 · Check production</div>", unsafe_allow_html=True)
    actual = st.number_input("Actual milk produced (optional)", min_value=0.0, value=None, format="%.2f", key="actual_milk",
                             help="Enter the measured liters per day to compare with expected yield. Leave empty for a prediction only. Never enter a model estimate here.")
    full_quality = st.toggle("Also estimate milk fat, protein & quality", value=True, help="Predicts the composition variables actually recorded in your historical dataset. More results take additional computation.")
    if st.button("CHECK COW", type="primary", use_container_width=True):
        selected = tuple(f for f in FEATURES if f in readings)
        absent = [f for f in selected if readings[f] is None]
        if not selected:
            st.error("Choose at least one measurement to continue.")
            return
        if absent:
            st.error("Enter the readings you selected, or deselect the sensors you don't have: " + ", ".join(absent))
            return
        if actual is not None and not np.isfinite(float(actual)):
            st.error("Enter a valid measured milk yield.")
            return
        with st.spinner("Checking similar cows…"):
            try:
                fit = train_for(frame, selected, TARGET)
                pred = predict_one(fit, readings)
            except (DataProblem, ValueError) as exc:
                st.error("Couldn't calculate this estimate: " + str(exc))
                return
            status, delta = status_for(actual, pred, fit)
            predictions, errors, failures = {}, {}, {}
            if full_quality:
                for outcome in MILK_OUTPUTS[1:]:
                    if outcome not in frame or frame[outcome].notna().sum() == 0:
                        failures[outcome] = "No measured historical values for this outcome"
                        continue
                    try:
                        trained = train_for(frame, selected, outcome)
                        predictions[outcome] = predict_one(trained, readings)
                        errors[outcome] = trained
                    except (DataProblem, ValueError) as exc:
                        failures[outcome] = str(exc)
        st.session_state.result = {
            "source": st.session_state.source, "key": key, "cow": cow_id.strip(),
            "date": str(sampled), "measurements": dict(readings), "actual": actual,
            "pred": pred, "fit": fit, "status": status, "delta": delta,
            "predictions": predictions, "errors": errors, "failures": failures,
        }
        st.rerun()
    if metadata:
        with st.expander("About the hackathon example data  ❔"):
            st.caption(metadata["source_warning"])


def sensor_page(frame):
    st.markdown("# Which sensors help?")
    st.markdown('<div class="touch-intro">See how much the prediction changes when you have fewer measurements. <b>Smaller error is better.</b></div>', unsafe_allow_html=True)
    if frame is None:
        st.info("Start with the hackathon example or upload farm history in Farm data.")
        return
    outcomes = [o for o in MILK_OUTPUTS if o in frame and frame[o].notna().any()]
    outcome = st.selectbox("What are we estimating? ❔", outcomes, format_func=lambda o: OUTPUT_DETAILS[o][0],
                           help="Different milk results have different units. Compare errors only for the same outcome.")
    one_by_one = st.toggle("Compare single sensors too", value=False,
                           help="Checks each measurement separately. This can take longer than the basic comparison.")
    if st.button("COMPARE SENSOR SETUPS", type="primary", use_container_width=True):
        options = [("THI + breed", CATEGORICAL),
                   ("+ Body sensors", CATEGORICAL + PHYSIOLOGY),
                   ("+ Feed intake", CATEGORICAL + FEED),
                   ("+ Blood tests", CATEGORICAL + BLOOD),
                   ("All measurements", FEATURES)]
        if one_by_one:
            options.extend([(f"Only {name}", (name,)) for name in available_features(frame)])
        unique = {}
        for name, members in options:
            chosen = tuple(f for f in members if f in available_features(frame))
            if chosen:
                unique.setdefault(chosen, name)
        all_fields = tuple(dict.fromkeys(x for fs in unique for x in fs))
        comparable = frame.dropna(subset=[outcome, *all_fields]).reset_index(drop=True)
        if comparable[ID].nunique() < 3 or len(comparable) < 15:
            st.error("Not enough cows with these measurements for a fair sensor comparison. Try uploading a more complete history.")
        else:
            results = []
            bar = st.progress(0, text="Comparing on held-out cows…")
            for idx, (fields, name) in enumerate(unique.items()):
                try:
                    fit = train_for(comparable, fields, outcome)
                    results.append({"Sensors": name, "Avg. error": fit.mae,
                                    "Usual error width": fit.q90, "Measurements": len(fields)})
                except (DataProblem, ValueError):
                    pass
                bar.progress((idx+1)/len(unique))
            bar.empty()
            st.session_state.comparison = (frame_hash(frame), outcome, one_by_one, pd.DataFrame(results))
    cache = st.session_state.get("comparison")
    if cache and cache[:3] == (frame_hash(frame), outcome, one_by_one) and not cache[3].empty:
        table = cache[3].sort_values("Avg. error").reset_index(drop=True)
        best = table.iloc[0]
        unit = OUTPUT_DETAILS[outcome][1]
        st.markdown(f'<div class="mini-card"><h3>Best performing setup: {escape(str(best["Sensors"]))}</h3>'
                    f'<p>Average historical error: <b>{best["Avg. error"]:.2f} {escape(unit)}</b></p></div>', unsafe_allow_html=True)
        st.bar_chart(table.set_index("Sensors")["Avg. error"], horizontal=True)
        with st.expander("See the exact results  ❔"):
            st.dataframe(table.round(3), use_container_width=True, hide_index=True)
            st.caption("All setups use the same complete observations and cow-grouped validation. These are historical average errors, not guaranteed errors on a new farm.")
    else:
        st.markdown('<div class="mini-card"><h3>How this helps</h3><p>Find out whether an extra sensor actually improves your farm’s predictions before buying or using it.</p></div>', unsafe_allow_html=True)


def verify_new_record(rec, history, previous):
    cow = norm_id(rec.get(ID))
    d = rec.get("Observation date")
    actual = pd.to_numeric(rec.get(TARGET), errors="coerce")
    if not cow or cow.lower() == "nan":
        raise DataProblem("A cow ID is required to save a real observation.")
    if not d:
        raise DataProblem("An observation date is required.")
    if not np.isfinite(actual) or actual < 0:
        raise DataProblem("Enter a real measured milk yield, not the AI's prediction.")
    if "Observation date" in history.columns:
        existed = ((history[ID].astype(str) == cow) & (history["Observation date"].astype(str) == d))
        if existed.any():
            raise DataProblem("That cow and date already exist in your farm history.")
    for item in previous:
        if norm_id(item.get(ID)) == cow and item.get("Observation date") == d:
            raise DataProblem("That cow and date are already waiting for approval.")


def data_page(frame, meta, key):
    st.markdown("# My farm data")
    st.markdown('<div class="touch-intro">Your farm’s real measurements make its own prediction model possible. New records are <b>reviewed first</b>—never silently used for training.</div>', unsafe_allow_html=True)
    choice = st.radio("Training data", ["Demo", "My farm"], index=0 if st.session_state.source == "Demo" else 1,
                      horizontal=True, help="Demo uses the organizer study. My farm trains on your own uploaded historical measurements.")
    if choice != st.session_state.source:
        st.session_state.source = choice
        st.session_state.result = None
        st.rerun()
    if choice == "My farm":
        file = st.file_uploader("Upload farm history (CSV or Excel)", type=["csv", "xlsx"],
                                help="One row per observation, with Cow ID, real milk yield, and the measurements available then.")
        if file is not None:
            content = file.getvalue()
            new_id = hashlib.sha256(content).hexdigest()
            old = st.session_state.farm_upload
            if old is None or hashlib.sha256(old["content"]).hexdigest() != new_id:
                try:
                    load_uploaded(file.name, content)
                except (DataProblem, ValueError) as exc:
                    st.error("Please check the uploaded file: " + str(exc))
                else:
                    st.session_state.farm_upload = {"name": file.name, "content": content}
                    st.success("Farm history imported. You can now check cows using these records.")
                    st.rerun()
        if st.session_state.farm_upload:
            st.caption("Using: " + st.session_state.farm_upload["name"])
        st.download_button("↓ Download blank history template", template_csv_bytes(),
                           file_name="milkwise_farm_history_template.csv", mime="text/csv", use_container_width=True)
    if frame is None:
        st.info("Upload your farm history here, or select Demo to explore the app.")
        return
    m1, m2 = st.columns(2, gap="small")
    m1.metric("Historical readings", f"{len(frame):,}")
    m2.metric("Cows represented", f"{frame[ID].nunique():,}")
    if meta:
        with st.expander("About the demo dataset  ❔"):
            st.warning(meta["source_warning"])
            st.caption("The three original workbooks each contain 750 observations; 675 have matching recorded IDs. A systematic discrepancy in the physiology IDs is not yet resolved.")
    with st.expander("View the dataset"):
        st.dataframe(frame.head(20), use_container_width=True)
    if choice == "My farm":
        st.markdown("### New measurements to review")
        pending = st.session_state.pending.setdefault(key, [])
        if pending:
            st.dataframe(pd.DataFrame(pending), hide_index=True, use_container_width=True)
            confirmed = st.checkbox("I checked cow IDs, dates, units, and actual measured milk yield.",
                                    help="Never approve predicted milk results, readings from different dates, duplicates, or uncertain cow IDs.")
            if st.button("APPROVE & UPDATE MODEL", type="primary", use_container_width=True, disabled=not confirmed):
                accepted = []
                remaining = []
                fresh = frame.copy()
                for rec in pending:
                    try:
                        verify_new_record(rec, fresh, accepted)
                        accepted.append(rec)
                        fresh = pd.concat([fresh, pd.DataFrame([rec])], ignore_index=True)
                    except DataProblem as exc:
                        remaining.append(rec)
                        st.warning(str(exc))
                if accepted:
                    st.session_state.approved.setdefault(key, []).extend(accepted)
                    st.success(f"Approved {len(accepted)} real observation(s). Future predictions use updated history.")
                st.session_state.pending[key] = remaining
                st.rerun()
            if st.button("Discard unapproved readings", use_container_width=True):
                st.session_state.pending[key] = []
                st.rerun()
        else:
            st.info("Nothing to review. Check a cow and enter its measured milk yield to queue a new observation.")
        st.download_button("↓ Save updated history to this device", frame.to_csv(index=False),
                           file_name="milkwise_my_farm_history.csv", mime="text/csv", use_container_width=True)
        st.caption("Approved records are stored only in this browser session. Download the updated history to keep them after closing the app.")
    with st.expander("When should I add new measurements?  ❔"):
        st.markdown("**Add:** Real milk yield measured for a clearly identified cow, with correct dates, units and readings from the same period.\n\n"
                    "**Don't add:** An AI prediction, a measurement from a different cow/date, or a duplicate.\n\n"
                    "**Retrain:** After reviewing a useful batch of new real observations. More data may help, but accuracy must be checked on held-out cows or future observations.")
    with st.expander("Advanced settings / limitations  ❔"):
        st.selectbox("Regression method", ["Ridge", "Random Forest"], key="algorithm",
                     help="Ridge is simpler and usually faster. Random Forest can learn nonlinear relationships. Compare accuracy before choosing.")
        st.caption("Demonstration prototype, not a veterinary diagnostic device. Results describe the current observation period; they do not establish future milk production or causal effects. The old study may not represent other farms.")


try:
    frame, meta, key = read_source()
    problem = None
except (FileNotFoundError, DataProblem, ValueError) as exc:
    frame, meta, key = None, None, "invalid"
    problem = str(exc)
result_mode = st.session_state.result is not None and st.session_state.page == "Check cow"
render_css(st.session_state.result["status"] if result_mode else None)
if problem:
    st.error("Could not load these farm records: " + problem)
if result_mode:
    result_screen()
else:
    app_header(False)
    if st.session_state.page == "Check cow":
        check_page(frame, key, meta)
    elif st.session_state.page == "Sensors":
        sensor_page(frame)
    else:
        data_page(frame, meta, key)
    st.markdown("---")
    st.caption("MilkWise · Hackathon demonstration. Production alerts are not veterinary diagnoses. Raw organizer data © dataset authors (CC BY 4.0).")
