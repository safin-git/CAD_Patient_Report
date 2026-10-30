import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Precision Cardiovascular Risk Platform", layout="wide")

# ---------- Dark theme (replaces .streamlit/config.toml) ----------
st.markdown("""
<style>
.stApp, [data-testid="stSidebar"] {background:#0b0b0d; color:#e8e8ea;}
[data-testid="stSidebar"] {border-right:1px solid #26262b;}
header[data-testid="stHeader"] {background:#0b0b0d;}
.block-container {padding-top:2rem; max-width:100%;}
.box {background:#131316;border:1px solid #26262b;border-radius:14px;padding:18px 20px;}
.lbl {font-size:12px;letter-spacing:.08em;color:#9a9aa3;text-transform:uppercase;}
.row {display:flex;justify-content:space-between;align-items:center;
      padding:9px 0;border-bottom:1px solid #1f1f24;font-size:14px;}
.pill {font-size:11px;font-weight:700;padding:3px 8px;border-radius:5px;
       border:1px solid;font-family:monospace;}
.ok {color:#22c55e;border-color:#166534;background:#0f2a1a;}
.warn {color:#f59e0b;border-color:#92400e;background:#2a1f0a;}
</style>
""", unsafe_allow_html=True)

# ---------- Mock data (replace with your model / dataset later) ----------
SCENARIOS = {
    "Healthy — Baseline": dict(
        risk=20, hr=68, sbp=122, dbp=81, hrv=70, fai=-82.0, cac=0,
        trend=[20, 20, 20, 19, 19, 20, 20], shd=6, echo="NORM",
        readouts=[("Blood Pressure (BP)", "117/78 mmHg", "HEALTHY", "ok"),
                  ("Heart Rate (HR)", "68 BPM", "NORMAL SINUS RHYTHM (NSR)", "ok"),
                  ("Corrected QT Interval (QTc)", "421 ms", "NORMAL", "ok"),
                  ("ST Segment", "0.00 mV", "ISOELECTRIC (NORMAL)", "ok"),
                  ("Pulse Transit Time (PTT)", "228 ms", "", ""),
                  ("Blood Oxygen Saturation (SpO₂)", "98%", "", "")],
        labs=[("Total Cholesterol", "180 mg/dL", "NORMAL", "ok"),
              ("LDL Cholesterol", "47 mg/dL", "NORMAL", "ok"),
              ("HDL Cholesterol", "55 mg/dL", "NORMAL", "ok"),
              ("Triglycerides", "388 mg/dL", "BORDERLINE", "warn")],
        diseases={"Atherosclerosis": 30, "Myocardial Ischemia": 6, "Arrhythmia": 0,
                  "Hypertensive": 35, "Heart Failure": 9},
    ),
    "Elevated — Example": dict(  # placeholder values, edit freely
        risk=58, hr=88, sbp=142, dbp=92, hrv=28, fai=-68.0, cac=210,
        trend=[48, 50, 53, 55, 56, 57, 58], shd=34, echo="ABNORMAL",
        readouts=[("Blood Pressure (BP)", "142/92 mmHg", "ELEVATED", "warn"),
                  ("Heart Rate (HR)", "88 BPM", "NORMAL SINUS RHYTHM (NSR)", "ok"),
                  ("Corrected QT Interval (QTc)", "448 ms", "BORDERLINE", "warn"),
                  ("ST Segment", "-0.08 mV", "DEPRESSED", "warn"),
                  ("Pulse Transit Time (PTT)", "205 ms", "", ""),
                  ("Blood Oxygen Saturation (SpO₂)", "95%", "", "")],
        labs=[("Total Cholesterol", "246 mg/dL", "HIGH", "warn"),
              ("LDL Cholesterol", "158 mg/dL", "HIGH", "warn"),
              ("HDL Cholesterol", "38 mg/dL", "LOW", "warn"),
              ("Triglycerides", "290 mg/dL", "BORDERLINE", "warn")],
        diseases={"Atherosclerosis": 62, "Myocardial Ischemia": 40, "Arrhythmia": 15,
                  "Hypertensive": 66, "Heart Failure": 30},
    ),
}

def category(score):
    for limit, name, color in [(20, "Very Low", "#22c55e"), (40, "Low", "#22c55e"),
                               (60, "Moderate", "#f59e0b"), (80, "High", "#f97316"),
                               (100, "Critical", "#ef4444")]:
        if score <= limit:
            return name, color

def card(title, value, unit="", sub="", color="#e8e8ea"):
    st.markdown(
        f"""<div class="box" style="min-height:190px">
        <div class="lbl">{title}</div>
        <div style="font-size:38px;font-weight:700;color:{color};margin-top:14px">
        {value} <span style="font-size:14px;color:#9a9aa3">{unit}</span></div>
        <div style="font-size:14px;color:{color};margin-top:8px">{sub}</div></div>""",
        unsafe_allow_html=True)

def bar(label, score):
    name, color = category(score)
    st.markdown(
        f"""<div style="margin-bottom:12px"><div style="display:flex;justify-content:space-between;font-size:14px">
        <span>{label}</span><span style="color:{color};font-weight:600">{score} {name}</span></div>
        <div style="background:#26262b;height:6px;border-radius:4px">
        <div style="width:{score}%;background:{color};height:6px;border-radius:4px"></div></div></div>""",
        unsafe_allow_html=True)

def table(title, rows, link=""):
    html = f'<div class="box" style="min-height:300px"><div class="lbl" style="display:flex;justify-content:space-between"><span>{title}</span><span style="color:#3b82f6">{link}</span></div>'
    for name, val, tag, cls in rows:
        pill = f'<span class="pill {cls}">{tag}</span>' if tag else ""
        html += f'<div class="row"><span class="lbl" style="text-transform:uppercase">{name}</span><span><b style="font-family:monospace">{val}</b> {pill}</span></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ❤️ Arohan")
    page = st.radio("Navigate", ["Dashboard", "Live Waveforms", "Patient Overview",
                                 "Patient Profile", "Lab Report", "Patient Report",
                                 "Scenarios", "EchoNext CNN"], label_visibility="collapsed")
    st.caption("v1.0.0")

# ---------- Header ----------
h1, h2 = st.columns([3, 1])
h1.markdown(f"#### Precision Cardiovascular Risk Intelligence Platform  /  :blue[{page}]")
scenario = h2.selectbox("Scenario", list(SCENARIOS), label_visibility="collapsed")
d = SCENARIOS[scenario]
cat, col = category(d["risk"])

if page != "Dashboard":
    st.info(f"'{page}' page is not built yet.")
    st.stop()

main, side = st.columns([3.2, 1])

with main:
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Risk Summary", f"{cat} Risk", sub=f"Score: {d['risk']}/100", color=col)
    with c2: card("Heart Rate (HR)", d["hr"], "BPM", "Normal: 60–100 bpm")
    with c3: card("Blood Pressure (BP)", f"{d['sbp']}/{d['dbp']}", "mmHg",
                  "Stage 1 Hypertensive" if d["sbp"] >= 130 or d["dbp"] >= 80 else "Normal",
                  color="#f59e0b")
    with c4: card("HRV (RMSSD)", d["hrv"], "ms", "Healthy" if d["hrv"] > 40 else "Reduced",
                  color="#22c55e" if d["hrv"] > 40 else "#f97316")

    st.write("")
    c5, c6 = st.columns(2)
    with c5: card("Fat Attenuation Index (FAI)", d["fai"], "HU",
                  "Normal · Low inflammation" if d["fai"] < -70.1 else "Elevated", "#22c55e")
    with c6: card("Coronary Artery Calcium Score (CAC)", d["cac"], "AU",
                  "None · Very low risk" if d["cac"] == 0 else "Calcification present",
                  "#22c55e" if d["cac"] == 0 else "#f59e0b")

    st.markdown("##### 7-DAY CARDIOVASCULAR RISK TREND")
    fig = go.Figure(go.Scatter(x=[f"D{i}" for i in range(1, 8)], y=d["trend"],
                               mode="lines+markers+text", text=d["trend"],
                               textposition="top center", line=dict(color=col)))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e8e8ea"), yaxis=dict(visible=False, range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)

    t1, t2 = st.columns(2)
    with t1: table("Cardiac Readouts", d["readouts"])
    with t2: table("Lab Summary", d["labs"], "View Details →")

    st.write("")
    echo_cls = "ok" if d["echo"] == "NORM" else "warn"
    st.markdown(f"""<div class="box" style="display:flex;justify-content:space-between;align-items:center">
        <div><div class="lbl">EchoNext 1D ResNet-34 Deep Learning (in-app)</div>
        <b>Classified:</b> <span class="pill {echo_cls}">{d['echo']}</span></div>
        <div class="lbl">Structural Heart Disease (SHD) Index: <span style="color:#3b82f6">{d['shd']}%</span></div>
        </div>""", unsafe_allow_html=True)

with side:
    st.caption("Confidence: 100%")
    st.markdown(f'<div class="box"><div class="lbl">What this means</div><p style="margin-top:10px">'
                f'The calculated score falls within the {cat} Risk category.</p></div>',
                unsafe_allow_html=True)
    st.markdown("##### DISEASE-SPECIFIC RISK")
    for k, v in d["diseases"].items():
        bar(k, v)