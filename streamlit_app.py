import streamlit as st

from src.inference.inference_pipeline import predict_stock


st.set_page_config(
    page_title="Quant Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COMPANIES = [
    ("Reliance Industries", "RELIANCE.NS"),
    ("Infosys", "INFY.NS"),
    ("HDFC Bank", "HDFCBANK.NS"),
    ("ICICI Bank", "ICICIBANK.NS"),
    ("Bharti Airtel", "BHARTIARTL.NS"),
    ("Adani Enterprises", "ADANIENT.NS"),
    ("State Bank of India", "SBIN.NS"),
    ("TCS", "TCS.NS"),
]

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(16, 185, 129, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.10), transparent 26%),
                linear-gradient(180deg, #07111f 0%, #0b1728 45%, #0d1b2a 100%);
            color: #ecf3ff;
        }

        .block-container {
            padding-top: 3.2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(17, 24, 39, 0.72));
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
            margin: 0.5rem 0 1.25rem 0;
            overflow: hidden;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.45rem;
            letter-spacing: -0.03em;
            color: #f8fafc;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #cbd5e1;
            max-width: 760px;
            line-height: 1.7;
        }

        .panel {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 20px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
        }

        .panel-title {
            font-size: 1rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.8rem;
        }

        .metric-card {
            border-radius: 20px;
            padding: 1.1rem 1rem;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.85));
            border: 1px solid rgba(148, 163, 184, 0.14);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
            min-height: 130px;
        }

        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-bottom: 0.55rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.1;
        }

        .metric-note {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: #cbd5e1;
        }

        .up-badge {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(16, 185, 129, 0.16);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.28);
            font-weight: 700;
            font-size: 0.9rem;
        }

        .down-badge {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(239, 68, 68, 0.16);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.28);
            font-weight: 700;
            font-size: 0.9rem;
        }

        .headline-card {
            padding: 1rem 1rem 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.14);
            margin-bottom: 0.8rem;
        }

        .headline-index {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 700;
            color: #93c5fd;
            margin-bottom: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .headline-text {
            color: #e5eefc;
            line-height: 1.55;
            font-size: 0.98rem;
        }

        .footer-note {
            color: #94a3b8;
            font-size: 0.88rem;
            margin-top: 0.8rem;
        }

        div[data-testid="stSelectbox"] > div {
            background-color: rgba(15, 23, 42, 0.82);
            border-radius: 14px;
        }

        .stButton > button {
            width: 100%;
            height: 3.1rem;
            border-radius: 14px;
            border: 0;
            font-weight: 700;
            font-size: 1rem;
            background: linear-gradient(90deg, #10b981, #06b6d4);
            color: #04111f;
            box-shadow: 0 12px 28px rgba(6, 182, 212, 0.25);
        }

        .stButton > button:hover {
            filter: brightness(1.04);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Quant Sentiment Analyzer</div>
        <div class="hero-subtitle">
            A multimodal stock movement dashboard combining financial news sentiment and market signals
            to estimate next-day directional bias.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.1, 2.2], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Run Inference</div>', unsafe_allow_html=True)

    company = st.selectbox(
        "Choose a company",
        options=COMPANIES,
        format_func=lambda x: f"{x[0]}  •  {x[1]}",
    )

    run_clicked = st.button("Generate Prediction")

    st.markdown(
        '<div class="footer-note">Model uses latest fetched headlines and recent market indicators for inference.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    if not run_clicked:
        st.markdown(
            """
            <div class="panel" style="min-height: 240px; display: flex; align-items: center;">
                <div>
                    <div class="panel-title">Awaiting Input</div>
                    <div style="color:#cbd5e1; line-height:1.7;">
                        Select a company on the left and run prediction to view direction, probability,
                        confidence, and the latest related headlines.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if run_clicked:
    name, ticker = company

    with st.spinner(f"Running inference for {name}..."):
        result = predict_stock(name, ticker)

    probability = float(result.get("probability", 0.0))
    confidence = float(result.get("confidence", probability))
    direction = result.get("prediction", "N/A")
    headlines = result.get("headlines", [])

    badge_class = "up-badge" if direction == "UP" else "down-badge"

    st.markdown("")
    m1, m2, m3 = st.columns(3, gap="large")

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Direction</div>
                <div class="metric-value"><span class="{badge_class}">{direction}</span></div>
                <div class="metric-note">Predicted next-session movement</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Probability</div>
                <div class="metric-value">{probability:.3f}</div>
                <div class="metric-note">Raw model output score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence:.3f}</div>
                <div class="metric-note">Uncertainty-adjusted confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Latest Headlines</div>', unsafe_allow_html=True)

    if headlines:
        for idx, headline in enumerate(headlines, start=1):
            st.markdown(
                f"""
                <div class="headline-card">
                    <div class="headline-index">Headline {idx:02d}</div>
                    <div class="headline-text">{headline}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No headlines were returned for the selected company.")

    st.markdown("</div>", unsafe_allow_html=True)
