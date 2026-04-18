"""
VARCC Dashboard  —  Portfolio Building Intelligence System
==========================================================
Streamlit multi-page dashboard to showcase VARCC engine results.

Pages
─────
  1. Overview          – Market pulse, sector heatmap, anomaly timeline
  2. Industry View     – Cement sector – all stocks comparative
  3. Stock Profile     – Deep-dive for one stock (date-range slider)
  4. Economic Sensitivity – Eco-factor drivers & correlation radar
  5. Portfolio Builder – Tier selection + recommendations

Run:
  streamlit run varcc_dashboard.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VARCC – Portfolio Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────
PALETTE = {
    "bg"         : "#0F1117",
    "card"       : "#1A1D27",
    "accent1"    : "#00D4FF",
    "accent2"    : "#7B2FBE",
    "accent3"    : "#FF6B35",
    "green"      : "#00C896",
    "red"        : "#FF4B6E",
    "yellow"     : "#FFD700",
    "text_light" : "#E8EAF0",
    "text_muted" : "#8B929E",
}

STATE_COLORS = {
    "High Return – Low Risk" : "#00C896",
    "High Return – High Risk": "#FFD700",
    "Low Return  – Low Risk" : "#00D4FF",
    "Low Return  – High Risk": "#FF4B6E",
}

STATE_ICONS = {
    "High Return – Low Risk" : "🟢",
    "High Return – High Risk": "🟡",
    "Low Return  – Low Risk" : "🔵",
    "Low Return  – High Risk": "🔴",
}

TIER_COLORS = {
    "short"   : "#00C896",
    "moderate": "#00D4FF",
    "long"    : "#7B2FBE",
}

ECO_COLORS = px.colors.qualitative.Plotly

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #0F1117; color: #E8EAF0; }
    .block-container { padding: 1.5rem 2rem; }

    /* Metric cards */
    .varcc-card {
        background: #1A1D27;
        border: 1px solid #2A2D3E;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        min-height: 100px;
    }
    .varcc-card h4 {
        color: #C0C4CE !important;
        font-size: 0.82rem;
        margin: 0 0 6px 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .varcc-card .val {
        color: #FFFFFF !important;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .varcc-card .sub {
        color: #A0A7B4 !important;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        color: #00D4FF;
        font-size: 1.05rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border-bottom: 1px solid #2A2D3E;
        padding-bottom: 6px;
        margin: 18px 0 14px 0;
    }

    /* State badge */
    .state-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* Sidebar - force text visibility */
    [data-testid="stSidebar"] {
        background: #12141F;
        border-right: 1px solid #2A2D3E;
    }
    [data-testid="stSidebar"] * { color: #E8EAF0 !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #C0C4CE !important; }
    [data-testid="stSidebar"] label[data-baseweb="radio"] { color: #E8EAF0 !important; }
    [data-testid="stSidebar"] p { color: #E8EAF0 !important; }
    [data-testid="stSidebar"] span { color: #E8EAF0 !important; }

    /* Radio buttons in sidebar */
    div[data-testid="stRadio"] label { color: #E8EAF0 !important; font-size: 0.95rem !important; }
    div[data-testid="stRadio"] p { color: #E8EAF0 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #A0A7B4; }
    .stTabs [aria-selected="true"] { color: #00D4FF !important; border-bottom-color: #00D4FF !important; }

    /* Dataframe */
    .stDataFrame { border-radius: 8px; }

    /* Remove plotly toolbar clutter */
    .js-plotly-plot .plotly .modebar { opacity: 0.3; }

    /* Plotly legend text fix */
    .js-plotly-plot .legend text { fill: #E8EAF0 !important; }
    .js-plotly-plot .gtitle { fill: #E8EAF0 !important; }
    .js-plotly-plot .xtick text, .js-plotly-plot .ytick text { fill: #C0C4CE !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PKL  = os.path.join(BASE, "varcc_results.pkl")

@st.cache_resource(show_spinner="Loading VARCC engine results …")
def load_results():
    with open(PKL, "rb") as f:
        return pickle.load(f)

try:
    R = load_results()
except FileNotFoundError:
    st.error("❌  `varcc_results.pkl` not found. Run `varcc_engine.py` first.")
    st.stop()

STOCKS           = R["stocks"]
ECO_LABELS       = R["eco_factor_labels"]
CAPM             = R["capm_results"]
VOL_DATA         = R["vol_data"]
VOL_ANOM         = R["vol_anomalies"]
VOL_Z            = R["vol_zscore"]
COMB_ANOM        = R["combined_anomalies"]
ECO_SENS         = R["eco_sensitivity"]
YEARLY_STATES    = R["yearly_states"]
OVERALL_STATES   = R["overall_states"]
FEAT_DF          = R["feat_df"]
OVERALL_DF       = R["overall_df"]
PORT_TIERS       = R["portfolio_tiers"]
FORECASTS        = R["forecasts"]
YEARLY_SUM       = R["yearly_summary"]
MONTHLY_SUM      = R["monthly_summary"]
LR               = R["lr"]
KSE              = R["kse"]
ECO              = R["eco"]
BEST_MODELS      = R["best_models"]
MODEL_METRICS    = R["model_metrics"]
INDUSTRY         = R["industry"]


# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def plotly_dark_layout(fig, title="", height=420, margin=None):
    m = margin or dict(l=40, r=30, t=50, b=40)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#FFFFFF")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(color="#C0C4CE", size=12),
        height=height,
        margin=m,
        legend=dict(
            bgcolor="rgba(26,29,39,0.85)",
            bordercolor="#3A3D4E",
            borderwidth=1,
            font=dict(color="#E8EAF0", size=12),
        ),
        xaxis=dict(
            gridcolor="#2A2D3E",
            zeroline=False,
            tickfont=dict(color="#C0C4CE", size=11),
            title_font=dict(color="#C0C4CE"),
        ),
        yaxis=dict(
            gridcolor="#2A2D3E",
            zeroline=False,
            tickfont=dict(color="#C0C4CE", size=11),
            title_font=dict(color="#C0C4CE"),
        ),
    )
    return fig


def metric_card(label, value, sub="", color=None):
    col_style = f"color:{color};" if color else ""
    return f"""
    <div class="varcc-card">
        <h4>{label}</h4>
        <div class="val" style="{col_style}">{value}</div>
        <div class="sub">{sub}</div>
    </div>"""


def state_badge(state):
    c = STATE_COLORS.get(state, "#888")
    ico = STATE_ICONS.get(state, "")
    return (f'<span class="state-badge" '
            f'style="background:{c}22;color:{c};border:1px solid {c}44;">'
            f'{ico} {state}</span>')


def pct_color(val):
    if isinstance(val, str):
        try:
            val = float(val.replace("%",""))
        except Exception:
            return PALETTE["text_muted"]
    return PALETTE["green"] if val >= 0 else PALETTE["red"]



def hex_to_rgba(hex_color, alpha=1.0):
    """Convert a hex color string to rgba() string compatible with Plotly."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div style="text-align:center;padding:10px 0 20px;">'
        f'<span style="font-size:1.8rem;font-weight:800;'
        f'background:linear-gradient(90deg,{PALETTE["accent1"]},{PALETTE["accent2"]});'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        f'VARCC</span><br>'
        f'<span style="color:{PALETTE["text_muted"]};font-size:0.72rem;">'
        f'Portfolio Building Intelligence</span></div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        ["📈  Overview",
         "🏭  Industry View",
         "🔍  Stock Profile",
         "🌐  Economic Sensitivity",
         "💼  Portfolio Builder"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:0.7rem;">'
        f'<b style="color:{PALETTE["accent1"]}">V</b>olatility &nbsp;'
        f'<b style="color:{PALETTE["accent1"]}">A</b>nomalies &nbsp;'
        f'<b style="color:{PALETTE["accent1"]}">R</b>eturns &nbsp;'
        f'<b style="color:{PALETTE["accent1"]}">C</b>orrelation &nbsp;'
        f'<b style="color:{PALETTE["accent1"]}">C</b>lassification'
        f'<br><br>Industry: <b>{INDUSTRY}</b><br>'
        f'Stocks: <b>{len(STOCKS)}</b><br>'
        f'Period: <b>Jul 2015 – Jun 2025</b><br>'
        //f'Anomaly Threshold: <b>|z| > {R["anomaly_threshold"]}</b>'//
        f'</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════
# PAGE 1 ─ OVERVIEW
# ═══════════════════════════════════════════════════════════════

if "Overview" in page:
    st.markdown(
        f'<h1 style="color:{PALETTE["accent1"]};font-size:1.7rem;margin-bottom:4px;">'
        f'Market Overview  <span style="font-size:1rem;color:{PALETTE["text_muted"]};">'
        f'— {INDUSTRY} Sector · KSE-100</span></h1>',
        unsafe_allow_html=True
    )

    # ── KPI Row ───────────────────────────────────────────────────
    kpi_cols = st.columns(5)
    sector_avg_ret  = np.mean([CAPM[s]["ann_actual_ret"] for s in STOCKS]) * 100
    sector_avg_vol  = np.mean([CAPM[s]["ann_vol"] for s in STOCKS]) * 100
    sector_avg_sh   = np.mean([CAPM[s]["sharpe_ratio"] for s in STOCKS if not np.isnan(CAPM[s]["sharpe_ratio"])])
    sector_avg_beta = np.mean([CAPM[s]["beta"] for s in STOCKS])
    total_anom_days = sum(COMB_ANOM[s].sum() for s in STOCKS if s in COMB_ANOM)

    with kpi_cols[0]:
        st.markdown(metric_card("Sector Avg Return",
                                f"{sector_avg_ret:.1f}%",
                                "Annualised (10Y)",
                                PALETTE["green"] if sector_avg_ret > 0 else PALETTE["red"]),
                    unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(metric_card("Sector Avg Volatility",
                                f"{sector_avg_vol:.1f}%",
                                "Annualised (10Y)", PALETTE["yellow"]),
                    unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(metric_card("Avg Sharpe Ratio",
                                f"{sector_avg_sh:.2f}",
                                "All 14 stocks", PALETTE["accent1"]),
                    unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(metric_card("Avg Beta (β)",
                                f"{sector_avg_beta:.2f}",
                                "vs KSE-100", PALETTE["accent2"]),
                    unsafe_allow_html=True)
    with kpi_cols[4]:
        st.markdown(metric_card("Total Anomaly Events",
                                f"{total_anom_days:,}",
                                "Combined V+R · All stocks", PALETTE["accent3"]),
                    unsafe_allow_html=True)

    st.markdown('<div class="section-header">VARCC State Matrix</div>',
                unsafe_allow_html=True)

    # ── State matrix grid ─────────────────────────────────────────
    state_cols = st.columns(4)
    quadrants = list(STATE_COLORS.keys())
    for qi, quad in enumerate(quadrants):
        stocks_in_quad = [s for s, st_ in OVERALL_STATES.items() if st_ == quad]
        c = STATE_COLORS[quad]
        ico = STATE_ICONS[quad]
        with state_cols[qi]:
            stock_pills = " ".join(
                f'<span style="background:{c}22;color:{c};border:1px solid {c}55;'
                f'border-radius:6px;padding:2px 7px;font-size:0.75rem;'
                f'font-weight:600;">{s}</span>'
                for s in stocks_in_quad
            )
            st.markdown(
                f'<div class="varcc-card" style="border-color:{c}44;">'
                f'<h4>{ico} {quad}</h4>'
                f'<div style="font-size:1.4rem;font-weight:700;color:{c};">'
                f'{len(stocks_in_quad)}</div>'
                f'<div style="color:#8B929E;font-size:0.7rem;">stocks</div>'
                f'<div style="margin-top:8px;">{stock_pills}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    col_l, col_r = st.columns([3, 2])

    # ── KSE-100 + Sector volatility overlay ──────────────────────
    with col_l:
        st.markdown('<div class="section-header">KSE-100 Index · 10-Year Trend</div>',
                    unsafe_allow_html=True)
        kse_plot = KSE.reset_index()
        fig_kse = go.Figure()
        fig_kse.add_trace(go.Scatter(
            x=kse_plot["Date"], y=kse_plot["Value"],
            mode="lines", name="KSE-100",
            line=dict(color=PALETTE["accent1"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.05)"
        ))
        fig_kse = plotly_dark_layout(fig_kse, height=320)
        st.plotly_chart(fig_kse, width='stretch')

    # ── Stock return vs vol scatter ───────────────────────────────
    with col_r:
        st.markdown('<div class="section-header">Risk–Return Map</div>',
                    unsafe_allow_html=True)
        rr_data = pd.DataFrame({
            "Stock"  : STOCKS,
            "Return" : [CAPM[s]["ann_actual_ret"] * 100 for s in STOCKS],
            "Vol"    : [CAPM[s]["ann_vol"] * 100 for s in STOCKS],
            "Sharpe" : [CAPM[s]["sharpe_ratio"] for s in STOCKS],
            "State"  : [OVERALL_STATES[s] for s in STOCKS],
        })
        fig_rr = px.scatter(
            rr_data, x="Vol", y="Return", text="Stock",
            color="State", color_discrete_map=STATE_COLORS,
            size=[abs(s)+1 for s in rr_data["Sharpe"]],
            hover_data=["Sharpe"],
        )
        fig_rr.update_traces(textposition="top center", textfont_size=9)
        fig_rr.add_hline(y=0, line_dash="dash", line_color="#2A2D3E")
        fig_rr = plotly_dark_layout(fig_rr, height=320)
        st.plotly_chart(fig_rr, width='stretch')

    # ── Anomaly Timeline Heatmap ──────────────────────────────────
    st.markdown('<div class="section-header">Combined Anomaly Intensity · Yearly Heatmap</div>',
                unsafe_allow_html=True)

    heat_data = []
    years = sorted(LR["date"].dt.year.unique())
    for stock in STOCKS:
        if stock not in COMB_ANOM:
            continue
        anom = COMB_ANOM[stock]
        for yr in years:
            count = int(anom[anom.index.year == yr].sum())
            heat_data.append({"Stock": stock, "Year": yr, "Anomaly Days": count})

    heat_df = pd.DataFrame(heat_data).pivot(index="Stock", columns="Year",
                                            values="Anomaly Days").fillna(0)
    fig_heat = go.Figure(go.Heatmap(
        z=heat_df.values,
        x=[str(y) for y in heat_df.columns],
        y=heat_df.index.tolist(),
        colorscale=[[0,"#1A1D27"],[0.3,"#7B2FBE"],[0.7,"#FF6B35"],[1,"#FF4B6E"]],
        hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Anomaly Days: %{z}<extra></extra>",
    ))
    fig_heat = plotly_dark_layout(fig_heat, height=320, margin=dict(l=80,r=30,t=40,b=40))
    st.plotly_chart(fig_heat, width='stretch')

    # ── CAPM summary table ────────────────────────────────────────
    st.markdown('<div class="section-header">VARCC Engine Summary — All Stocks</div>',
                unsafe_allow_html=True)

    summary_rows = []
    for s in STOCKS:
        cr = CAPM[s]
        fc = FORECASTS.get(s, {})
        summary_rows.append({
            "Stock"          : s,
            "Best Model"     : BEST_MODELS.get(s, "—"),
            "Beta (β)"       : cr["beta"],
            "Alpha (Ann %)"  : f"{cr['alpha_annual']*100:.2f}%",
            "Ann Return (%)" : f"{cr['ann_actual_ret']*100:.2f}%",
            "Ann Vol (%)"    : f"{cr['ann_vol']*100:.2f}%",
            "Sharpe"         : f"{cr['sharpe_ratio']:.2f}",
            "VARCC State"    : OVERALL_STATES.get(s,"—"),
            "Fwd Vol (%)"    : f"{fc.get('forecast_vol_pct','—')}%" if fc else "—",
            "Fwd Ret (%)"    : f"{fc.get('forecast_ann_ret_pct','—')}%" if fc else "—",
        })

    sum_df = pd.DataFrame(summary_rows)
    st.dataframe(sum_df, width='stretch', hide_index=True,
                 column_config={
                     "VARCC State": st.column_config.TextColumn(width="medium"),
                 })


# ═══════════════════════════════════════════════════════════════
# PAGE 2 ─ INDUSTRY VIEW
# ═══════════════════════════════════════════════════════════════

elif "Industry" in page:
    st.markdown(
        f'<h1 style="color:{PALETTE["accent1"]};font-size:1.7rem;margin-bottom:4px;">'
        f'🏭  {INDUSTRY} Sector — All Stocks Comparative</h1>',
        unsafe_allow_html=True
    )

    # ── Industry selector (future-ready) ─────────────────────────
    industry_sel = st.selectbox(
        "Select Industry", [INDUSTRY],
        help="More industries coming soon"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊  Returns & Risk", "⚡  Volatility Regimes", "🔔  Anomalies", "📅  Yearly Heatmap"]
    )

    # ── Tab 1: Returns & Risk ─────────────────────────────────────
    with tab1:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-header">Annualised Returns — All Stocks</div>',
                        unsafe_allow_html=True)
            ret_df = pd.DataFrame({
                "Stock" : STOCKS,
                "Return": [CAPM[s]["ann_actual_ret"]*100 for s in STOCKS],
                "State" : [OVERALL_STATES[s] for s in STOCKS],
            }).sort_values("Return", ascending=True)
            fig_bar = go.Figure(go.Bar(
                x=ret_df["Return"], y=ret_df["Stock"],
                orientation="h",
                marker_color=[STATE_COLORS.get(st_, "#888") for st_ in ret_df["State"]],
                text=[f"{v:.1f}%" for v in ret_df["Return"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Return: %{x:.2f}%<extra></extra>"
            ))
            fig_bar.add_vline(x=0, line_color="#2A2D3E")
            fig_bar = plotly_dark_layout(fig_bar, height=420)
            st.plotly_chart(fig_bar, width='stretch')

        with col_r:
            st.markdown('<div class="section-header">Beta vs Sharpe Ratio</div>',
                        unsafe_allow_html=True)
            bs_df = pd.DataFrame({
                "Stock" : STOCKS,
                "Beta"  : [CAPM[s]["beta"] for s in STOCKS],
                "Sharpe": [CAPM[s]["sharpe_ratio"] for s in STOCKS],
                "State" : [OVERALL_STATES[s] for s in STOCKS],
                "Alpha" : [CAPM[s]["alpha_annual"]*100 for s in STOCKS],
            })
            fig_bs = px.scatter(
                bs_df, x="Beta", y="Sharpe", text="Stock",
                color="State", color_discrete_map=STATE_COLORS,
                size=[abs(a)+0.5 for a in bs_df["Alpha"]],
                hover_data=["Alpha"],
            )
            fig_bs.update_traces(textposition="top center", textfont_size=9)
            fig_bs.add_vline(x=1, line_dash="dash", line_color="#2A2D3E",
                             annotation_text="β=1")
            fig_bs = plotly_dark_layout(fig_bs, height=420)
            st.plotly_chart(fig_bs, width='stretch')

        # Forward forecast comparison
        st.markdown('<div class="section-header">Forward Forecast: Risk vs Return (Next ~3 Months)</div>',
                    unsafe_allow_html=True)
        fc_rows = []
        for s in STOCKS:
            fc = FORECASTS.get(s, {})
            if fc:
                fc_rows.append({
                    "Stock"     : s,
                    "Fwd Vol %" : fc.get("forecast_vol_pct", np.nan),
                    "Fwd Ret %" : fc.get("forecast_ann_ret_pct", np.nan),
                    "State"     : OVERALL_STATES.get(s, "—"),
                    "Model"     : fc.get("forecast_model", "—"),
                })
        fc_df = pd.DataFrame(fc_rows)
        fig_fc = px.scatter(
            fc_df, x="Fwd Vol %", y="Fwd Ret %", text="Stock",
            color="State", color_discrete_map=STATE_COLORS,
            hover_data=["Model"],
            title=""
        )
        fig_fc.update_traces(textposition="top center", textfont_size=9)
        fig_fc = plotly_dark_layout(fig_fc, height=360)
        st.plotly_chart(fig_fc, width='stretch')

    # ── Tab 2: Volatility Regimes ─────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Conditional Volatility — All Stocks (ARCH Family)</div>',
                    unsafe_allow_html=True)
        stocks_sel = st.multiselect("Select stocks", STOCKS, default=STOCKS[:6])

        fig_vol = go.Figure()
        for s in stocks_sel:
            if s not in VOL_DATA:
                continue
            v = VOL_DATA[s]
            fig_vol.add_trace(go.Scatter(
                x=v.index, y=v.values,
                mode="lines", name=s,
                line=dict(width=1),
                hovertemplate=f"<b>{s}</b><br>Date: %{{x}}<br>Cond Vol: %{{y:.2f}}<extra></extra>"
            ))
        fig_vol = plotly_dark_layout(fig_vol, "Conditional Volatility (% daily)", height=420)
        st.plotly_chart(fig_vol, width='stretch')

        # Model selection table
        st.markdown('<div class="section-header">Best ARCH Model Per Stock (AIC Selection)</div>',
                    unsafe_allow_html=True)
        model_rows = []
        for s in STOCKS:
            metrics = MODEL_METRICS.get(s, {})
            row = {"Stock": s, "Best Model": BEST_MODELS.get(s, "—")}
            for m, vals in metrics.items():
                row[f"{m} AIC"] = round(vals["AIC"], 1)
                row[f"{m} BIC"] = round(vals["BIC"], 1)
            model_rows.append(row)
        st.dataframe(pd.DataFrame(model_rows), width='stretch', hide_index=True)

    # ── Tab 3: Anomalies ──────────────────────────────────────────
    with tab3:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-header">Volatility Anomaly Days per Stock</div>',
                        unsafe_allow_html=True)
            anom_v = pd.DataFrame({
                "Stock": STOCKS,
                "Vol Anom Days": [int(VOL_ANOM[s].sum()) if s in VOL_ANOM else 0 for s in STOCKS],
                "Ret Anom Days": [int(CAPM[s]["ret_anomaly"].sum()) for s in STOCKS],
                "Combined Anom": [int(COMB_ANOM[s].sum()) if s in COMB_ANOM else 0 for s in STOCKS],
            }).sort_values("Combined Anom", ascending=True)

            fig_anom = go.Figure()
            fig_anom.add_trace(go.Bar(
                x=anom_v["Vol Anom Days"], y=anom_v["Stock"],
                orientation="h", name="Vol Anomaly",
                marker_color=PALETTE["accent2"]
            ))
            fig_anom.add_trace(go.Bar(
                x=anom_v["Ret Anom Days"], y=anom_v["Stock"],
                orientation="h", name="Return Anomaly",
                marker_color=PALETTE["accent3"]
            ))
            fig_anom.update_layout(barmode="group")
            fig_anom = plotly_dark_layout(fig_anom, height=420)
            st.plotly_chart(fig_anom, width='stretch')

        with col_r:
            st.markdown('<div class="section-header">Combined Anomaly % of Trading Days</div>',
                        unsafe_allow_html=True)
            n_days = 2471
            anom_pct = pd.DataFrame({
                "Stock"  : STOCKS,
                "Anom %" : [round(COMB_ANOM[s].sum() / n_days * 100, 1)
                            if s in COMB_ANOM else 0 for s in STOCKS],
            }).sort_values("Anom %", ascending=False)

            fig_pct = go.Figure(go.Bar(
                x=anom_pct["Stock"], y=anom_pct["Anom %"],
                marker_color=[PALETTE["red"] if v > 15 else PALETTE["yellow"]
                              if v > 10 else PALETTE["green"]
                              for v in anom_pct["Anom %"]],
                text=[f"{v:.1f}%" for v in anom_pct["Anom %"]],
                textposition="outside",
            ))
            fig_pct.add_hline(y=10, line_dash="dash", line_color="#2A2D3E",
                              annotation_text="10% threshold")
            fig_pct = plotly_dark_layout(fig_pct, "% of Trading Days Flagged", height=420)
            st.plotly_chart(fig_pct, width='stretch')

    # ── Tab 4: Yearly Heatmap ─────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Annual Return (%) — All Stocks</div>',
                    unsafe_allow_html=True)

        ysum = YEARLY_SUM.copy()
        pivot_ret = ysum.pivot(index="Stock", columns="Year",
                               values="Ann. Return (%)").fillna(np.nan)
        fig_yr = go.Figure(go.Heatmap(
            z=pivot_ret.values,
            x=[str(y) for y in pivot_ret.columns],
            y=pivot_ret.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Return: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Ann Ret %", tickfont=dict(size=9))
        ))
        fig_yr = plotly_dark_layout(fig_yr, height=420, margin=dict(l=80,r=30,t=40,b=40))
        st.plotly_chart(fig_yr, width='stretch')

        st.markdown('<div class="section-header">Annual Volatility (%) — All Stocks</div>',
                    unsafe_allow_html=True)
        pivot_vol = ysum.pivot(index="Stock", columns="Year",
                               values="Ann. Vol (%)").fillna(np.nan)
        fig_yvol = go.Figure(go.Heatmap(
            z=pivot_vol.values,
            x=[str(y) for y in pivot_vol.columns],
            y=pivot_vol.index.tolist(),
            colorscale=[[0,"#1A1D27"],[0.5,"#7B2FBE"],[1,"#FF4B6E"]],
            hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Vol: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Ann Vol %", tickfont=dict(size=9))
        ))
        fig_yvol = plotly_dark_layout(fig_yvol, height=420, margin=dict(l=80,r=30,t=40,b=40))
        st.plotly_chart(fig_yvol, width='stretch')


# ═══════════════════════════════════════════════════════════════
# PAGE 3 ─ STOCK PROFILE
# ═══════════════════════════════════════════════════════════════

elif "Stock Profile" in page:
    st.markdown(
        f'<h1 style="color:{PALETTE["accent1"]};font-size:1.7rem;margin-bottom:4px;">'
        f'🔍  Stock Profile</h1>',
        unsafe_allow_html=True
    )

    col_sel, col_ind = st.columns([2, 2])
    with col_ind:
        industry_sel = st.selectbox("Industry", [INDUSTRY],
                                    help="More industries will be added")
    with col_sel:
        stock_sel = st.selectbox("Select Stock", STOCKS)

    # Date range
    min_date = LR["date"].min().date()
    max_date = LR["date"].max().date()
    date_range = st.slider(
        "Date Range",
        min_value=min_date, max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Filter data
    cr      = CAPM[stock_sel]
    daily   = cr["daily_df"]
    mask    = (daily.index.date >= date_range[0]) & (daily.index.date <= date_range[1])
    daily_f = daily[mask]

    vol_s   = VOL_DATA.get(stock_sel, pd.Series())
    vmask   = (vol_s.index.date >= date_range[0]) & (vol_s.index.date <= date_range[1])
    vol_f   = vol_s[vmask]

    va_f    = VOL_ANOM[stock_sel][vmask] if stock_sel in VOL_ANOM else pd.Series()
    ca_f    = COMB_ANOM[stock_sel]
    camask  = (ca_f.index.date >= date_range[0]) & (ca_f.index.date <= date_range[1])
    ca_f    = ca_f[camask]

    # ── KPI Row ───────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)

    period_ret = (daily_f["stock_ret"].mean() * 252 * 100
                  if len(daily_f) > 0 else 0)
    period_vol = (daily_f["stock_ret"].std() * np.sqrt(252) * 100
                  if len(daily_f) > 0 else 0)
    period_sh  = period_ret / period_vol if period_vol > 0 else 0

    with k1:
        st.markdown(metric_card("Stock", stock_sel, industry_sel,
                                PALETTE["accent1"]), unsafe_allow_html=True)
    with k2:
        st.markdown(metric_card("Period Return",
                                f"{period_ret:.1f}%", "Annualised",
                                pct_color(period_ret)), unsafe_allow_html=True)
    with k3:
        st.markdown(metric_card("Period Vol",
                                f"{period_vol:.1f}%", "Annualised",
                                PALETTE["yellow"]), unsafe_allow_html=True)
    with k4:
        st.markdown(metric_card("Sharpe Ratio",
                                f"{period_sh:.2f}", "Period",
                                PALETTE["accent1"]), unsafe_allow_html=True)
    with k5:
        st.markdown(metric_card("Beta (β)",
                                f"{cr['beta']:.3f}", "vs KSE-100",
                                PALETTE["accent2"]), unsafe_allow_html=True)
    with k6:
        state = OVERALL_STATES.get(stock_sel, "—")
        fc    = FORECASTS.get(stock_sel, {})
        st.markdown(
            f'<div class="varcc-card">'
            f'<h4>VARCC State</h4>'
            f'{state_badge(state)}'
            f'<div class="sub" style="margin-top:6px;">Best Model: {BEST_MODELS.get(stock_sel,"—")}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Main Charts ───────────────────────────────────────────────
    tab_p, tab_v, tab_capm, tab_yr, tab_mth = st.tabs(
        ["📈 Price & Returns", "⚡ Volatility", "📐 CAPM Analysis",
         "📅 Yearly Breakdown", "🗓️ Monthly Detail"]
    )

    with tab_p:
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown('<div class="section-header">Log Returns — with Anomaly Markers</div>',
                        unsafe_allow_html=True)
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Scatter(
                x=daily_f.index, y=daily_f["stock_ret"] * 100,
                mode="lines", name="Log Return",
                line=dict(color=PALETTE["accent1"], width=0.8)
            ))
            # Anomaly markers
            anom_days = ca_f[ca_f].index
            if len(anom_days) > 0:
                anom_ret  = daily_f.loc[daily_f.index.isin(anom_days), "stock_ret"] * 100
                fig_ret.add_trace(go.Scatter(
                    x=anom_ret.index, y=anom_ret.values,
                    mode="markers", name="Anomaly",
                    marker=dict(color=PALETTE["red"], size=5, symbol="x")
                ))
            fig_ret.add_hline(y=0, line_color="#2A2D3E")
            fig_ret = plotly_dark_layout(fig_ret, "Daily Log Return (%)", height=340)
            st.plotly_chart(fig_ret, width='stretch')

        with col_r:
            st.markdown('<div class="section-header">Return Distribution</div>',
                        unsafe_allow_html=True)
            ret_vals = daily_f["stock_ret"] * 100
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=ret_vals, nbinsx=60,
                marker_color=PALETTE["accent1"],
                opacity=0.7, name="Returns",
                hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>"
            ))
            # Normal overlay
            mu, sig = ret_vals.mean(), ret_vals.std()
            xn = np.linspace(ret_vals.min(), ret_vals.max(), 200)
            from scipy.stats import norm
            yn = norm.pdf(xn, mu, sig) * len(ret_vals) * (ret_vals.max()-ret_vals.min()) / 60
            fig_hist.add_trace(go.Scatter(
                x=xn, y=yn, mode="lines", name="Normal",
                line=dict(color=PALETTE["yellow"], dash="dash", width=1.5)
            ))
            fig_hist = plotly_dark_layout(fig_hist, "Return Distribution", height=340)
            st.plotly_chart(fig_hist, width='stretch')

        # Cumulative return
        st.markdown('<div class="section-header">Cumulative Return vs KSE-100</div>',
                    unsafe_allow_html=True)
        cum_stock  = (1 + daily_f["stock_ret"]).cumprod() - 1
        cum_market = (1 + daily_f["market_return"]).cumprod() - 1
        cum_rfr    = (1 + daily_f["rfr_daily"]).cumprod() - 1

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum_stock.index, y=cum_stock * 100,
            mode="lines", name=stock_sel,
            line=dict(color=PALETTE["accent1"], width=2)
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum_market.index, y=cum_market * 100,
            mode="lines", name="KSE-100",
            line=dict(color=PALETTE["yellow"], width=1.5, dash="dot")
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum_rfr.index, y=cum_rfr * 100,
            mode="lines", name="Risk-Free Rate",
            line=dict(color=PALETTE["text_muted"], width=1, dash="dash")
        ))
        fig_cum = plotly_dark_layout(fig_cum, "Cumulative Return (%)", height=340)
        st.plotly_chart(fig_cum, width='stretch')

    with tab_v:
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown(f'<div class="section-header">Conditional Volatility — {BEST_MODELS.get(stock_sel,"")}</div>',
                        unsafe_allow_html=True)
            fig_vol2 = go.Figure()
            fig_vol2.add_trace(go.Scatter(
                x=vol_f.index, y=vol_f.values,
                mode="lines", name="Cond. Vol",
                line=dict(color=PALETTE["accent2"], width=1.2),
                fill="tozeroy", fillcolor="rgba(123,47,190,0.1)"
            ))
            # vol anomaly markers
            if len(va_f) > 0:
                va_days = va_f[va_f].index
                va_vals = vol_f[vol_f.index.isin(va_days)]
                fig_vol2.add_trace(go.Scatter(
                    x=va_vals.index, y=va_vals.values,
                    mode="markers", name="Vol Anomaly",
                    marker=dict(color=PALETTE["red"], size=5, symbol="x")
                ))
            fig_vol2 = plotly_dark_layout(fig_vol2, "Conditional Volatility (% daily)", height=340)
            st.plotly_chart(fig_vol2, width='stretch')

        with col_r:
            st.markdown('<div class="section-header">Volatility Z-Score</div>',
                        unsafe_allow_html=True)
            vz = VOL_Z.get(stock_sel, pd.Series())
            vz_mask = (vz.index.date >= date_range[0]) & (vz.index.date <= date_range[1])
            vz_f = vz[vz_mask]

            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(
                x=vz_f.index, y=vz_f.values,
                mode="lines", name="Z-Score",
                line=dict(color=PALETTE["accent3"], width=0.8)
            ))
            fig_z.add_hline(y=2,  line_dash="dash", line_color=PALETTE["red"],   annotation_text="+2σ")
            fig_z.add_hline(y=-2, line_dash="dash", line_color=PALETTE["green"], annotation_text="-2σ")
            fig_z.add_hline(y=0,  line_color="#2A2D3E")
            fig_z = plotly_dark_layout(fig_z, "Vol Anomaly Z-Score", height=340)
            st.plotly_chart(fig_z, width='stretch')

        # Model metrics
        st.markdown('<div class="section-header">ARCH Model Comparison — Information Criteria</div>',
                    unsafe_allow_html=True)
        m_metrics = MODEL_METRICS.get(stock_sel, {})
        if m_metrics:
            m_df = pd.DataFrame(m_metrics).T.reset_index().rename(columns={"index":"Model"})
            m_df = m_df.round(1)
            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(
                x=m_df["Model"], y=m_df["AIC"],
                name="AIC",
                marker_color=[PALETTE["green"] if v == m_df["AIC"].min()
                              else PALETTE["accent2"] for v in m_df["AIC"]],
            ))
            fig_m.add_trace(go.Bar(
                x=m_df["Model"], y=m_df["BIC"],
                name="BIC",
                marker_color=[PALETTE["yellow"] if v == m_df["BIC"].min()
                              else PALETTE["text_muted"] for v in m_df["BIC"]],
            ))
            fig_m.update_layout(barmode="group")
            fig_m = plotly_dark_layout(fig_m, "Lower is better (Best model highlighted)", height=300)
            st.plotly_chart(fig_m, width='stretch')

    with tab_capm:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-header">Actual vs CAPM Expected Return</div>',
                        unsafe_allow_html=True)
            fig_capm = go.Figure()
            # Rolling 63-day actual vs expected
            roll_actual = daily_f["stock_ret"].rolling(63).mean() * 252 * 100
            roll_capm   = daily_f["capm_expected"].rolling(63).mean() * 252 * 100
            fig_capm.add_trace(go.Scatter(
                x=roll_actual.index, y=roll_actual,
                mode="lines", name="Actual (Roll 63d)",
                line=dict(color=PALETTE["accent1"], width=1.5)
            ))
            fig_capm.add_trace(go.Scatter(
                x=roll_capm.index, y=roll_capm,
                mode="lines", name="CAPM Expected",
                line=dict(color=PALETTE["yellow"], width=1.5, dash="dot")
            ))
            fig_capm.add_hline(y=0, line_color="#2A2D3E")
            fig_capm = plotly_dark_layout(fig_capm, "Rolling Annualised Return (%)", height=340)
            st.plotly_chart(fig_capm, width='stretch')

        with col_r:
            st.markdown('<div class="section-header">Jensen\'s Alpha (Rolling)</div>',
                        unsafe_allow_html=True)
            roll_alpha = (daily_f["return_residual"].rolling(126).mean() * 252 * 100)
            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Scatter(
                x=roll_alpha.index, y=roll_alpha,
                mode="lines", name="Rolling Alpha",
                line=dict(color=PALETTE["green"], width=1.5),
                fill="tozeroy",
                fillcolor="rgba(0,200,150,0.08)"
            ))
            fig_alpha.add_hline(y=0, line_dash="dash", line_color="#2A2D3E",
                                annotation_text="α=0")
            fig_alpha = plotly_dark_layout(fig_alpha, "Rolling 6M Jensen's Alpha (%)", height=340)
            st.plotly_chart(fig_alpha, width='stretch')

        # CAPM scatter
        st.markdown('<div class="section-header">Security Characteristic Line (SCL)</div>',
                    unsafe_allow_html=True)
        sample = daily_f.sample(min(500, len(daily_f)), random_state=42) if len(daily_f) > 0 else daily_f
        fig_scl = go.Figure()
        fig_scl.add_trace(go.Scatter(
            x=sample["excess_market"]*100, y=sample["excess_stock"]*100,
            mode="markers", name="Daily Observations",
            marker=dict(color=PALETTE["accent1"], size=3, opacity=0.4)
        ))
        xr = np.linspace(sample["excess_market"].min(), sample["excess_market"].max(), 100)*100
        yr = cr["alpha_daily"]*100 + cr["beta"] * xr
        fig_scl.add_trace(go.Scatter(
            x=xr, y=yr, mode="lines", name=f"SCL (β={cr['beta']:.2f})",
            line=dict(color=PALETTE["yellow"], width=2)
        ))
        fig_scl = plotly_dark_layout(fig_scl, "Excess Stock Return vs Excess Market Return", height=340)
        st.plotly_chart(fig_scl, width='stretch')

        # Forecast card
        fc = FORECASTS.get(stock_sel, {})
        if fc:
            st.markdown('<div class="section-header">Forward Outlook (~3 Months)</div>',
                        unsafe_allow_html=True)
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown(metric_card("Forecast Vol",
                                        f"{fc['forecast_vol_pct']:.1f}%",
                                        "Annualised", PALETTE["yellow"]),
                            unsafe_allow_html=True)
            with fc2:
                v = fc["forecast_ann_ret_pct"]
                st.markdown(metric_card("Forecast Return",
                                        f"{v:.1f}%",
                                        "Annualised", pct_color(v)),
                            unsafe_allow_html=True)
            with fc3:
                st.markdown(metric_card("Forecast Model",
                                        fc.get("forecast_model","—"),
                                        "ARCH family", PALETTE["accent2"]),
                            unsafe_allow_html=True)

    with tab_yr:
        st.markdown('<div class="section-header">Yearly State Evolution</div>',
                    unsafe_allow_html=True)
        yr_states = YEARLY_STATES.get(stock_sel, {})
        yr_summary = YEARLY_SUM[YEARLY_SUM["Stock"] == stock_sel].copy()

        # State bar
        if yr_states:
            ys_df = pd.DataFrame(
                [{"Year": y, "State": s, "Color": STATE_COLORS.get(s,"#888")}
                 for y, s in sorted(yr_states.items())]
            )
            yr_summary["State"] = yr_summary["Year"].map(yr_states)
            fig_ys = go.Figure()
            for _, row in ys_df.iterrows():
                fig_ys.add_vrect(
                    x0=row["Year"]-0.5, x1=row["Year"]+0.5,
                    fillcolor=row["Color"], opacity=0.15, layer="below",
                    line_width=0
                )
            # Overlay return bars
            fig_ys.add_trace(go.Bar(
                x=yr_summary["Year"],
                y=yr_summary["Ann. Return (%)"],
                name="Ann. Return %",
                marker_color=[STATE_COLORS.get(s,"#888")
                              for s in yr_summary["State"].fillna("—")],
                text=[f"{v:.1f}%" for v in yr_summary["Ann. Return (%)"]],
                textposition="outside",
            ))
            fig_ys.add_hline(y=0, line_color="#2A2D3E")
            fig_ys = plotly_dark_layout(fig_ys, "Annual Return (%) with VARCC State", height=380)
            st.plotly_chart(fig_ys, width='stretch')

        # Yearly table
        yr_summary_disp = yr_summary.drop(columns=["Stock"], errors="ignore")
        st.dataframe(yr_summary_disp, width='stretch', hide_index=True)

    with tab_mth:
        st.markdown('<div class="section-header">Monthly Detail</div>',
                    unsafe_allow_html=True)
        mth = MONTHLY_SUM[MONTHLY_SUM["Stock"] == stock_sel].copy()
        mth_mask = (
            (mth["Month"].dt.date >= date_range[0]) &
            (mth["Month"].dt.date <= date_range[1])
        )
        mth_f = mth[mth_mask].copy()

        col_l, col_r = st.columns(2)
        with col_l:
            fig_mret = go.Figure(go.Bar(
                x=mth_f["Month"], y=mth_f["Monthly Ret (%)"],
                marker_color=[PALETTE["green"] if v >= 0 else PALETTE["red"]
                              for v in mth_f["Monthly Ret (%)"]],
                name="Monthly Return %",
            ))
            fig_mret = plotly_dark_layout(fig_mret, "Monthly Return (%)", height=300)
            st.plotly_chart(fig_mret, width='stretch')
        with col_r:
            fig_mvol = go.Figure(go.Bar(
                x=mth_f["Month"], y=mth_f["Monthly Vol (%)"],
                marker_color=PALETTE["accent2"],
                name="Monthly Vol %",
            ))
            fig_mvol = plotly_dark_layout(fig_mvol, "Monthly Volatility (%)", height=300)
            st.plotly_chart(fig_mvol, width='stretch')

        mth_f_disp = mth_f.drop(columns=["Stock"], errors="ignore")
        mth_f_disp["Month"] = mth_f_disp["Month"].dt.strftime("%b %Y")
        st.dataframe(mth_f_disp, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 ─ ECONOMIC SENSITIVITY
# ═══════════════════════════════════════════════════════════════

elif "Economic" in page:
    st.markdown(
        f'<h1 style="color:{PALETTE["accent1"]};font-size:1.7rem;margin-bottom:4px;">'
        f'🌐  Economic Sensitivity Analysis</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="color:{PALETTE["text_muted"]};font-size:0.82rem;margin-bottom:18px;">'
        f'How macroeconomic shocks drive anomalies & excess returns across cement stocks</div>',
        unsafe_allow_html=True
    )

    tab_heat, tab_single, tab_eco_ts = st.tabs(
        ["📊 Sector Heatmap", "🔍 Single Stock Deep-Dive", "📉 Eco Factor Timeline"]
    )

    with tab_heat:
        view_sel = st.radio("View", ["Anomaly Sensitivity", "Return Sensitivity"],
                            horizontal=True)
        key = "corr_anom" if "Anomaly" in view_sel else "corr_return"

        eco_factors = list(ECO_LABELS.keys())

        heat_r = np.zeros((len(STOCKS), len(eco_factors)))
        heat_p = np.zeros((len(STOCKS), len(eco_factors)))

        for i, s in enumerate(STOCKS):
            sens = ECO_SENS.get(s, {}).get(key, {})
            for j, f in enumerate(eco_factors):
                heat_r[i, j] = sens.get(f, {}).get("r", 0)
                heat_p[i, j] = sens.get(f, {}).get("p", 1)

        eco_labels_short = [ECO_LABELS[f].split(" (")[0] for f in eco_factors]

        # Mask insignificant (p > 0.1) with lighter text
        hover_text = [
            [f"<b>{STOCKS[i]} × {eco_labels_short[j]}</b><br>"
             f"r = {heat_r[i,j]:.3f}<br>p = {heat_p[i,j]:.3f}<br>"
             f"{'✓ Significant' if heat_p[i,j] < 0.1 else '— Not significant'}"
             for j in range(len(eco_factors))]
            for i in range(len(STOCKS))
        ]

        fig_eheat = go.Figure(go.Heatmap(
            z=heat_r,
            x=eco_labels_short,
            y=STOCKS,
            colorscale="RdBu",
            zmid=0,
            zmin=-0.5, zmax=0.5,
            text=[[f"{v:.2f}" for v in row] for row in heat_r],
            texttemplate="%{text}",
            hovertext=hover_text,
            hoverinfo="text",
            colorbar=dict(title="Pearson r", tickfont=dict(size=9))
        ))
        fig_eheat = plotly_dark_layout(
            fig_eheat,
            f"Pearson Correlation: Stock {view_sel} vs Economic Factors (Monthly)",
            height=520,
            margin=dict(l=80, r=30, t=60, b=100)
        )
        fig_eheat.update_layout(
            xaxis=dict(tickangle=-30, tickfont=dict(size=10))
        )
        st.plotly_chart(fig_eheat, width='stretch')

        st.info("🔵 Blue = negative correlation  |  🔴 Red = positive correlation  |  "
                "Values ±0.1–0.3 = moderate  |  ±0.3+ = strong")

    with tab_single:
        eco_stock = st.selectbox("Select Stock", STOCKS, key="eco_stock")
        sens = ECO_SENS.get(eco_stock, {})

        if not sens:
            st.warning("No sensitivity data for this stock.")
        else:
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown('<div class="section-header">Anomaly Sensitivity — Radar</div>',
                            unsafe_allow_html=True)
                corr_a = sens.get("corr_anom", {})
                r_vals = [corr_a.get(f, {}).get("r", 0) for f in eco_factors]
                fig_radar = go.Figure(go.Scatterpolar(
                    r=[abs(v) for v in r_vals],
                    theta=eco_labels_short,
                    fill="toself",
                    fillcolor=f"rgba(123,47,190,0.2)",
                    line_color=PALETTE["accent2"],
                    name="Abs Correlation",
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 0.5], color=PALETTE["text_muted"]),
                        angularaxis=dict(color=PALETTE["text_muted"])
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=360, margin=dict(l=40,r=40,t=40,b=40),
                    showlegend=False
                )
                st.plotly_chart(fig_radar, width='stretch')

            with col_r:
                st.markdown('<div class="section-header">Return Sensitivity — Radar</div>',
                            unsafe_allow_html=True)
                corr_r2 = sens.get("corr_return", {})
                r_vals2 = [corr_r2.get(f, {}).get("r", 0) for f in eco_factors]
                fig_radar2 = go.Figure(go.Scatterpolar(
                    r=[abs(v) for v in r_vals2],
                    theta=eco_labels_short,
                    fill="toself",
                    fillcolor="rgba(0,212,255,0.15)",
                    line_color=PALETTE["accent1"],
                    name="Abs Correlation",
                ))
                fig_radar2.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 0.5], color=PALETTE["text_muted"]),
                        angularaxis=dict(color=PALETTE["text_muted"])
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=360, margin=dict(l=40,r=40,t=40,b=40),
                    showlegend=False
                )
                st.plotly_chart(fig_radar2, width='stretch')

            # Correlation bar chart
            st.markdown('<div class="section-header">Signed Correlation Coefficients</div>',
                        unsafe_allow_html=True)
            corr_df = pd.DataFrame({
                "Factor"   : eco_labels_short,
                "Anom r"   : [corr_a.get(f, {}).get("r", 0) for f in eco_factors],
                "Return r" : [corr_r2.get(f, {}).get("r", 0) for f in eco_factors],
                "Anom p"   : [corr_a.get(f, {}).get("p", 1) for f in eco_factors],
                "Return p" : [corr_r2.get(f, {}).get("p", 1) for f in eco_factors],
            })
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Bar(
                x=corr_df["Factor"], y=corr_df["Anom r"],
                name="Anomaly r",
                marker_color=[PALETTE["red"] if v > 0 else PALETTE["green"]
                              for v in corr_df["Anom r"]],
            ))
            fig_corr.add_trace(go.Bar(
                x=corr_df["Factor"], y=corr_df["Return r"],
                name="Return r",
                marker_color=PALETTE["accent1"],
                opacity=0.7,
            ))
            fig_corr.add_hline(y=0, line_color="#2A2D3E")
            fig_corr.add_hline(y=0.2,  line_dash="dot", line_color="#444")
            fig_corr.add_hline(y=-0.2, line_dash="dot", line_color="#444")
            fig_corr.update_layout(barmode="group")
            fig_corr = plotly_dark_layout(fig_corr, "Anomaly & Return Correlation with Eco Factors", height=340)
            st.plotly_chart(fig_corr, width='stretch')

            # OLS sensitivity table
            ols_a = sens.get("ols_anom_coef", {})
            ols_r = sens.get("ols_return_coef", {})
            if ols_a:
                st.markdown('<div class="section-header">OLS Sensitivity Coefficients (Standardised)</div>',
                            unsafe_allow_html=True)
                ols_df = pd.DataFrame({
                    "Eco Factor"        : list(ECO_LABELS.values()),
                    "Anomaly β"         : [ols_a.get(f, np.nan) for f in eco_factors],
                    "Return β"          : [ols_r.get(f, np.nan) for f in eco_factors],
                }).round(4)
                r2_row = pd.DataFrame({
                    "Eco Factor": [f"Model R² (Anom={sens.get('ols_anom_r2','—')} / Ret={sens.get('ols_return_r2','—')})"],
                    "Anomaly β": [np.nan], "Return β": [np.nan]
                })
                st.dataframe(ols_df, width='stretch', hide_index=True)
                st.caption(f"OLS R² — Anomaly model: {sens.get('ols_anom_r2','—')}  |  "
                           f"Return model: {sens.get('ols_return_r2','—')}")

    with tab_eco_ts:
        st.markdown('<div class="section-header">Economic Factor Trends (Monthly)</div>',
                    unsafe_allow_html=True)
        eco_factor_sel = st.selectbox("Select Eco Factor",
                                      list(ECO_LABELS.values()),
                                      key="eco_ts_factor")
        eco_col = [k for k, v in ECO_LABELS.items() if v == eco_factor_sel][0]

        fig_eco_ts = go.Figure()
        fig_eco_ts.add_trace(go.Scatter(
            x=ECO.index, y=ECO[eco_col],
            mode="lines+markers",
            name=eco_factor_sel,
            line=dict(color=PALETTE["accent1"], width=2),
            marker=dict(size=4)
        ))
        fig_eco_ts = plotly_dark_layout(fig_eco_ts, eco_factor_sel, height=360)
        st.plotly_chart(fig_eco_ts, width='stretch')

        # Overlay with sector anomaly intensity
        eco_stock2 = st.selectbox("Overlay anomaly intensity for stock:", STOCKS, key="eco_overlay")
        sens2 = ECO_SENS.get(eco_stock2, {})
        merged = sens2.get("merged_monthly", pd.DataFrame())

        if len(merged) > 0 and eco_col in merged.columns:
            fig_ov = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ov.add_trace(go.Scatter(
                x=merged.index, y=merged[eco_col],
                mode="lines", name=eco_factor_sel,
                line=dict(color=PALETTE["accent1"], width=1.5)
            ), secondary_y=False)
            fig_ov.add_trace(go.Bar(
                x=merged.index, y=merged["anom_count"],
                name=f"{eco_stock2} Anomaly Days",
                marker_color=PALETTE["accent3"],
                opacity=0.5
            ), secondary_y=True)
            fig_ov.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["text_muted"]),
                height=360,
                margin=dict(l=40,r=40,t=50,b=40),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            fig_ov.update_yaxes(gridcolor="#2A2D3E")
            fig_ov.update_xaxes(gridcolor="#2A2D3E")
            st.plotly_chart(fig_ov, width='stretch')


# ═══════════════════════════════════════════════════════════════
# PAGE 5 ─ PORTFOLIO BUILDER
# ═══════════════════════════════════════════════════════════════

elif "Portfolio" in page:
    st.markdown(
        f'<h1 style="color:{PALETTE["accent1"]};font-size:1.7rem;margin-bottom:4px;">'
        f'💼  Portfolio Builder</h1>',
        unsafe_allow_html=True
    )

    # Tier selection
    tier_col, info_col = st.columns([2, 3])
    with tier_col:
        tier = st.radio(
            "Select Investment Horizon",
            ["Short Term (0–3 Yrs)", "Moderate Term (3–5 Yrs)", "Long Term (5–10 Yrs)"],
        )

    tier_key = "short" if "Short" in tier else ("moderate" if "Moderate" in tier else "long")

    with info_col:
        tier_descriptions = {
            "short"   : ("Prioritises **High Return – Low Risk** stocks with strong Sharpe "
                         "ratios and positive recent alpha. Best for capital preservation "
                         "with moderate growth."),
            "moderate": ("Balances quality (High Return – Low Risk) and stable performers "
                         "(Low Return – Low Risk). Suitable for steady compounding over a "
                         "medium horizon."),
            "long"    : ("Full spectrum ranked by risk-adjusted return. Includes "
                         "high-beta names that may underperform short-term but reward "
                         "patient investors over a decade."),
        }
        st.info(tier_descriptions[tier_key])

    ranked_stocks = PORT_TIERS[tier_key]
    tier_color    = TIER_COLORS[tier_key]

    # ── Recommendation Table ──────────────────────────────────────
    st.markdown(f'<div class="section-header">Recommended Stocks — {tier}</div>',
                unsafe_allow_html=True)

    rec_rows = []
    for rank, s in enumerate(ranked_stocks, 1):
        cr  = CAPM[s]
        fc  = FORECASTS.get(s, {})
        state = OVERALL_STATES.get(s, "—")
        rec_rows.append({
            "Rank"           : rank,
            "Stock"          : s,
            "VARCC State"    : state,
            "Ann Return (%)" : f"{cr['ann_actual_ret']*100:.1f}%",
            "Ann Vol (%)"    : f"{cr['ann_vol']*100:.1f}%",
            "Sharpe"         : f"{cr['sharpe_ratio']:.2f}",
            "Beta"           : f"{cr['beta']:.2f}",
            "Alpha (Ann %)"  : f"{cr['alpha_annual']*100:.2f}%",
            "Fwd Ret (%)"    : f"{fc.get('forecast_ann_ret_pct','—')}%" if fc else "—",
            "Fwd Vol (%)"    : f"{fc.get('forecast_vol_pct','—')}%" if fc else "—",
        })

    rec_df = pd.DataFrame(rec_rows)
    st.dataframe(rec_df, width='stretch', hide_index=True)

    # ── Visual: Portfolio bubble chart ───────────────────────────
    st.markdown('<div class="section-header">Portfolio Risk–Return Map</div>',
                unsafe_allow_html=True)
    bubble_data = pd.DataFrame({
        "Stock"  : STOCKS,
        "Return" : [CAPM[s]["ann_actual_ret"]*100 for s in STOCKS],
        "Vol"    : [CAPM[s]["ann_vol"]*100 for s in STOCKS],
        "Sharpe" : [max(CAPM[s]["sharpe_ratio"], 0) for s in STOCKS],
        "State"  : [OVERALL_STATES.get(s,"—") for s in STOCKS],
        "Rank"   : [ranked_stocks.index(s)+1 if s in ranked_stocks else 99 for s in STOCKS],
        "Selected": [s in ranked_stocks[:6] for s in STOCKS],
    })
    fig_port = px.scatter(
        bubble_data, x="Vol", y="Return", text="Stock",
        color="State", color_discrete_map=STATE_COLORS,
        size=[s*4+2 for s in bubble_data["Sharpe"]],
        opacity=[0.95 if sel else 0.3 for sel in bubble_data["Selected"]],
        hover_data=["Sharpe","Rank"],
    )
    fig_port.update_traces(textposition="top center", textfont_size=9)
    fig_port.add_hline(y=0, line_dash="dash", line_color="#2A2D3E")
    # Add quadrant shading
    fig_port.add_vrect(
        x0=bubble_data["Vol"].min(), x1=bubble_data["Vol"].mean(),
        fillcolor="rgba(0,200,150,0.03)", layer="below", line_width=0
    )
    fig_port = plotly_dark_layout(
        fig_port,
        f"Risk–Return Map — highlighted: top 6 picks for {tier}",
        height=440
    )
    st.plotly_chart(fig_port, width='stretch')

    # ── Equal-weight portfolio simulation ────────────────────────
    st.markdown('<div class="section-header">Simulated Equal-Weight Portfolio — Top 6 Picks</div>',
                unsafe_allow_html=True)

    top6 = ranked_stocks[:6]
    if top6:
        port_ret_series = []
        for s in top6:
            s_ret = LR[LR["stock"] == s].set_index("date")["log_return"]
            port_ret_series.append(s_ret)

        port_df  = pd.concat(port_ret_series, axis=1)
        port_df.columns = top6
        port_df  = port_df.dropna()
        ew_ret   = port_df.mean(axis=1)
        cum_port = (1 + ew_ret).cumprod() - 1
        cum_mkt  = (1 + KSE["market_return"].reindex(ew_ret.index).fillna(0)).cumprod() - 1

        port_ann_ret = ew_ret.mean() * 252 * 100
        port_ann_vol = ew_ret.std()  * np.sqrt(252) * 100
        port_sharpe  = port_ann_ret / port_ann_vol if port_ann_vol > 0 else 0

        pm1, pm2, pm3 = st.columns(3)
        with pm1:
            st.markdown(metric_card("Portfolio Ann. Return",
                                    f"{port_ann_ret:.1f}%", "Equal-weight top 6",
                                    pct_color(port_ann_ret)), unsafe_allow_html=True)
        with pm2:
            st.markdown(metric_card("Portfolio Ann. Vol",
                                    f"{port_ann_vol:.1f}%", "Equal-weight top 6",
                                    PALETTE["yellow"]), unsafe_allow_html=True)
        with pm3:
            st.markdown(metric_card("Portfolio Sharpe",
                                    f"{port_sharpe:.2f}", "Equal-weight top 6",
                                    PALETTE["accent1"]), unsafe_allow_html=True)

        fig_psim = go.Figure()
        fig_psim.add_trace(go.Scatter(
            x=cum_port.index, y=cum_port * 100,
            mode="lines", name=f"VARCC Portfolio ({tier_key})",
            line=dict(color=tier_color, width=2.5),
            fill="tozeroy", fillcolor=hex_to_rgba(tier_color, 0.094)
        ))
        fig_psim.add_trace(go.Scatter(
            x=cum_mkt.index, y=cum_mkt * 100,
            mode="lines", name="KSE-100 Benchmark",
            line=dict(color=PALETTE["yellow"], width=1.5, dash="dot")
        ))
        fig_psim.add_hline(y=0, line_color="#2A2D3E")
        fig_psim = plotly_dark_layout(fig_psim, "Cumulative Return (%): Portfolio vs KSE-100", height=380)
        st.plotly_chart(fig_psim, width='stretch')

        # Yearly state heatmap for top6
        st.markdown(f'<div class="section-header">Yearly VARCC States — Top 6 Picks</div>',
                    unsafe_allow_html=True)
        years = sorted(LR["date"].dt.year.unique())
        state_z = []
        state_text = []
        for s in top6:
            row_z, row_t = [], []
            for yr in years:
                state = YEARLY_STATES.get(s, {}).get(yr, "—")
                state_idx = list(STATE_COLORS.keys()).index(state) if state in STATE_COLORS else -1
                row_z.append(state_idx)
                row_t.append(f"{STATE_ICONS.get(state,'')} {state}")
            state_z.append(row_z)
            state_text.append(row_t)

        fig_state_heat = go.Figure(go.Heatmap(
            z=state_z,
            x=[str(y) for y in years],
            y=top6,
            text=state_text,
            texttemplate="%{text}",
            colorscale=[
                [0.0, STATE_COLORS["High Return – Low Risk"]],
                [0.33, STATE_COLORS["High Return – High Risk"]],
                [0.66, STATE_COLORS["Low Return  – Low Risk"]],
                [1.0, STATE_COLORS["Low Return  – High Risk"]],
            ],
            showscale=False,
            hovertemplate="<b>%{y}</b><br>Year: %{x}<br>State: %{text}<extra></extra>",
        ))
        fig_state_heat = plotly_dark_layout(
            fig_state_heat, height=280, margin=dict(l=80,r=30,t=40,b=40)
        )
        fig_state_heat.update_layout(font=dict(size=8))
        st.plotly_chart(fig_state_heat, width='stretch')

    # ── Analyst notes ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Analyst Guidance</div>',
                unsafe_allow_html=True)
    notes = {
        "short": """
**Short-Term Picks Logic:**
- Stocks ranked by `High Return – Low Risk` state (green quadrant), then `High Return – High Risk`.
- Emphasis on **positive Jensen's Alpha** and **Sharpe > 0.4**.
- Lower beta preferred to limit market-shock exposure.
- Forward volatility forecast should be < 40% annualised.
        """,
        "moderate": """
**Moderate-Term Picks Logic:**
- Balanced mix of **quality compounders** (HR-LR) and **stable defensives** (LR-LR).
- CAPM R² > 0.15 preferred — these stocks are meaningfully linked to market cycles.
- Positive or near-zero alpha over the full 10Y window.
- Look for KIBOR & Inflation sensitivity to hedge macro cycles.
        """,
        "long": """
**Long-Term Picks Logic:**
- Full universe ranked by risk-adjusted return over the complete 10Y period.
- High-beta cyclical names can be included — cement demand tracks GDP/construction.
- Eco-sensitivity to **PKR/USD** and **Imports** matters for long-run input cost dynamics.
- Annual VARCC state consistency (fewer years in `LR-HR`) is a positive signal.
        """,
    }
    st.markdown(notes[tier_key])
