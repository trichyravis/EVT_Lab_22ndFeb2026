"""
Extreme Value Theory (EVT) Risk Lab — The Mountain Path: World of Finance
Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence

Extreme Value Theory · Tail Risk · VaR/ES via EVT · Single Stock & Portfolio EVT
Real-Time NSE Data · GEV / GPD Fitting · Block Maxima · Peaks-Over-Threshold

Design: Mountain Path Master Dark Theme
"""

import streamlit as st
import numpy as np
import pandas as pd
import warnings
import datetime
from scipy import stats
from scipy.stats import genpareto, genextreme, norm
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# ── yfinance ──────────────────────────────────────────────────────────────────
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ============================================================================
# NSE UNIVERSE
# ============================================================================
NSE_INDICES = {
    "NIFTY 50":      "^NSEI",
    "BANK NIFTY":    "^NSEBANK",
    "FINNIFTY":      "^CNXFIN",
    "NIFTY IT":      "^CNXIT",
    "NIFTY MIDCAP":  "^CNXMID",
}

NSE_STOCKS = {
    "RELIANCE":   "RELIANCE.NS",  "TCS":        "TCS.NS",
    "INFY":       "INFY.NS",      "HDFCBANK":   "HDFCBANK.NS",
    "ICICIBANK":  "ICICIBANK.NS", "SBIN":       "SBIN.NS",
    "WIPRO":      "WIPRO.NS",     "AXISBANK":   "AXISBANK.NS",
    "BAJFINANCE": "BAJFINANCE.NS","BHARTIARTL": "BHARTIARTL.NS",
    "HCLTECH":    "HCLTECH.NS",   "HINDUNILVR": "HINDUNILVR.NS",
    "ITC":        "ITC.NS",       "KOTAKBANK":  "KOTAKBANK.NS",
    "LT":         "LT.NS",        "MARUTI":     "MARUTI.NS",
    "TATAMOTORS": "TATAMOTORS.NS","TATASTEEL":  "TATASTEEL.NS",
    "SUNPHARMA":  "SUNPHARMA.NS", "DRREDDY":    "DRREDDY.NS",
    "NTPC":       "NTPC.NS",      "POWERGRID":  "POWERGRID.NS",
    "ADANIENT":   "ADANIENT.NS",  "CIPLA":      "CIPLA.NS",
    "COALINDIA":  "COALINDIA.NS", "ONGC":       "ONGC.NS",
    "TECHM":      "TECHM.NS",     "JSWSTEEL":   "JSWSTEEL.NS",
    "M&M":        "M&M.NS",       "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
}

ALL_INSTRUMENTS = {**NSE_INDICES, **NSE_STOCKS}

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="EVT Risk Lab | Mountain Path",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DESIGN — Mountain Path Master Dark Theme
# ============================================================================
COLORS = {
    'dark_blue':     '#003366',
    'medium_blue':   '#004d80',
    'accent_gold':   '#FFD700',
    'light_blue':    '#ADD8E6',
    'bg_dark':       '#0a1628',
    'card_bg':       '#112240',
    'text_primary':  '#e6f1ff',
    'text_secondary':'#8892b0',
    'text_dark':     '#1a1a2e',
    'success':       '#28a745',
    'danger':        '#dc3545',
    'warning':       '#ffc107',
}

BRANDING = {
    'name':        'The Mountain Path - World of Finance',
    'instructor':  'Prof. V. Ravichandran',
    'credentials': '28+ Years Corporate Finance & Banking | 10+ Years Academic Excellence',
    'icon':        '🏔️',
    'linkedin':    'https://www.linkedin.com/in/trichyravis',
    'github':      'https://github.com/trichyravis',
}

def apply_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

        .stApp {{
            background: linear-gradient(135deg, #1a2332 0%, #243447 50%, #2a3f5f 100%);
        }}
        .main {{ color: {COLORS['text_primary']} !important; }}
        .main *, .main p, .main span, .main div, .main li, .main label {{
            color: {COLORS['text_primary']} !important;
        }}
        .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {{
            color: {COLORS['accent_gold']} !important;
            font-family: 'Playfair Display', serif;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['bg_dark']} 0%, {COLORS['dark_blue']} 100%);
            border-right: 1px solid rgba(255,215,0,0.2);
        }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: {COLORS['text_primary']} !important;
        }}
        section[data-testid="stSidebar"] input {{
            color: {COLORS['text_dark']} !important;
            background-color: #ffffff !important;
        }}
        .header-container {{
            background: linear-gradient(135deg, {COLORS['dark_blue']}, {COLORS['medium_blue']});
            border: 2px solid {COLORS['accent_gold']};
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }}
        .header-container h1 {{
            font-family: 'Playfair Display', serif;
            color: {COLORS['accent_gold']};
            margin: 0; font-size: 2rem;
        }}
        .header-container p {{
            color: {COLORS['text_primary']};
            font-family: 'Source Sans Pro', sans-serif;
            margin: 0.3rem 0 0; font-size: 0.9rem;
        }}
        .metric-card {{
            background: {COLORS['card_bg']};
            border: 1px solid rgba(255,215,0,0.3);
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
            margin-bottom: 0.8rem;
        }}
        .metric-card .label {{
            color: {COLORS['text_secondary']};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-card .value {{
            color: {COLORS['accent_gold']};
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'Playfair Display', serif;
            margin-top: 0.3rem;
        }}
        .metric-card .sub {{
            color: {COLORS['text_secondary']};
            font-size: 0.75rem;
            margin-top: 0.3rem;
        }}
        .info-box {{
            background: rgba(0,51,102,0.5);
            border: 1px solid {COLORS['accent_gold']};
            border-radius: 8px;
            padding: 1rem 1.5rem;
            color: {COLORS['text_primary']};
            margin: 0.8rem 0;
        }}
        .section-title {{
            font-family: 'Playfair Display', serif;
            color: {COLORS['accent_gold']};
            font-size: 1.3rem;
            border-bottom: 2px solid rgba(255,215,0,0.3);
            padding-bottom: 0.5rem;
            margin: 1.5rem 0 1rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{
            background: {COLORS['card_bg']};
            border: 1px solid rgba(255,215,0,0.3);
            border-radius: 8px;
            color: {COLORS['text_primary']};
            padding: 0.5rem 1rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: {COLORS['dark_blue']};
            border: 2px solid {COLORS['accent_gold']};
            color: {COLORS['accent_gold']};
        }}
        [data-testid="stExpander"] {{
            background: {COLORS['card_bg']} !important;
            border: 1px solid rgba(255,215,0,0.35) !important;
            border-radius: 10px !important;
            margin-bottom: 0.5rem !important;
        }}
        [data-testid="stExpander"] summary {{
            background: {COLORS['card_bg']} !important;
            border-radius: 8px !important;
            padding: 0.7rem 1rem !important;
        }}
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary div,
        [data-testid="stExpander"] summary {{
            color: {COLORS['accent_gold']} !important;
            font-weight: 600 !important;
        }}
        [data-testid="stExpanderDetails"] p,
        [data-testid="stExpanderDetails"] span,
        [data-testid="stExpanderDetails"] div,
        [data-testid="stExpanderDetails"] li {{
            color: {COLORS['text_primary']} !important;
        }}
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['medium_blue']}, {COLORS['dark_blue']}) !important;
            color: {COLORS['accent_gold']} !important;
            border: 2px solid {COLORS['accent_gold']} !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, {COLORS['accent_gold']}, #d4af37) !important;
            color: {COLORS['dark_blue']} !important;
        }}
        .stAlert {{ background-color: rgba(255,255,255,0.95) !important; }}
        .stAlert p, .stAlert span, .stAlert div {{ color: {COLORS['text_dark']} !important; }}
        footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

apply_styles()

# ============================================================================
# COMPONENT HELPERS
# ============================================================================
def header_container(title, subtitle=None, description=None):
    s_html = f'<p style="font-size:1rem;color:{COLORS["accent_gold"]};font-weight:600;margin:0.5rem 0;">{subtitle}</p>' if subtitle else ""
    d_html = f'<p style="font-size:0.85rem;color:{COLORS["text_primary"]};margin:0.3rem 0;">{description}</p>' if description else ""
    st.markdown(f"""
    <div class="header-container">
        <h1>{BRANDING['icon']} {title}</h1>
        {s_html}{d_html}
        <p>{BRANDING['name']}</p>
        <p style="font-size:0.8rem;color:{COLORS['text_secondary']};">
            {BRANDING['instructor']} | {BRANDING['credentials']}
        </p>
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub=None, color=None):
    val_color = color or COLORS['accent_gold']
    s_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="color:{val_color};">{value}</div>
        {s_html}
    </div>""", unsafe_allow_html=True)

def section_title(t):
    st.markdown(f'<div class="section-title">{t}</div>', unsafe_allow_html=True)

def info_box(content, title=None):
    t_html = f"<h4 style='color:{COLORS['accent_gold']};margin-top:0;'>{title}</h4>" if title else ""
    st.markdown(f'<div class="info-box">{t_html}{content}</div>', unsafe_allow_html=True)

def sidebar_label(text):
    st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']};font-weight:700;margin:0.5rem 0 0.2rem;'>{text}</p>", unsafe_allow_html=True)

def footer():
    st.divider()
    st.markdown(f"""
    <div style="text-align:center;padding:1.5rem;">
        <p style="color:{COLORS['accent_gold']};font-family:'Playfair Display',serif;
                  font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;">
            {BRANDING['icon']} {BRANDING['name']}
        </p>
        <p style="color:{COLORS['text_secondary']};font-size:0.85rem;margin:0.3rem 0;">
            {BRANDING['instructor']} | {BRANDING['credentials']}
        </p>
        <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,215,0,0.3);">
            <a href="{BRANDING['linkedin']}" target="_blank"
               style="color:{COLORS['accent_gold']};text-decoration:none;margin:0 1rem;">
                🔗 LinkedIn
            </a>
            <a href="{BRANDING['github']}" target="_blank"
               style="color:{COLORS['accent_gold']};text-decoration:none;margin:0 1rem;">
                💻 GitHub
            </a>
        </div>
    </div>""", unsafe_allow_html=True)

# ============================================================================
# DATA FETCHING
# ============================================================================
@st.cache_data(ttl=3600)
def fetch_returns(ticker: str, period: str = "5y") -> pd.Series | None:
    if not YF_AVAILABLE:
        return None
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period)
        if hist.empty or len(hist) < 50:
            return None
        prices = hist['Close'].dropna()
        returns = np.log(prices / prices.shift(1)).dropna() * 100  # log returns in %
        return returns
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_multi_returns(tickers: list, period: str = "3y") -> pd.DataFrame | None:
    if not YF_AVAILABLE:
        return None
    try:
        frames = {}
        for name, ticker in tickers:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period)
            if not hist.empty and len(hist) > 100:
                prices = hist['Close'].dropna()
                frames[name] = np.log(prices / prices.shift(1)).dropna() * 100
        if not frames:
            return None
        df = pd.DataFrame(frames).dropna()
        return df
    except Exception:
        return None

# ============================================================================
# EVT ENGINE
# ============================================================================

# ── Block Maxima (GEV) ───────────────────────────────────────────────────────
def block_maxima_losses(returns: np.ndarray, block_size: int = 21) -> np.ndarray:
    """Extract block minima (max losses) from return series."""
    n_blocks = len(returns) // block_size
    losses = -returns  # convert returns to losses (positive = loss)
    block_max = np.array([losses[i*block_size:(i+1)*block_size].max() for i in range(n_blocks)])
    return block_max

def fit_gev(block_maxima: np.ndarray) -> dict:
    """Fit GEV distribution to block maxima."""
    try:
        xi, loc, scale = genextreme.fit(block_maxima)
        ks_stat, ks_p = stats.kstest(block_maxima, 'genextreme', args=(xi, loc, scale))
        # Compute log-likelihood
        ll = np.sum(genextreme.logpdf(block_maxima, xi, loc, scale))
        return {
            'xi': xi, 'loc': loc, 'scale': scale,
            'ks_stat': ks_stat, 'ks_p': ks_p,
            'loglik': ll, 'n_blocks': len(block_maxima),
            'type': 'Fréchet' if xi > 0.1 else ('Weibull' if xi < -0.1 else 'Gumbel'),
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def gev_quantile(p: float, gev: dict) -> float:
    """Return level for probability p (GEV)."""
    return genextreme.ppf(p, gev['xi'], gev['loc'], gev['scale'])

# ── Peaks Over Threshold (GPD) ───────────────────────────────────────────────
def pot_exceedances(losses: np.ndarray, threshold_pct: float = 95.0) -> tuple:
    """Return threshold value and exceedances above it."""
    u = np.percentile(losses, threshold_pct)
    exceedances = losses[losses > u] - u
    return u, exceedances

def fit_gpd(exceedances: np.ndarray) -> dict:
    """Fit GPD to exceedances."""
    try:
        xi, loc, scale = genpareto.fit(exceedances, floc=0)
        ks_stat, ks_p = stats.kstest(exceedances, 'genpareto', args=(xi, loc, scale))
        ll = np.sum(genpareto.logpdf(exceedances, xi, loc, scale))
        return {
            'xi': xi, 'loc': loc, 'scale': scale,
            'ks_stat': ks_stat, 'ks_p': ks_p,
            'loglik': ll, 'n_exc': len(exceedances),
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def gpd_var(p: float, n_total: int, n_exc: int, threshold: float, gpd: dict) -> float:
    """VaR at confidence level p using GPD (POT method)."""
    xi, scale = gpd['xi'], gpd['scale']
    n_u = n_exc
    n = n_total
    if xi == 0:
        return threshold + scale * np.log(n/n_u * (1-p))
    else:
        return threshold + (scale/xi) * ((n/n_u * (1-p))**(-xi) - 1)

def gpd_es(p: float, n_total: int, n_exc: int, threshold: float, gpd: dict) -> float:
    """Expected Shortfall at confidence level p using GPD."""
    var = gpd_var(p, n_total, n_exc, threshold, gpd)
    xi, scale = gpd['xi'], gpd['scale']
    if xi >= 1:
        return np.inf
    u = threshold
    beta_u = scale + xi * (u - threshold)
    es = var / (1 - xi) + (beta_u - xi * threshold) / (1 - xi)
    # Simpler formula: ES = VaR/(1-xi) + (scale - xi*threshold)/(1-xi)
    es = (var + scale - xi * threshold) / (1 - xi)
    return es

def historical_var_es(returns: np.ndarray, confidence: float = 0.99) -> tuple:
    """Historical simulation VaR and ES."""
    losses = -returns
    var = np.percentile(losses, confidence * 100)
    es = losses[losses >= var].mean()
    return var, es

def parametric_var_es(returns: np.ndarray, confidence: float = 0.99) -> tuple:
    """Normal parametric VaR and ES."""
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(confidence)
    var = -(mu - z * sigma)
    es = -(mu - sigma * norm.pdf(z) / (1 - confidence))
    return var, es

# ── Mean Excess Plot ─────────────────────────────────────────────────────────
def mean_excess(losses: np.ndarray, n_thresholds: int = 50) -> tuple:
    """Compute mean excess (residual life) function for threshold selection."""
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)
    # Use losses between 10th and 95th percentile as candidate thresholds
    u_min = np.percentile(losses, 10)
    u_max = np.percentile(losses, 95)
    thresholds = np.linspace(u_min, u_max, n_thresholds)
    mean_exc = []
    ci_upper = []
    ci_lower = []
    for u in thresholds:
        exc = losses[losses > u] - u
        if len(exc) < 5:
            break
        me = exc.mean()
        se = exc.std() / np.sqrt(len(exc))
        mean_exc.append(me)
        ci_upper.append(me + 1.96 * se)
        ci_lower.append(me - 1.96 * se)
    thresholds = thresholds[:len(mean_exc)]
    return thresholds, np.array(mean_exc), np.array(ci_lower), np.array(ci_upper)

# ── Portfolio EVT ────────────────────────────────────────────────────────────
def portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Compute weighted portfolio returns."""
    return (returns_df.values @ weights)

def compute_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr()

def compute_tail_dependence(returns1: np.ndarray, returns2: np.ndarray, q: float = 0.95) -> float:
    """Empirical lower tail dependence coefficient."""
    losses1 = -returns1
    losses2 = -returns2
    u = q
    q1 = np.percentile(losses1, u * 100)
    q2 = np.percentile(losses2, u * 100)
    joint_exceed = np.sum((losses1 > q1) & (losses2 > q2))
    exceed1 = np.sum(losses1 > q1)
    if exceed1 == 0:
        return 0.0
    return joint_exceed / exceed1

# ============================================================================
# PLOTLY DARK LAYOUT
# ============================================================================
def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor='#0f1824',
        plot_bgcolor='#0f1824',
        font=dict(color=COLORS['text_primary'], family='Source Sans Pro'),
        title_font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=15),
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
        legend=dict(bgcolor='rgba(17,34,64,0.8)', bordercolor=COLORS['accent_gold'],
                    borderwidth=1, font=dict(color=COLORS['text_primary'])),
    )
    base.update(kwargs)
    return base

# ============================================================================
# HEADER
# ============================================================================
header_container(
    title="Extreme Value Theory (EVT) Risk Lab",
    subtitle="Tail Risk Modelling | GEV · GPD · VaR · Expected Shortfall",
    description="Block Maxima Method · Peaks-Over-Threshold · Single Stock & Portfolio EVT · Real-Time NSE Data"
)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown(f"""
<div style="text-align:center;padding:1.2rem;background:rgba(255,215,0,0.08);
     border-radius:10px;margin-bottom:1.5rem;border:2px solid {COLORS['accent_gold']};">
    <h3 style="color:{COLORS['accent_gold']};margin:0;">🏔️ EVT LAB</h3>
    <p style="color:{COLORS['text_secondary']};font-size:0.75rem;margin:5px 0 0;">
        Extreme Value Theory · NSE Live Data</p>
</div>
""", unsafe_allow_html=True)

# ── Mode Selection ────────────────────────────────────────────────────────────
sidebar_label("🎯 Analysis Mode")
mode = st.sidebar.radio(
    "Mode", ["Single Stock / Index", "Portfolio EVT"],
    label_visibility="collapsed"
)

# ── Single Stock ──────────────────────────────────────────────────────────────
if mode == "Single Stock / Index":
    sidebar_label("📊 Select Instrument")
    section_options = list(NSE_INDICES.keys()) + ["── F&O Stocks ──"] + list(NSE_STOCKS.keys())
    selected_name = st.sidebar.selectbox(
        "Instrument", list(ALL_INSTRUMENTS.keys()),
        index=0, label_visibility="collapsed"
    )
    ticker = ALL_INSTRUMENTS[selected_name]

    sidebar_label("📅 Historical Period")
    period = st.sidebar.selectbox(
        "Period", ["2y", "3y", "5y", "10y"],
        index=2, label_visibility="collapsed",
        format_func=lambda x: {"2y":"2 Years","3y":"3 Years","5y":"5 Years","10y":"10 Years"}[x]
    )

# ── Portfolio ─────────────────────────────────────────────────────────────────
else:
    sidebar_label("📊 Select Portfolio Stocks")
    all_stocks = list(NSE_STOCKS.keys())
    portfolio_names = st.sidebar.multiselect(
        "Choose 2–8 stocks", all_stocks,
        default=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
        label_visibility="collapsed",
        max_selections=8
    )
    if len(portfolio_names) < 2:
        portfolio_names = ["RELIANCE", "TCS", "HDFCBANK"]

    sidebar_label("📅 Historical Period")
    period = st.sidebar.selectbox(
        "Period", ["2y", "3y", "5y"],
        index=1, label_visibility="collapsed",
        format_func=lambda x: {"2y":"2 Years","3y":"3 Years","5y":"5 Years"}[x]
    )

    sidebar_label("⚖️ Portfolio Weights")
    weight_method = st.sidebar.radio(
        "Weight Method", ["Equal Weight", "Custom Weights"],
        label_visibility="collapsed"
    )
    n_stocks = len(portfolio_names)
    if weight_method == "Equal Weight":
        weights = np.ones(n_stocks) / n_stocks
        st.sidebar.markdown(f"""
        <div style="background:rgba(255,215,0,0.06);border:1px solid rgba(255,215,0,0.2);
             border-radius:6px;padding:0.5rem 0.8rem;margin-top:0.3rem;">
            <p style="color:{COLORS['text_secondary']};font-size:0.72rem;margin:0;">
                Equal weights: {100/n_stocks:.1f}% each</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"<p style='color:{COLORS['text_secondary']};font-size:0.75rem;'>Weights must sum to 100%</p>", unsafe_allow_html=True)
        raw_weights = []
        remaining = 100.0
        for i, name in enumerate(portfolio_names):
            if i < n_stocks - 1:
                default_w = round(100.0 / n_stocks, 1)
                w = st.sidebar.number_input(f"{name} %", 1.0, 99.0, default_w, 1.0, key=f"w_{name}")
                raw_weights.append(w)
                remaining -= w
            else:
                st.sidebar.markdown(f"<p style='color:{COLORS['accent_gold']};font-size:0.8rem;margin:0.3rem 0;'>{name}: {remaining:.1f}%</p>", unsafe_allow_html=True)
                raw_weights.append(max(remaining, 0.1))
        weights = np.array(raw_weights) / sum(raw_weights)

# ── EVT Parameters ────────────────────────────────────────────────────────────
sidebar_label("⚙️ EVT Parameters")
block_size  = st.sidebar.slider("Block Size (days)", 5, 63, 21,
    help="Days per block for GEV block maxima. 21 = monthly, 63 = quarterly")
threshold_pct = st.sidebar.slider("POT Threshold Percentile (%)", 80, 99, 95,
    help="Percentile threshold for Peaks-Over-Threshold (GPD). Higher = fewer exceedances")
confidence = st.sidebar.slider("VaR/ES Confidence (%)", 90, 99, 99,
    help="Confidence level for VaR and Expected Shortfall")
conf = confidence / 100.0

# ============================================================================
# DATA LOADING
# ============================================================================
st.markdown("")

if mode == "Single Stock / Index":
    with st.spinner(f"⚡ Fetching {selected_name} data from Yahoo Finance..."):
        returns = fetch_returns(ticker, period)

    if returns is None or len(returns) < 100:
        st.error(f"❌ Unable to fetch data for {selected_name}. Please check your internet connection or try a different instrument.")
        st.stop()

    returns_arr = returns.values
    losses_arr  = -returns_arr

    # ── Live Stats Bar ────────────────────────────────────────────────────────
    section_title("⚡ Live Data Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: metric_card("Instrument", selected_name, f"{len(returns_arr)} obs")
    with c2: metric_card("Period", period.upper(), f"{returns.index[0].strftime('%b %Y')} → Now")
    with c3: metric_card("Mean Return", f"{returns_arr.mean():.3f}%", "Daily log return")
    with c4: metric_card("Volatility", f"{returns_arr.std():.3f}%", "Daily std dev")
    with c5: metric_card("Worst Day", f"{returns_arr.min():.2f}%", f"{returns.index[returns_arr.argmin()].strftime('%d %b %Y')}", color=COLORS['danger'])
    with c6: metric_card("Best Day", f"{returns_arr.max():.2f}%", f"{returns.index[returns_arr.argmax()].strftime('%d %b %Y')}", color=COLORS['success'])

    # ── Fit EVT Models ────────────────────────────────────────────────────────
    block_max = block_maxima_losses(returns_arr, block_size)
    gev_fit   = fit_gev(block_max)

    threshold, exceedances = pot_exceedances(losses_arr, threshold_pct)
    gpd_fit   = fit_gpd(exceedances) if len(exceedances) >= 10 else {'success': False, 'error': 'Too few exceedances'}

    hist_var, hist_es   = historical_var_es(returns_arr, conf)
    para_var, para_es   = parametric_var_es(returns_arr, conf)
    evt_var = gpd_var(conf, len(losses_arr), len(exceedances), threshold, gpd_fit) if gpd_fit['success'] else np.nan
    evt_es  = gpd_es(conf, len(losses_arr), len(exceedances), threshold, gpd_fit)  if gpd_fit['success'] else np.nan

    # ── Key EVT Metrics Bar ───────────────────────────────────────────────────
    section_title(f"📊 EVT Risk Metrics — {confidence}% Confidence")
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    with mc1: metric_card("EVT VaR (POT)", f"{evt_var:.3f}%", "GPD POT method", color=COLORS['danger'])
    with mc2: metric_card("EVT ES (POT)", f"{evt_es:.3f}%", "Expected Shortfall", color=COLORS['danger'])
    with mc3: metric_card("Historical VaR", f"{hist_var:.3f}%", "Simulation", color=COLORS['warning'])
    with mc4: metric_card("Historical ES", f"{hist_es:.3f}%", "Simulation", color=COLORS['warning'])
    with mc5: metric_card("Normal VaR", f"{para_var:.3f}%", "Parametric", color=COLORS['light_blue'])
    with mc6: metric_card("Normal ES", f"{para_es:.3f}%", "Parametric", color=COLORS['light_blue'])

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Returns & Distribution",
        "🏔️ Block Maxima (GEV)",
        "🎯 Peaks-Over-Threshold",
        "📊 VaR Comparison",
        "🔍 Tail Analysis",
        "📚 Theory & Formulae",
        "🎓 EVT Education Hub",
    ])

    # ══ TAB 0: RETURNS & DISTRIBUTION ════════════════════════════════════════
    with tab0:
        section_title("📈 Returns Time Series & Distribution")

        fig_ret = make_subplots(rows=2, cols=2,
            subplot_titles=["Daily Log Returns (%)", "Return Distribution",
                            "Rolling 21d Volatility (%)", "QQ Plot vs Normal"],
            vertical_spacing=0.15, horizontal_spacing=0.1)

        # Returns time series
        ret_dates = returns.index.tolist()
        colors_ret = [COLORS['danger'] if r < 0 else COLORS['success'] for r in returns_arr]
        fig_ret.add_trace(go.Scatter(
            x=ret_dates, y=returns_arr, mode='lines',
            line=dict(color=COLORS['accent_gold'], width=0.8), name='Returns',
            fill='tozeroy', fillcolor='rgba(255,215,0,0.05)'),
            row=1, col=1)

        # Histogram with KDE
        hist_bins = np.linspace(returns_arr.min(), returns_arr.max(), 60)
        hist_counts, hist_edges = np.histogram(returns_arr, bins=hist_bins, density=True)
        fig_ret.add_trace(go.Bar(
            x=hist_edges[:-1], y=hist_counts, width=np.diff(hist_edges),
            marker_color=COLORS['medium_blue'],
            marker_line=dict(color=COLORS['light_blue'], width=0.5),
            opacity=0.7, name='Empirical'),
            row=1, col=2)
        # Normal overlay
        x_range = np.linspace(returns_arr.min(), returns_arr.max(), 200)
        normal_pdf = norm.pdf(x_range, returns_arr.mean(), returns_arr.std())
        fig_ret.add_trace(go.Scatter(x=x_range, y=normal_pdf, name='Normal',
            line=dict(color=COLORS['danger'], width=2, dash='dash')), row=1, col=2)

        # Rolling volatility
        rolling_vol = pd.Series(returns_arr).rolling(21).std().values
        fig_ret.add_trace(go.Scatter(
            x=ret_dates, y=rolling_vol,
            line=dict(color=COLORS['light_blue'], width=1.5), name='Rolling Vol',
            fill='tozeroy', fillcolor='rgba(173,216,230,0.05)'),
            row=2, col=1)

        # QQ Plot
        osm, osr = stats.probplot(returns_arr, dist='norm')[:2]
        qq_theoretical = osm[0]
        qq_observed    = osm[1]
        fig_ret.add_trace(go.Scatter(x=qq_theoretical, y=qq_observed,
            mode='markers', marker=dict(color=COLORS['accent_gold'], size=3, opacity=0.6),
            name='QQ'), row=2, col=2)
        qq_line = np.array([qq_theoretical.min(), qq_theoretical.max()])
        # Fit line
        slope, intercept = np.polyfit(qq_theoretical, qq_observed, 1)
        fig_ret.add_trace(go.Scatter(x=qq_line, y=slope * qq_line + intercept,
            mode='lines', line=dict(color=COLORS['danger'], width=1.5, dash='dash'),
            name='Normal Line'), row=2, col=2)

        fig_ret.update_layout(height=620, showlegend=False,
            title=dict(text=f"{selected_name} — Return Diagnostics",
                       font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=14)),
            **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis']})
        fig_ret.update_xaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
        fig_ret.update_yaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
        for ann in fig_ret.layout.annotations:
            ann.font.color = COLORS['accent_gold']
        st.plotly_chart(fig_ret, use_container_width=True)

        # Descriptive stats
        section_title("📋 Descriptive Statistics")
        skewness = stats.skew(returns_arr)
        kurtosis = stats.kurtosis(returns_arr)  # excess kurtosis
        jb_stat, jb_p = stats.jarque_bera(returns_arr)
        ad_stat, ad_crit, ad_sig = stats.anderson(returns_arr, dist='norm')

        stats_data = {
            "Statistic": ["Mean", "Median", "Std Dev", "Skewness", "Excess Kurtosis",
                          "Min (Worst)", "Max (Best)", "5th Percentile", "1st Percentile",
                          "JB Statistic", "JB p-value", "Normality"],
            "Value": [
                f"{returns_arr.mean():.4f}%",
                f"{np.median(returns_arr):.4f}%",
                f"{returns_arr.std():.4f}%",
                f"{skewness:.4f}",
                f"{kurtosis:.4f}",
                f"{returns_arr.min():.4f}%",
                f"{returns_arr.max():.4f}%",
                f"{np.percentile(returns_arr, 5):.4f}%",
                f"{np.percentile(returns_arr, 1):.4f}%",
                f"{jb_stat:.2f}",
                f"{jb_p:.6f}",
                "❌ Non-Normal" if jb_p < 0.05 else "✅ Normal"
            ],
            "Interpretation": [
                "Average daily log return", "Median daily return",
                "Daily return volatility (standard deviation)",
                "Negative = left-skewed (fat left tail)" if skewness < 0 else "Positive = right-skewed",
                f"{'Fat tails (leptokurtic)' if kurtosis > 0 else 'Thin tails'} — Normal = 0",
                "Worst single-day loss", "Best single-day gain",
                "VaR at 95% (Historical)", "VaR at 99% (Historical)",
                "Jarque-Bera test statistic", "JB p-value (< 0.05 = reject normality)",
                f"{'Non-normal — EVT approach justified' if jb_p < 0.05 else 'Approximately normal'}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # ══ TAB 1: BLOCK MAXIMA (GEV) ═════════════════════════════════════════════
    with tab1:
        section_title("🏔️ Block Maxima Method — GEV Distribution")

        col_gev1, col_gev2 = st.columns([2, 1])

        with col_gev2:
            if gev_fit['success']:
                info_box(f"""
                <b>GEV Parameters:</b><br>
                ξ (Shape) = <b style="color:{COLORS['accent_gold']};">{gev_fit['xi']:.4f}</b><br>
                μ (Location) = {gev_fit['loc']:.4f}<br>
                σ (Scale) = {gev_fit['scale']:.4f}<br><br>
                <b>Distribution Type:</b> {gev_fit['type']}<br>
                <b>Blocks:</b> {gev_fit['n_blocks']} ({block_size}-day blocks)<br><br>
                <b>KS Test:</b> stat={gev_fit['ks_stat']:.4f}, p={gev_fit['ks_p']:.4f}<br>
                {'✅ Good fit (p>0.05)' if gev_fit['ks_p'] > 0.05 else '⚠️ Check fit quality'}
                """, title="GEV Fit Results")

                # Return levels
                return_periods = [2, 5, 10, 20, 50, 100]
                section_title("📋 GEV Return Levels")
                rl_data = []
                for rp in return_periods:
                    p_exceed = 1 - 1/rp  # probability of non-exceedance per block
                    rl = gev_quantile(p_exceed, gev_fit)
                    rl_data.append({
                        "Return Period (months)": rp,
                        "Max Loss (%)": f"{rl:.3f}%",
                        "Annualised VaR": f"{rl:.3f}% ({block_size}d horizon)",
                    })
                st.dataframe(pd.DataFrame(rl_data), use_container_width=True, hide_index=True)
            else:
                st.error(f"GEV fitting failed: {gev_fit.get('error','Unknown')}")

        with col_gev1:
            if gev_fit['success']:
                fig_gev = make_subplots(rows=2, cols=2,
                    subplot_titles=["Block Maxima Series", "GEV Histogram Fit",
                                    "GEV Probability Plot", "Return Level Plot"],
                    vertical_spacing=0.18, horizontal_spacing=0.12)

                # Block maxima series
                block_indices = np.arange(len(block_max)) * block_size
                fig_gev.add_trace(go.Scatter(x=block_indices, y=block_max,
                    mode='markers+lines',
                    marker=dict(color=COLORS['accent_gold'], size=5),
                    line=dict(color=COLORS['accent_gold'], width=1),
                    name='Block Max Loss'), row=1, col=1)
                fig_gev.add_hline(y=block_max.mean(), line_dash="dash",
                    line_color=COLORS['danger'], line_width=1.5, row=1, col=1)

                # GEV histogram vs fitted
                x_bm = np.linspace(block_max.min()*0.9, block_max.max()*1.1, 200)
                gev_pdf = genextreme.pdf(x_bm, gev_fit['xi'], gev_fit['loc'], gev_fit['scale'])
                h_counts, h_edges = np.histogram(block_max, bins=20, density=True)
                fig_gev.add_trace(go.Bar(x=h_edges[:-1], y=h_counts, width=np.diff(h_edges),
                    marker_color='rgba(0,77,128,0.6)',
                    marker_line=dict(color=COLORS['light_blue'], width=0.5),
                    name='Empirical'), row=1, col=2)
                fig_gev.add_trace(go.Scatter(x=x_bm, y=gev_pdf,
                    line=dict(color=COLORS['accent_gold'], width=2.5),
                    name='GEV Fit'), row=1, col=2)

                # Probability plot
                sorted_bm = np.sort(block_max)
                n_bm = len(sorted_bm)
                empirical_p = (np.arange(1, n_bm+1) - 0.5) / n_bm
                theoretical_p = genextreme.cdf(sorted_bm, gev_fit['xi'], gev_fit['loc'], gev_fit['scale'])
                fig_gev.add_trace(go.Scatter(x=theoretical_p, y=empirical_p,
                    mode='markers', marker=dict(color=COLORS['accent_gold'], size=5),
                    name='GEV PP'), row=2, col=1)
                fig_gev.add_trace(go.Scatter(x=[0,1], y=[0,1],
                    line=dict(color=COLORS['danger'], dash='dash', width=1.5),
                    name='45° Line'), row=2, col=1)

                # Return level plot
                return_p = np.linspace(0.51, 0.999, 200)
                return_levels = [gev_quantile(p, gev_fit) for p in return_p]
                return_periods_plot = 1 / (1 - return_p)
                fig_gev.add_trace(go.Scatter(x=return_periods_plot, y=return_levels,
                    line=dict(color=COLORS['accent_gold'], width=2.5),
                    name='Return Level'), row=2, col=2)
                # Scatter observed
                obs_periods = 1 / (1 - empirical_p)
                fig_gev.add_trace(go.Scatter(x=obs_periods, y=sorted_bm,
                    mode='markers',
                    marker=dict(color=COLORS['light_blue'], size=4, opacity=0.7),
                    name='Observed'), row=2, col=2)

                fig_gev.update_layout(height=600, showlegend=False,
                    title=dict(text=f"GEV Block Maxima Analysis — {selected_name} ({block_size}-day blocks)",
                               font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=13)),
                    **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis']})
                fig_gev.update_xaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
                fig_gev.update_yaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
                for ann in fig_gev.layout.annotations:
                    ann.font.color = COLORS['accent_gold']
                st.plotly_chart(fig_gev, use_container_width=True)

    # ══ TAB 2: PEAKS OVER THRESHOLD ══════════════════════════════════════════
    with tab2:
        section_title("🎯 Peaks-Over-Threshold (POT) — GPD Fitting")

        # Mean excess plot for threshold selection
        section_title("🔎 Mean Excess Plot — Threshold Selection")
        me_thresholds, me_vals, me_ci_lower, me_ci_upper = mean_excess(losses_arr)

        fig_me = go.Figure()
        fig_me.add_trace(go.Scatter(x=me_thresholds, y=me_ci_upper,
            fill=None, mode='lines', line=dict(color='rgba(255,215,0,0.1)', width=0),
            showlegend=False))
        fig_me.add_trace(go.Scatter(x=me_thresholds, y=me_ci_lower,
            fill='tonexty', mode='lines', line=dict(color='rgba(255,215,0,0.1)', width=0),
            fillcolor='rgba(255,215,0,0.08)', showlegend=False))
        fig_me.add_trace(go.Scatter(x=me_thresholds, y=me_vals,
            line=dict(color=COLORS['accent_gold'], width=2.5),
            name='Mean Excess'))
        fig_me.add_vline(x=threshold, line_dash="dash", line_color=COLORS['danger'],
            line_width=2,
            annotation_text=f"Selected u={threshold:.2f}%",
            annotation_font_color=COLORS['danger'])
        fig_me.update_layout(height=300,
            title=dict(text="Mean Excess Function — Linear region suggests suitable threshold",
                       font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=13)),
            xaxis=dict(title="Threshold u (%)", gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
            yaxis=dict(title="Mean Excess e(u)", gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
            **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis','title']})
        st.plotly_chart(fig_me, use_container_width=True)

        info_box(
            "The mean excess function e(u) = E[X−u | X>u] should be <b>approximately linear</b> "
            "in the region where GPD is a good fit. A positively-sloped linear section indicates "
            "heavy-tailed behaviour (ξ > 0). Choose a threshold at the start of the linear region.",
            title="📌 Threshold Selection Guide"
        )

        # GPD Fit
        col_gpd1, col_gpd2 = st.columns([2, 1])

        with col_gpd2:
            if gpd_fit['success']:
                info_box(f"""
                <b>GPD Parameters:</b><br>
                ξ (Shape) = <b style="color:{COLORS['accent_gold']};">{gpd_fit['xi']:.4f}</b><br>
                σ (Scale) = {gpd_fit['scale']:.4f}<br><br>
                <b>Threshold u:</b> {threshold:.4f}%<br>
                <b>Exceedances:</b> {gpd_fit['n_exc']} of {len(losses_arr)}<br>
                <b>Exc. Rate:</b> {gpd_fit['n_exc']/len(losses_arr)*100:.2f}%<br><br>
                <b>KS Test:</b> stat={gpd_fit['ks_stat']:.4f}, p={gpd_fit['ks_p']:.4f}<br>
                {'✅ Good fit (p>0.05)' if gpd_fit['ks_p'] > 0.05 else '⚠️ Marginal fit — try different threshold'}
                """, title="GPD Fit Results")

                # VaR at multiple confidence levels
                section_title("📋 EVT VaR & ES")
                conf_levels = [0.90, 0.95, 0.99, 0.995, 0.999]
                risk_data = []
                for cl in conf_levels:
                    v_evt = gpd_var(cl, len(losses_arr), len(exceedances), threshold, gpd_fit)
                    e_evt = gpd_es(cl, len(losses_arr), len(exceedances), threshold, gpd_fit)
                    v_hist, e_hist = historical_var_es(returns_arr, cl)
                    v_norm, e_norm = parametric_var_es(returns_arr, cl)
                    risk_data.append({
                        "Confidence": f"{cl*100:.1f}%",
                        "EVT VaR": f"{v_evt:.3f}%",
                        "EVT ES": f"{e_evt:.3f}%",
                        "Hist VaR": f"{v_hist:.3f}%",
                        "Normal VaR": f"{v_norm:.3f}%",
                    })
                st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

        with col_gpd1:
            if gpd_fit['success']:
                fig_gpd = make_subplots(rows=2, cols=2,
                    subplot_titles=["Exceedances Distribution", "GPD Histogram Fit",
                                    "GPD Probability Plot", "Tail Distribution"],
                    vertical_spacing=0.18, horizontal_spacing=0.12)

                # Exceedances scatter
                exc_indices = np.where(losses_arr > threshold)[0]
                fig_gpd.add_trace(go.Scatter(
                    x=np.arange(len(losses_arr)), y=losses_arr,
                    mode='markers', marker=dict(color='rgba(173,216,230,0.2)', size=2),
                    name='All Losses'), row=1, col=1)
                fig_gpd.add_trace(go.Scatter(
                    x=exc_indices, y=losses_arr[exc_indices],
                    mode='markers', marker=dict(color=COLORS['danger'], size=5),
                    name='Exceedances'), row=1, col=1)
                fig_gpd.add_hline(y=threshold, line_dash="dash",
                    line_color=COLORS['accent_gold'], line_width=1.5, row=1, col=1)

                # GPD histogram
                x_exc = np.linspace(0, exceedances.max()*1.1, 200)
                gpd_pdf = genpareto.pdf(x_exc, gpd_fit['xi'], gpd_fit['loc'], gpd_fit['scale'])
                h_exc, h_edges = np.histogram(exceedances, bins=25, density=True)
                fig_gpd.add_trace(go.Bar(x=h_edges[:-1], y=h_exc, width=np.diff(h_edges),
                    marker_color='rgba(0,77,128,0.6)',
                    marker_line=dict(color=COLORS['light_blue'], width=0.5),
                    name='Exceedances'), row=1, col=2)
                fig_gpd.add_trace(go.Scatter(x=x_exc, y=gpd_pdf,
                    line=dict(color=COLORS['accent_gold'], width=2.5),
                    name='GPD Fit'), row=1, col=2)

                # Probability plot
                sorted_exc = np.sort(exceedances)
                n_exc_pp = len(sorted_exc)
                emp_p = (np.arange(1, n_exc_pp+1) - 0.5) / n_exc_pp
                theo_p = genpareto.cdf(sorted_exc, gpd_fit['xi'], gpd_fit['loc'], gpd_fit['scale'])
                fig_gpd.add_trace(go.Scatter(x=theo_p, y=emp_p,
                    mode='markers', marker=dict(color=COLORS['accent_gold'], size=5),
                    name='GPD PP'), row=2, col=1)
                fig_gpd.add_trace(go.Scatter(x=[0,1], y=[0,1],
                    line=dict(color=COLORS['danger'], dash='dash', width=1.5)),
                    row=2, col=1)

                # Tail loss distribution with VaR/ES markers
                conf_range = np.linspace(0.9, 0.999, 200)
                var_line = [gpd_var(c, len(losses_arr), len(exceedances), threshold, gpd_fit) for c in conf_range]
                fig_gpd.add_trace(go.Scatter(x=conf_range*100, y=var_line,
                    line=dict(color=COLORS['accent_gold'], width=2.5),
                    name='EVT VaR'), row=2, col=2)
                fig_gpd.add_vline(x=conf*100, line_dash="dash",
                    line_color=COLORS['danger'], line_width=2, row=2, col=2)
                fig_gpd.add_hline(y=evt_var, line_dash="dot",
                    line_color=COLORS['danger'], line_width=1.5, row=2, col=2)

                fig_gpd.update_layout(height=600, showlegend=False,
                    title=dict(text=f"GPD Peaks-Over-Threshold — {selected_name} | u={threshold:.2f}%",
                               font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=13)),
                    **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis']})
                fig_gpd.update_xaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
                fig_gpd.update_yaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
                for ann in fig_gpd.layout.annotations:
                    ann.font.color = COLORS['accent_gold']
                st.plotly_chart(fig_gpd, use_container_width=True)

    # ══ TAB 3: VAR COMPARISON ════════════════════════════════════════════════
    with tab3:
        section_title("📊 VaR & ES Method Comparison")

        conf_levels_cmp = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
        evt_vars, evt_ess, hist_vars, hist_ess, norm_vars, norm_ess = [], [], [], [], [], []

        for cl in conf_levels_cmp:
            if gpd_fit['success']:
                evt_vars.append(gpd_var(cl, len(losses_arr), len(exceedances), threshold, gpd_fit))
                evt_ess.append(gpd_es(cl, len(losses_arr), len(exceedances), threshold, gpd_fit))
            else:
                evt_vars.append(np.nan)
                evt_ess.append(np.nan)
            hv, he = historical_var_es(returns_arr, cl)
            hist_vars.append(hv); hist_ess.append(he)
            nv, ne = parametric_var_es(returns_arr, cl)
            norm_vars.append(nv); norm_ess.append(ne)

        conf_pcts = [c*100 for c in conf_levels_cmp]

        fig_cmp = make_subplots(rows=1, cols=2,
            subplot_titles=["VaR Comparison", "Expected Shortfall Comparison"])

        for y, name, color, dash in [
            (evt_vars, "EVT (POT)", COLORS['danger'], 'solid'),
            (hist_vars, "Historical", COLORS['warning'], 'dash'),
            (norm_vars, "Normal", COLORS['light_blue'], 'dot')
        ]:
            fig_cmp.add_trace(go.Scatter(x=conf_pcts, y=y, name=name,
                line=dict(color=color, width=2.5, dash=dash), mode='lines+markers',
                marker=dict(size=6)), row=1, col=1)

        for y, name, color, dash in [
            (evt_ess, "EVT ES (POT)", COLORS['danger'], 'solid'),
            (hist_ess, "Historical ES", COLORS['warning'], 'dash'),
            (norm_ess, "Normal ES", COLORS['light_blue'], 'dot')
        ]:
            fig_cmp.add_trace(go.Scatter(x=conf_pcts, y=y, name=name,
                line=dict(color=color, width=2.5, dash=dash), mode='lines+markers',
                marker=dict(size=6), showlegend=False), row=1, col=2)

        fig_cmp.add_vline(x=confidence, line_dash="dash",
            line_color=COLORS['accent_gold'], line_width=1.5, row=1, col=1)
        fig_cmp.add_vline(x=confidence, line_dash="dash",
            line_color=COLORS['accent_gold'], line_width=1.5, row=1, col=2)

        fig_cmp.update_layout(height=450,
            title=dict(text=f"{selected_name} — EVT vs Historical vs Normal",
                       font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=14)),
            **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis']})
        fig_cmp.update_xaxes(title_text="Confidence Level (%)",
            gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
        fig_cmp.update_yaxes(title_text="Loss (%)",
            gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
        for ann in fig_cmp.layout.annotations:
            ann.font.color = COLORS['accent_gold']
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Comparison table
        section_title("📋 VaR & ES Summary Table")
        cmp_table = []
        for cl, ev, ee, hv, he, nv, ne in zip(conf_levels_cmp, evt_vars, evt_ess, hist_vars, hist_ess, norm_vars, norm_ess):
            diff_hist  = ev - hv if not np.isnan(ev) else np.nan
            diff_norm  = ev - nv if not np.isnan(ev) else np.nan
            cmp_table.append({
                "Confidence": f"{cl*100:.1f}%",
                "EVT VaR": f"{ev:.3f}%" if not np.isnan(ev) else "N/A",
                "EVT ES": f"{ee:.3f}%" if not np.isnan(ee) else "N/A",
                "Historical VaR": f"{hv:.3f}%",
                "Normal VaR": f"{nv:.3f}%",
                "EVT vs Hist ΔVaR": f"{diff_hist:+.3f}%" if not np.isnan(diff_hist) else "N/A",
                "EVT vs Normal ΔVaR": f"{diff_norm:+.3f}%" if not np.isnan(diff_norm) else "N/A",
            })
        st.dataframe(pd.DataFrame(cmp_table), use_container_width=True, hide_index=True)

        info_box(
            f"<b>EVT advantage:</b> At extreme confidence levels (99.5%, 99.9%), EVT systematically "
            f"produces <b>higher (more conservative) VaR estimates</b> than both historical simulation "
            f"and normal parametric methods. This is because EVT explicitly models the tail behaviour — "
            f"it does not extrapolate from the centre of the distribution. For {selected_name}, "
            f"EVT {confidence}% VaR = <b>{evt_var:.3f}%</b> vs Normal = <b>{para_var:.3f}%</b> "
            f"({'higher' if evt_var > para_var else 'lower'} by {abs(evt_var-para_var):.3f}%).",
            title="💡 EVT vs Other Methods"
        )

    # ══ TAB 4: TAIL ANALYSIS ════════════════════════════════════════════════
    with tab4:
        section_title("🔍 Tail Analysis — Extreme Loss Events")

        # Hill estimator
        section_title("📊 Hill Estimator (Tail Index α)")
        sorted_losses_desc = np.sort(losses_arr)[::-1]
        k_range = np.arange(10, min(200, len(sorted_losses_desc)//2))
        hill_estimates = []
        for k in k_range:
            top_k = sorted_losses_desc[:k]
            hill_k = k / np.sum(np.log(top_k / sorted_losses_desc[k]))
            hill_estimates.append(hill_k)

        fig_hill = go.Figure()
        fig_hill.add_trace(go.Scatter(
            x=k_range, y=hill_estimates,
            line=dict(color=COLORS['accent_gold'], width=2),
            fill='tozeroy', fillcolor='rgba(255,215,0,0.05)'))
        fig_hill.add_hline(y=np.median(hill_estimates),
            line_dash="dash", line_color=COLORS['danger'], line_width=1.5,
            annotation_text=f"Median α = {np.median(hill_estimates):.2f}",
            annotation_font_color=COLORS['danger'])
        fig_hill.update_layout(height=300,
            title=dict(text="Hill Estimator of Tail Index α (stable region = reliable estimate)",
                       font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=13)),
            xaxis=dict(title="k (number of order statistics)",
                       gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
            yaxis=dict(title="Hill Estimate of α",
                       gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
            **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis','title']})
        st.plotly_chart(fig_hill, use_container_width=True)

        alpha_hill = np.median(hill_estimates)
        info_box(f"""
        <b>Tail Index α (Hill) = {alpha_hill:.2f}</b> (from {k_range[0]}–{k_range[-1]} order statistics)<br>
        • α > 4: thin tails (normal-like); α ∈ [2,4]: moderately heavy; α < 2: very heavy (infinite variance)<br>
        • ξ (GPD shape) ≈ 1/α = <b>{1/alpha_hill:.4f}</b> — consistent with GPD shape = {gpd_fit.get('xi', 'N/A') if gpd_fit['success'] else 'N/A'}<br>
        • <b>Interpretation:</b> {'Very heavy tail — extreme losses occur more frequently than normal predicts' if alpha_hill < 2 else 'Moderately heavy tail' if alpha_hill < 4 else 'Relatively thin tail'}
        """, title="Hill Estimator Interpretation")

        # Extreme loss table
        section_title("📋 Top 20 Extreme Loss Events")
        worst_idx = np.argsort(returns_arr)[:20]
        extreme_df = pd.DataFrame({
            "Rank": np.arange(1, 21),
            "Date": [returns.index[i].strftime('%d %b %Y') for i in worst_idx],
            "Loss (%)": [f"{-returns_arr[i]:.3f}%" for i in worst_idx],
            "Return": [f"{returns_arr[i]:.3f}%" for i in worst_idx],
            "Exc. Threshold?": ["✅ Yes" if -returns_arr[i] > threshold else "❌ No" for i in worst_idx],
            "z-Score": [f"{(returns_arr[i]-returns_arr.mean())/returns_arr.std():.2f}σ" for i in worst_idx],
        })
        st.dataframe(extreme_df, use_container_width=True, hide_index=True)

        # Loss distribution tail zoom
        section_title("🔍 Tail Zoom — Loss Distribution")
        loss_range = np.linspace(threshold * 0.5, losses_arr.max() * 1.1, 300)
        fig_tail = go.Figure()
        h_l, h_e = np.histogram(losses_arr, bins=100, density=True)
        fig_tail.add_trace(go.Bar(x=h_e[:-1], y=h_l, width=np.diff(h_e),
            marker_color='rgba(0,77,128,0.5)',
            marker_line=dict(color=COLORS['light_blue'], width=0.3),
            name='Empirical'))
        if gpd_fit['success']:
            # GPD tail density (f(x) = n_exc/n * gpd.pdf(x-u) for x > u)
            x_tail = np.linspace(threshold, losses_arr.max()*1.1, 200)
            gpd_tail_pdf = (len(exceedances)/len(losses_arr)) * \
                genpareto.pdf(x_tail - threshold, gpd_fit['xi'], gpd_fit['loc'], gpd_fit['scale'])
            fig_tail.add_trace(go.Scatter(x=x_tail, y=gpd_tail_pdf,
                line=dict(color=COLORS['danger'], width=2.5),
                name='GPD Tail Fit'))
        norm_pdf_losses = norm.pdf(loss_range, -returns_arr.mean(), returns_arr.std())
        fig_tail.add_trace(go.Scatter(x=loss_range, y=norm_pdf_losses,
            line=dict(color=COLORS['light_blue'], width=1.5, dash='dash'),
            name='Normal'))
        fig_tail.add_vline(x=threshold, line_dash="dash",
            line_color=COLORS['accent_gold'], line_width=1.5,
            annotation_text=f"Threshold {threshold:.2f}%",
            annotation_font_color=COLORS['accent_gold'])
        fig_tail.add_vline(x=evt_var if not np.isnan(evt_var) else threshold,
            line_dash="dot", line_color=COLORS['danger'], line_width=2,
            annotation_text=f"EVT VaR {evt_var:.2f}%",
            annotation_font_color=COLORS['danger'])
        fig_tail.update_layout(height=380,
            title=dict(text=f"Tail Region — EVT vs Normal Density | {selected_name}",
                       font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=13)),
            xaxis=dict(title="Daily Loss (%)", range=[threshold*0.3, losses_arr.max()*1.05],
                       gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
            yaxis=dict(title="Density", gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
            **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis','title']})
        st.plotly_chart(fig_tail, use_container_width=True)

    # ══ TAB 5: THEORY ════════════════════════════════════════════════════════
    with tab5:
        section_title("📚 EVT Theory & Formulae")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            info_box("""
            <b>Fisher-Tippett-Gnedenko Theorem:</b><br>
            For any i.i.d. random variables X₁,...,Xₙ, if the normalised block maxima converge,
            they converge to the <b>Generalised Extreme Value (GEV)</b> distribution:<br><br>
            <code>G(x; ξ, μ, σ) = exp{−[1 + ξ(x−μ)/σ]<sup>−1/ξ</sup>}</code><br><br>
            • <b>ξ = 0:</b> Gumbel (light tails — exponential decay)<br>
            • <b>ξ > 0:</b> Fréchet (heavy tails — power law decay)<br>
            • <b>ξ < 0:</b> Weibull (bounded upper tail)<br><br>
            <b>For financial returns:</b> ξ > 0 (Fréchet domain) is empirically universal —
            equity returns have fat tails / heavy tails.
            """, title="Block Maxima — GEV Distribution")

            info_box("""
            <b>Pickands-Balkema-de Haan Theorem:</b><br>
            For sufficiently high threshold u, excess losses Y = X − u | X > u
            converge to the <b>Generalised Pareto Distribution (GPD)</b>:<br><br>
            <code>F(y; ξ, σ) = 1 − (1 + ξy/σ)<sup>−1/ξ</sup>, &nbsp; y > 0</code><br><br>
            • ξ > 0: heavy tail (Pareto-type); ξ = 0: exponential; ξ < 0: bounded<br><br>
            <b>EVT VaR (POT method):</b><br>
            <code>VaRₚ = u + (σ/ξ)[(n/Nᵤ·(1−p))⁻ξ − 1]</code><br><br>
            <b>EVT Expected Shortfall:</b><br>
            <code>ESₚ = VaRₚ/(1−ξ) + (σ − ξu)/(1−ξ)</code>
            """, title="Peaks-Over-Threshold — GPD Distribution")

        with col_t2:
            info_box("""
            <b>Why EVT for Financial Risk?</b><br><br>
            Standard VaR (Normal assumption) systematically <b>underestimates</b> tail risk because:<br>
            1. Financial returns are leptokurtic (excess kurtosis > 0)<br>
            2. Left tail is fatter than Gaussian<br>
            3. Normal distribution extrapolation at 99.9% is unreliable<br><br>
            <b>EVT provides a statistically rigorous framework</b> for modelling the tail — 
            it is the only asymptotically justified method for extreme quantile estimation.<br><br>
            <b>Basel III / IV link:</b> Stressed ES at 97.5% replaces VaR at 99% as the
            primary regulatory capital measure. EVT-ES is more conservative and better 
            captures tail scenarios used in Internal Models Approach (IMA).
            """, title="EVT for Risk Management")

            info_box("""
            <b>Key EVT Parameters and Interpretation:</b><br><br>
            <b>Shape parameter ξ (xi):</b><br>
            • Most important: determines tail heaviness<br>
            • Indian equities: ξ ≈ 0.1–0.4 (Fréchet domain)<br>
            • Higher ξ → heavier tail → higher VaR/ES<br><br>
            <b>Scale parameter σ:</b> Spread of exceedances<br>
            <b>Location parameter μ (GEV):</b> Level of block maxima<br><br>
            <b>Hill Estimator:</b> α̂ = k / Σ log(Xᵢ/X_(k+1))<br>
            Tail index α = 1/ξ. Stable plateau in Hill plot → reliable ξ estimate.
            """, title="Parameter Interpretation")

        section_title("🐍 Python Implementation")
        st.code("""
from scipy.stats import genpareto, genextreme
import numpy as np

# ── Block Maxima / GEV ──────────────────────────────────────────────────
def block_maxima_losses(returns, block_size=21):
    losses = -returns
    n_blocks = len(losses) // block_size
    return [losses[i*block_size:(i+1)*block_size].max() for i in range(n_blocks)]

# Fit GEV
block_max = block_maxima_losses(log_returns, block_size=21)
xi, loc, scale = genextreme.fit(np.array(block_max))
print(f"GEV: xi={xi:.4f}, loc={loc:.4f}, scale={scale:.4f}")

# Return level for 100-month return period
rl_100 = genextreme.ppf(1 - 1/100, xi, loc, scale)
print(f"100-month return level: {rl_100:.3f}%")

# ── POT / GPD ────────────────────────────────────────────────────────────
losses = -log_returns * 100  # to percentage
threshold = np.percentile(losses, 95)  # 95th percentile threshold
exceedances = losses[losses > threshold] - threshold

xi_gpd, loc_gpd, scale_gpd = genpareto.fit(exceedances, floc=0)

def evt_var(p, n, n_u, u, xi, scale):
    if xi == 0:
        return u + scale * np.log(n/n_u * (1-p))
    return u + (scale/xi) * ((n/n_u*(1-p))**(-xi) - 1)

def evt_es(p, n, n_u, u, xi, scale):
    var = evt_var(p, n, n_u, u, xi, scale)
    return (var + scale - xi*u) / (1 - xi)

n, n_u = len(losses), len(exceedances)
var_99 = evt_var(0.99, n, n_u, threshold, xi_gpd, scale_gpd)
es_99  = evt_es(0.99, n, n_u, threshold, xi_gpd, scale_gpd)
print(f"EVT VaR(99%) = {var_99:.3f}%  |  EVT ES(99%) = {es_99:.3f}%")
        """, language='python')

    # ══ TAB 6: EVT EDUCATION HUB ═════════════════════════════════════════════
    with tab6:
        section_title("🎓 EVT Concepts & Applications")

        faqs = [
            ("Why is EVT superior to normal VaR for tail risk?",
             "Normal VaR assumes returns follow a Gaussian distribution, but empirical returns exhibit "
             "excess kurtosis (fat tails) and negative skewness. At extreme confidence levels (99%, 99.9%), "
             "normal VaR severely underestimates losses. EVT directly models the tail without assumptions "
             "about the bulk of the distribution — it is the only asymptotically justified approach for "
             "extreme quantile estimation."),
            ("What is the connection between GEV and GPD?",
             "They are two equivalent approaches to EVT: GEV models the maximum of blocks of observations "
             "(Block Maxima Method), while GPD models the distribution of exceedances above a high threshold "
             "(Peaks-Over-Threshold). The shape parameter ξ is the same in both frameworks. GPD (POT) is "
             "generally preferred as it uses data more efficiently — every extreme event contributes, "
             "not just the block maximum."),
            ("How do I choose the POT threshold?",
             "The mean excess plot (MEP) is the primary diagnostic. A good threshold is where the MEP "
             "becomes approximately linear. Too low a threshold violates the GPD approximation; too high "
             "leaves too few observations for reliable parameter estimation. A rule of thumb: the 90th–95th "
             "percentile gives a good balance. Always check that the KS test p-value > 0.05."),
            ("What does the shape parameter ξ tell us about Indian equities?",
             "For NSE indices and large-cap stocks, ξ typically ranges from 0.1 to 0.4 (Fréchet domain). "
             "This confirms that Indian equity returns have heavy tails — extreme events (circuit breakers, "
             "crashes, Budget reactions, geopolitical shocks) occur more frequently than the normal "
             "distribution predicts. Higher ξ = heavier tail = larger EVT VaR vs Normal VaR divergence."),
            ("How is EVT used in Basel III/IV regulatory capital?",
             "Basel IV (FRTB — Fundamental Review of the Trading Book) replaced VaR 99% with Expected "
             "Shortfall (ES) at 97.5%, with liquidity-adjusted holding periods. EVT-based ES is more "
             "stable and captures tail events better than historical simulation ES, which depends on the "
             "specific historical sample. Banks using Internal Models Approach (IMA) are required to "
             "demonstrate comprehensive tail risk capture — EVT provides the theoretical foundation."),
            ("What is tail dependence and why does it matter for portfolios?",
             "Tail dependence measures the probability that two assets jointly experience extreme losses. "
             "Standard correlation is a measure of linear co-movement in the centre of the distribution — "
             "it dramatically underestimates joint extreme losses. During the 2008 GFC and COVID-2020 "
             "crash, most asset correlations spiked toward 1.0 even though pre-crisis correlations were "
             "moderate. EVT copula models (e.g., Gumbel copula) explicitly model upper/lower tail "
             "dependence and provide more realistic portfolio tail risk measures."),
        ]

        for q, a in faqs:
            with st.expander(f"❓ {q}"):
                st.markdown(f"""
                <div style="color:{COLORS['text_primary']};font-size:0.9rem;line-height:1.8;
                            background:rgba(0,51,102,0.3);border-radius:8px;padding:1rem;">
                    {a}
                </div>""", unsafe_allow_html=True)


# ============================================================================
# PORTFOLIO EVT MODE
# ============================================================================
else:
    with st.spinner(f"⚡ Fetching data for {len(portfolio_names)} stocks..."):
        tickers_list = [(name, NSE_STOCKS[name]) for name in portfolio_names]
        ret_df = fetch_multi_returns(tickers_list, period)

    if ret_df is None or len(ret_df) < 100:
        st.error("❌ Unable to fetch sufficient data. Please check your connection or reduce the number of stocks.")
        st.stop()

    # Portfolio returns
    port_returns = portfolio_returns(ret_df, weights)
    port_losses  = -port_returns

    # ── Portfolio Summary ─────────────────────────────────────────────────────
    section_title("⚡ Portfolio Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Stocks", str(len(portfolio_names)), f"{len(ret_df)} observations")
    with c2: metric_card("Port. Mean", f"{port_returns.mean():.3f}%", "Daily log return")
    with c3: metric_card("Port. Vol", f"{port_returns.std():.3f}%", "Daily std dev")
    with c4: metric_card("Worst Day", f"{port_returns.min():.2f}%", "Max loss", color=COLORS['danger'])
    with c5: metric_card("Skewness", f"{stats.skew(port_returns):.3f}", "Negative = left tail")

    # ── Fit EVT on Portfolio Returns ──────────────────────────────────────────
    port_block_max = block_maxima_losses(port_returns, block_size)
    port_gev       = fit_gev(port_block_max)
    port_threshold, port_exc = pot_exceedances(port_losses, threshold_pct)
    port_gpd       = fit_gpd(port_exc) if len(port_exc) >= 10 else {'success': False, 'error': 'Too few exceedances'}

    port_hist_var, port_hist_es = historical_var_es(port_returns, conf)
    port_norm_var, port_norm_es = parametric_var_es(port_returns, conf)
    port_evt_var = gpd_var(conf, len(port_losses), len(port_exc), port_threshold, port_gpd) if port_gpd['success'] else np.nan
    port_evt_es  = gpd_es(conf, len(port_losses), len(port_exc), port_threshold, port_gpd) if port_gpd['success'] else np.nan

    # Portfolio risk metrics bar
    section_title(f"📊 Portfolio EVT Risk Metrics — {confidence}% Confidence")
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    with mc1: metric_card("EVT VaR (POT)", f"{port_evt_var:.3f}%", "GPD POT method", color=COLORS['danger'])
    with mc2: metric_card("EVT ES (POT)", f"{port_evt_es:.3f}%", "Expected Shortfall", color=COLORS['danger'])
    with mc3: metric_card("Historical VaR", f"{port_hist_var:.3f}%", "Simulation", color=COLORS['warning'])
    with mc4: metric_card("Normal VaR", f"{port_norm_var:.3f}%", "Parametric", color=COLORS['light_blue'])
    with mc5: metric_card("GEV Shape ξ", f"{port_gev.get('xi', np.nan):.4f}", f"Type: {port_gev.get('type','N/A')}" if port_gev['success'] else "Fitting failed", color=COLORS['accent_gold'])
    with mc6: metric_card("GPD Shape ξ", f"{port_gpd.get('xi', np.nan):.4f}" if port_gpd['success'] else "N/A", f"Threshold {port_threshold:.2f}%", color=COLORS['accent_gold'])

    # ── Portfolio Tabs ────────────────────────────────────────────────────────
    ptab0, ptab1, ptab2, ptab3, ptab4 = st.tabs([
        "📈 Portfolio Overview",
        "🏔️ Portfolio EVT",
        "🔗 Tail Dependence",
        "⚖️ Individual EVT Comparison",
        "📊 Risk Decomposition",
    ])

    # ══ PTAB 0: PORTFOLIO OVERVIEW ═══════════════════════════════════════════
    with ptab0:
        section_title("📈 Portfolio Returns & Composition")

        col_po1, col_po2 = st.columns([2, 1])

        with col_po1:
            fig_port = make_subplots(rows=2, cols=2,
                subplot_titles=["Portfolio Daily Returns (%)", "Return Distribution",
                                "Cumulative Portfolio Return", "Correlation Heatmap"],
                vertical_spacing=0.18, horizontal_spacing=0.12)

            # Returns time series
            fig_port.add_trace(go.Scatter(
                x=ret_df.index.tolist(), y=port_returns,
                line=dict(color=COLORS['accent_gold'], width=0.9),
                fill='tozeroy', fillcolor='rgba(255,215,0,0.04)',
                name='Portfolio'), row=1, col=1)

            # Histogram
            h_p, h_pe = np.histogram(port_returns, bins=60, density=True)
            fig_port.add_trace(go.Bar(x=h_pe[:-1], y=h_p, width=np.diff(h_pe),
                marker_color='rgba(0,77,128,0.6)',
                marker_line=dict(color=COLORS['light_blue'], width=0.5),
                name='Empirical'), row=1, col=2)
            x_p = np.linspace(port_returns.min(), port_returns.max(), 200)
            fig_port.add_trace(go.Scatter(x=x_p,
                y=norm.pdf(x_p, port_returns.mean(), port_returns.std()),
                line=dict(color=COLORS['danger'], width=2, dash='dash'),
                name='Normal'), row=1, col=2)

            # Cumulative return
            cum_ret = np.cumsum(port_returns)
            fig_port.add_trace(go.Scatter(
                x=ret_df.index.tolist(), y=cum_ret,
                line=dict(color=COLORS['success'], width=1.5),
                fill='tozeroy', fillcolor='rgba(40,167,69,0.05)',
                name='Cumulative'), row=2, col=1)

            # Correlation heatmap
            corr = ret_df.corr().values
            n_c = len(portfolio_names)
            fig_port.add_trace(go.Heatmap(
                z=corr, x=portfolio_names, y=portfolio_names,
                colorscale=[[0,'#003366'],[0.5,'#ADD8E6'],[1,'#FFD700']],
                zmin=-1, zmax=1, showscale=False,
                text=[[f"{corr[i,j]:.2f}" for j in range(n_c)] for i in range(n_c)],
                texttemplate="%{text}", textfont=dict(size=9, color='white')),
                row=2, col=2)

            fig_port.update_layout(height=620, showlegend=False,
                title=dict(text=f"Portfolio EVT Analysis — {', '.join(portfolio_names[:4])}{'...' if len(portfolio_names)>4 else ''}",
                           font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=13)),
                **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis']})
            fig_port.update_xaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
            fig_port.update_yaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
            for ann in fig_port.layout.annotations:
                ann.font.color = COLORS['accent_gold']
            st.plotly_chart(fig_port, use_container_width=True)

        with col_po2:
            # Weight pie
            fig_wt = go.Figure(go.Pie(
                labels=portfolio_names,
                values=weights * 100,
                hole=0.45,
                textfont=dict(size=10, color='white'),
                marker=dict(
                    colors=[COLORS['accent_gold'], COLORS['light_blue'], COLORS['success'],
                            COLORS['danger'], '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c'][:n_stocks],
                    line=dict(color='#0f1824', width=2)
                )
            ))
            fig_wt.update_layout(
                height=280,
                title=dict(text="Portfolio Weights", font=dict(color=COLORS['accent_gold'], size=13)),
                paper_bgcolor='#0f1824',
                font=dict(color=COLORS['text_primary']),
                legend=dict(bgcolor='rgba(17,34,64,0.8)', font=dict(size=9, color=COLORS['text_primary'])),
                margin=dict(t=50, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_wt, use_container_width=True)

            # Individual stock stats
            section_title("📋 Stock Stats")
            stock_stats = []
            for name in portfolio_names:
                col_ret = ret_df[name].values
                stock_stats.append({
                    "Stock": name,
                    "Weight": f"{weights[portfolio_names.index(name)]*100:.1f}%",
                    "Vol (Daily)": f"{col_ret.std():.3f}%",
                    "Skew": f"{stats.skew(col_ret):.3f}",
                    "Kurt": f"{stats.kurtosis(col_ret):.2f}",
                    "Worst Day": f"{col_ret.min():.2f}%",
                })
            st.dataframe(pd.DataFrame(stock_stats), use_container_width=True, hide_index=True)

    # ══ PTAB 1: PORTFOLIO EVT ════════════════════════════════════════════════
    with ptab1:
        section_title("🏔️ Portfolio EVT — GEV & GPD")

        if port_gev['success'] and port_gpd['success']:
            col_pe1, col_pe2 = st.columns([2, 1])

            with col_pe2:
                info_box(f"""
                <b>Portfolio GEV:</b><br>
                ξ = {port_gev['xi']:.4f} ({port_gev['type']})<br>
                μ = {port_gev['loc']:.4f}, σ = {port_gev['scale']:.4f}<br>
                Blocks: {port_gev['n_blocks']}<br><br>
                <b>Portfolio GPD:</b><br>
                ξ = {port_gpd['xi']:.4f}<br>
                σ = {port_gpd['scale']:.4f}<br>
                Threshold: {port_threshold:.3f}%<br>
                Exceedances: {port_gpd['n_exc']}<br><br>
                <b>EVT VaR {confidence}%:</b> {port_evt_var:.3f}%<br>
                <b>EVT ES {confidence}%:</b> {port_evt_es:.3f}%
                """, title="Portfolio EVT Results")

                # Multi-level risk table
                section_title("📋 Portfolio Risk Table")
                conf_levels_p = [0.90, 0.95, 0.99, 0.995, 0.999]
                p_risk_data = []
                for cl in conf_levels_p:
                    pv = gpd_var(cl, len(port_losses), len(port_exc), port_threshold, port_gpd)
                    pe = gpd_es(cl, len(port_losses), len(port_exc), port_threshold, port_gpd)
                    hv, _ = historical_var_es(port_returns, cl)
                    nv, _ = parametric_var_es(port_returns, cl)
                    p_risk_data.append({
                        "Conf": f"{cl*100:.1f}%",
                        "EVT VaR": f"{pv:.3f}%",
                        "EVT ES": f"{pe:.3f}%",
                        "Hist VaR": f"{hv:.3f}%",
                        "Norm VaR": f"{nv:.3f}%",
                    })
                st.dataframe(pd.DataFrame(p_risk_data), use_container_width=True, hide_index=True)

            with col_pe1:
                fig_pev = make_subplots(rows=2, cols=2,
                    subplot_titles=["Block Maxima (Portfolio)", "GPD Fit (Portfolio Losses)",
                                    "VaR Methods Comparison", "Tail Loss Distribution"],
                    vertical_spacing=0.18, horizontal_spacing=0.12)

                # Block maxima
                port_bm = np.array(port_block_max) if isinstance(port_block_max, list) else port_block_max
                bm_idx = np.arange(len(port_bm)) * block_size
                fig_pev.add_trace(go.Scatter(x=bm_idx, y=port_bm,
                    mode='markers+lines',
                    marker=dict(color=COLORS['accent_gold'], size=5),
                    line=dict(color=COLORS['accent_gold'], width=1),
                    name='Block Max Loss'), row=1, col=1)

                # GPD histogram fit (portfolio)
                x_pe = np.linspace(0, port_exc.max()*1.1, 200)
                gpd_pdf_p = genpareto.pdf(x_pe, port_gpd['xi'], port_gpd['loc'], port_gpd['scale'])
                h_pexc, h_pedges = np.histogram(port_exc, bins=25, density=True)
                fig_pev.add_trace(go.Bar(x=h_pedges[:-1], y=h_pexc, width=np.diff(h_pedges),
                    marker_color='rgba(0,77,128,0.6)',
                    marker_line=dict(color=COLORS['light_blue'], width=0.5)),
                    row=1, col=2)
                fig_pev.add_trace(go.Scatter(x=x_pe, y=gpd_pdf_p,
                    line=dict(color=COLORS['accent_gold'], width=2.5),
                    name='GPD Fit'), row=1, col=2)

                # VaR comparison
                conf_range_p = np.linspace(0.90, 0.999, 100)
                p_evt_line  = [gpd_var(c, len(port_losses), len(port_exc), port_threshold, port_gpd) for c in conf_range_p]
                p_hist_line = [historical_var_es(port_returns, c)[0] for c in conf_range_p]
                p_norm_line = [parametric_var_es(port_returns, c)[0] for c in conf_range_p]
                for y, name, color, dash in [
                    (p_evt_line, "EVT", COLORS['danger'], 'solid'),
                    (p_hist_line, "Historical", COLORS['warning'], 'dash'),
                    (p_norm_line, "Normal", COLORS['light_blue'], 'dot')
                ]:
                    fig_pev.add_trace(go.Scatter(x=[c*100 for c in conf_range_p], y=y,
                        name=name, line=dict(color=color, width=2, dash=dash)),
                        row=2, col=1)

                # Tail loss distribution
                p_loss_range = np.linspace(port_threshold*0.5, port_losses.max()*1.1, 300)
                hp_l, hp_e = np.histogram(port_losses, bins=80, density=True)
                fig_pev.add_trace(go.Bar(x=hp_e[:-1], y=hp_l, width=np.diff(hp_e),
                    marker_color='rgba(0,77,128,0.5)',
                    marker_line=dict(color=COLORS['light_blue'], width=0.3)),
                    row=2, col=2)
                x_ptail = np.linspace(port_threshold, port_losses.max()*1.1, 200)
                gpd_tail_p = (len(port_exc)/len(port_losses)) * \
                    genpareto.pdf(x_ptail-port_threshold, port_gpd['xi'], port_gpd['loc'], port_gpd['scale'])
                fig_pev.add_trace(go.Scatter(x=x_ptail, y=gpd_tail_p,
                    line=dict(color=COLORS['danger'], width=2.5)), row=2, col=2)
                fig_pev.add_vline(x=port_evt_var, line_dash="dot",
                    line_color=COLORS['danger'], line_width=2, row=2, col=2)

                fig_pev.update_layout(height=600, showlegend=False,
                    title=dict(text="Portfolio EVT Analysis",
                               font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=13)),
                    **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis']})
                fig_pev.update_xaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
                fig_pev.update_yaxes(gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary'])
                for ann in fig_pev.layout.annotations:
                    ann.font.color = COLORS['accent_gold']
                st.plotly_chart(fig_pev, use_container_width=True)

    # ══ PTAB 2: TAIL DEPENDENCE ══════════════════════════════════════════════
    with ptab2:
        section_title("🔗 Tail Dependence Analysis")

        # Pairwise tail dependence matrix
        n_st = len(portfolio_names)
        tail_dep_matrix = np.zeros((n_st, n_st))
        for i in range(n_st):
            for j in range(n_st):
                if i == j:
                    tail_dep_matrix[i, j] = 1.0
                else:
                    td = compute_tail_dependence(
                        ret_df[portfolio_names[i]].values,
                        ret_df[portfolio_names[j]].values,
                        q=threshold_pct/100
                    )
                    tail_dep_matrix[i, j] = td

        fig_td = make_subplots(rows=1, cols=2,
            subplot_titles=["Tail Dependence Matrix", "Standard Correlation Matrix"])

        fig_td.add_trace(go.Heatmap(
            z=tail_dep_matrix,
            x=portfolio_names, y=portfolio_names,
            colorscale=[[0,'#003366'],[0.5,'#ADD8E6'],[1,'#dc3545']],
            zmin=0, zmax=1,
            text=[[f"{tail_dep_matrix[i,j]:.2f}" for j in range(n_st)] for i in range(n_st)],
            texttemplate="%{text}", textfont=dict(size=9, color='white'),
            showscale=True),
            row=1, col=1)

        corr_matrix = ret_df.corr().values
        fig_td.add_trace(go.Heatmap(
            z=corr_matrix,
            x=portfolio_names, y=portfolio_names,
            colorscale=[[0,'#003366'],[0.5,'#ADD8E6'],[1,'#FFD700']],
            zmin=-1, zmax=1,
            text=[[f"{corr_matrix[i,j]:.2f}" for j in range(n_st)] for i in range(n_st)],
            texttemplate="%{text}", textfont=dict(size=9, color='white'),
            showscale=True),
            row=1, col=2)

        fig_td.update_layout(height=500,
            title=dict(text="Tail Dependence vs Linear Correlation",
                       font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=13)),
            **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis']})
        for ann in fig_td.layout.annotations:
            ann.font.color = COLORS['accent_gold']
        st.plotly_chart(fig_td, use_container_width=True)

        info_box(f"""
        <b>Tail Dependence vs Correlation:</b><br>
        Correlation measures linear co-movement across the entire distribution.
        <b>Tail dependence (λ)</b> measures the probability of joint extreme losses
        above the {threshold_pct}th percentile threshold.<br><br>
        <b>Key insight:</b> Pairs with high standard correlation may have LOW tail dependence
        (good diversification in extremes) or HIGH tail dependence (correlation increases in crises —
        the so-called 'correlation breakdown' phenomenon).<br><br>
        <b>Risk implication:</b> If tail dependence is high between your holdings, diversification
        benefits disappear precisely when you need them most — during market crashes.
        """, title="📌 Tail Dependence Interpretation")

    # ══ PTAB 3: INDIVIDUAL EVT COMPARISON ════════════════════════════════════
    with ptab3:
        section_title("⚖️ Individual Stock EVT — Side-by-Side Comparison")

        individual_results = []
        for name in portfolio_names:
            col_ret = ret_df[name].values
            col_loss = -col_ret
            u_i, exc_i = pot_exceedances(col_loss, threshold_pct)
            if len(exc_i) >= 10:
                gpd_i = fit_gpd(exc_i)
                if gpd_i['success']:
                    var_i = gpd_var(conf, len(col_loss), len(exc_i), u_i, gpd_i)
                    es_i  = gpd_es(conf, len(col_loss), len(exc_i), u_i, gpd_i)
                    hv_i, he_i = historical_var_es(col_ret, conf)
                    nv_i, ne_i = parametric_var_es(col_ret, conf)
                    individual_results.append({
                        "Stock": name,
                        "Weight": f"{weights[portfolio_names.index(name)]*100:.1f}%",
                        f"EVT VaR {confidence}%": f"{var_i:.3f}%",
                        f"EVT ES {confidence}%": f"{es_i:.3f}%",
                        f"Hist VaR {confidence}%": f"{hv_i:.3f}%",
                        f"Normal VaR {confidence}%": f"{nv_i:.3f}%",
                        "GPD ξ": f"{gpd_i['xi']:.4f}",
                        "Threshold": f"{u_i:.3f}%",
                        "EVT > Normal?": "✅ Yes" if var_i > nv_i else "❌ No",
                    })

        if individual_results:
            df_ind = pd.DataFrame(individual_results)
            st.dataframe(df_ind, use_container_width=True, hide_index=True)

            # Bar chart comparison
            names_ind = [r["Stock"] for r in individual_results]
            evt_vars_ind  = [float(r[f"EVT VaR {confidence}%"].replace('%','')) for r in individual_results]
            hist_vars_ind = [float(r[f"Hist VaR {confidence}%"].replace('%','')) for r in individual_results]
            norm_vars_ind = [float(r[f"Normal VaR {confidence}%"].replace('%','')) for r in individual_results]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name='EVT VaR', x=names_ind, y=evt_vars_ind,
                marker_color=COLORS['danger'],
                marker_line=dict(color=COLORS['accent_gold'], width=1)))
            fig_bar.add_trace(go.Bar(name='Historical VaR', x=names_ind, y=hist_vars_ind,
                marker_color=COLORS['warning'],
                marker_line=dict(color=COLORS['accent_gold'], width=1)))
            fig_bar.add_trace(go.Bar(name='Normal VaR', x=names_ind, y=norm_vars_ind,
                marker_color=COLORS['medium_blue'],
                marker_line=dict(color=COLORS['accent_gold'], width=1)))
            fig_bar.update_layout(
                barmode='group', height=400,
                title=dict(text=f"Individual Stock VaR Comparison ({confidence}% Confidence)",
                           font=dict(color=COLORS['accent_gold'], family='Playfair Display', size=13)),
                yaxis=dict(title="VaR (%)", gridcolor='rgba(255,255,255,0.08)', color=COLORS['text_secondary']),
                xaxis=dict(color=COLORS['text_secondary']),
                **{k: v for k, v in dark_layout().items() if k not in ['xaxis','yaxis','title']})
            st.plotly_chart(fig_bar, use_container_width=True)

    # ══ PTAB 4: RISK DECOMPOSITION ═══════════════════════════════════════════
    with ptab4:
        section_title("📊 Portfolio Risk Decomposition — Component EVT")

        info_box("""
        <b>Component VaR:</b> Each asset's marginal contribution to portfolio VaR.<br>
        <b>Diversification Benefit:</b> Reduction in VaR from combining assets vs holding each separately.<br>
        <b>Tail Contribution:</b> Share of portfolio tail risk attributed to each position.
        """, title="Risk Decomposition Concepts")

        decomp_data = []
        stand_alone_sum_var = 0
        stand_alone_sum_es  = 0

        for i, name in enumerate(portfolio_names):
            col_ret = ret_df[name].values
            col_loss = -col_ret
            u_i, exc_i = pot_exceedances(col_loss, threshold_pct)
            if len(exc_i) >= 10:
                gpd_i = fit_gpd(exc_i)
                if gpd_i['success']:
                    var_i = gpd_var(conf, len(col_loss), len(exc_i), u_i, gpd_i)
                    es_i  = gpd_es(conf, len(col_loss), len(exc_i), u_i, gpd_i)
                    weighted_var = var_i * weights[i]
                    weighted_es  = es_i  * weights[i]
                    stand_alone_sum_var += weighted_var
                    stand_alone_sum_es  += weighted_es
                    decomp_data.append({
                        "Stock": name,
                        "Weight": f"{weights[i]*100:.1f}%",
                        "Stand-alone EVT VaR": f"{var_i:.3f}%",
                        "Weighted VaR": f"{weighted_var:.3f}%",
                        "Stand-alone EVT ES": f"{es_i:.3f}%",
                        "Weighted ES": f"{weighted_es:.3f}%",
                    })

        if decomp_data:
            st.dataframe(pd.DataFrame(decomp_data), use_container_width=True, hide_index=True)

            if not np.isnan(port_evt_var):
                div_benefit_var = stand_alone_sum_var - port_evt_var
                div_benefit_es  = stand_alone_sum_es  - port_evt_es

                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    metric_card("Weighted Sum VaR", f"{stand_alone_sum_var:.3f}%",
                               "No diversification", color=COLORS['danger'])
                with col_d2:
                    metric_card("Portfolio EVT VaR", f"{port_evt_var:.3f}%",
                               "With diversification", color=COLORS['warning'])
                with col_d3:
                    div_pct = div_benefit_var / stand_alone_sum_var * 100
                    metric_card("Diversification Benefit", f"{div_benefit_var:.3f}%",
                               f"{div_pct:.1f}% reduction", color=COLORS['success'])

                info_box(f"""
                <b>Diversification Analysis:</b><br>
                Weighted sum of stand-alone EVT VaRs = <b>{stand_alone_sum_var:.3f}%</b><br>
                Portfolio EVT VaR (accounting for correlations) = <b>{port_evt_var:.3f}%</b><br>
                <b>Diversification benefit = {div_benefit_var:.3f}% ({div_pct:.1f}% reduction)</b><br><br>
                This gap narrows in extreme markets (high tail dependence). If λ (tail dependence)
                is high between your holdings, the diversification benefit may be overstated by EVT
                applied to the portfolio return series alone — use a copula-based approach for maximum rigor.
                """, title="Portfolio Diversification Effect")

# ============================================================================
# FOOTER
# ============================================================================
footer()
