# app.py
# Winery Intelligence Platform - Enhanced Version
# Improvements: grouped navigation, Claude AI chatbot, winery-themed UI, KPI trend arrows

import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import anthropic

st.set_page_config(page_title="Winery Intelligence Platform", page_icon="🍷", layout="wide")

# ── Winery-themed UI ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit nav */
[data-testid="stSidebarNav"] {display: none;}

/* Color palette: burgundy, gold, cream */
:root {
    --burgundy: #722F37;
    --burgundy-light: #9B3A45;
    --gold: #C9A84C;
    --gold-light: #E8C97A;
    --cream: #F5F0E8;
    --dark-bg: #1A0A0C;
    --card-bg: rgba(114, 47, 55, 0.08);
    --border: rgba(201, 168, 76, 0.25);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2D0E13 0%, #1A0A0C 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {color: #F5F0E8 !important;}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem !important;
    padding: 2px 0 !important;
}

/* Main background */
.stApp {background-color: #0F0608;}
.block-container {padding-top: 1.5rem;}

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    padding: 16px 20px;
    border-radius: 12px;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover {border-color: var(--gold);}
div[data-testid="metric-container"] label {
    font-size: 0.8rem !important;
    color: var(--gold-light) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #F5F0E8 !important;
    font-size: 1.5rem !important;
}

/* Page title */
h1 {
    color: var(--gold) !important;
    font-size: 1.8rem !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem !important;
}
h2, h3 {color: var(--gold-light) !important;}

/* Divider */
hr {border-color: var(--border) !important;}

/* Buttons */
.stButton > button {
    background: var(--burgundy) !important;
    color: #F5F0E8 !important;
    border: 1px solid var(--gold) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover {background: var(--burgundy-light) !important;}

/* Section nav headers in sidebar */
.nav-section {
    color: var(--gold) !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 12px !important;
    margin-bottom: 2px !important;
    opacity: 0.85;
}

/* Dataframes */
[data-testid="stDataFrame"] {border: 1px solid var(--border); border-radius: 8px;}

/* Info/success/warning boxes */
.stAlert {border-radius: 8px !important;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def money(x):
    try:
        x = float(x)
    except Exception:
        return "$0"
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"

def money_full(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "$0"

def pct(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "0.0%"

def trend_arrow(current, previous, invert=False):
    """Return delta and direction for st.metric delta."""
    if previous == 0:
        return None
    change = (current - previous) / abs(previous)
    label = f"{'+' if change >= 0 else ''}{change:.1%} vs prev period"
    return label

def clean_dates(df):
    for col in ["date", "join_date", "last_purchase"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "date" in df.columns:
        df = df.dropna(subset=["date"])
    return df

def load_or_sample(uploaded_file, sample_df):
    if uploaded_file is None:
        return sample_df.copy(), "Sample Data"
    return pd.read_csv(uploaded_file), "Uploaded Data"

def data_badge(source):
    if source == "Uploaded Data":
        st.success("📂 Data Source: Your Uploaded Winery Data")
    else:
        st.info("🍇 Data Source: Sample Demo Data — upload your CSVs in the sidebar to use real data")

def get_period_split(df, date_col="date"):
    """Split dataframe into current and previous half for trend comparison."""
    if date_col not in df.columns or len(df) == 0:
        return df, df
    mid = df[date_col].min() + (df[date_col].max() - df[date_col].min()) / 2
    return df[df[date_col] >= mid], df[df[date_col] < mid]

# ── Login ────────────────────────────────────────────────────────────────────
def login_screen():
    st.markdown("""
    <div style='text-align:center; padding: 3rem 0 1rem 0;'>
        <div style='font-size: 4rem;'>🍷</div>
        <h1 style='border:none !important; font-size: 2.2rem !important;'>Winery Intelligence Platform</h1>
        <p style='color: #C9A84C; font-size: 1.1rem;'>Analytics & insights for winery owners and managers</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login"):
            st.markdown("#### Sign In")
            username = st.text_input("Username", value="demo")
            password = st.text_input("Password", value="winery123", type="password")
            submitted = st.form_submit_button("🍷 Sign In", use_container_width=True)
            st.caption("Demo: username `demo` · password `winery123`")
        if submitted:
            if username == "demo" and password == "winery123":
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Incorrect login. Try demo / winery123.")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_screen()
    st.stop()

# ── Sample Data ──────────────────────────────────────────────────────────────
@st.cache_data
def generate_sample_data(n_days=365, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    wineries = ["Livermore Estate", "Napa Tasting Room", "Sonoma Reserve"]
    wines = ["Cabernet Sauvignon", "Chardonnay", "Merlot", "Rosé", "Sauvignon Blanc", "Pinot Noir"]
    channels = ["Tasting Room", "Wine Club", "Event", "Online", "Wholesale"]
    event_types = ["Live Music", "Food Truck", "Release Party", "Wedding", "Corporate Event"]
    price_map = {"Cabernet Sauvignon": 48, "Chardonnay": 34, "Merlot": 39, "Rosé": 29, "Sauvignon Blanc": 31, "Pinot Noir": 44}
    cost_map = {"Cabernet Sauvignon": 18, "Chardonnay": 12, "Merlot": 14, "Rosé": 10, "Sauvignon Blanc": 11, "Pinot Noir": 16}

    sales_rows = []
    for d in dates:
        weekend = d.weekday() >= 5
        season_boost = 1.25 if d.month in [5, 6, 7, 8, 9, 10] else 0.9
        for winery in wineries:
            winery_boost = {"Livermore Estate": 1.00, "Napa Tasting Room": 1.18, "Sonoma Reserve": 0.92}[winery]
            transactions = rng.integers(8, 24 if weekend else 15)
            for _ in range(transactions):
                wine = rng.choice(wines)
                channel = rng.choice(channels, p=[0.38, 0.24, 0.16, 0.12, 0.10])
                bottles = int(max(1, rng.normal(3.3 if channel != "Wholesale" else 16, 1.9)))
                discount = rng.choice([0, .05, .10, .15, .20], p=[.55, .18, .15, .08, .04])
                revenue = bottles * price_map[wine] * (1 - discount) * season_boost * winery_boost
                cogs = bottles * cost_map[wine] * rng.uniform(.92, 1.12)
                customer_id = f"C{rng.integers(1000, 1999)}"
                sales_rows.append([d, winery, customer_id, wine, channel, bottles, round(revenue, 2), round(cogs, 2)])

    sales = pd.DataFrame(sales_rows, columns=["date", "winery", "customer_id", "wine", "channel", "bottles_sold", "revenue", "cogs"])

    event_rows = []
    for d in dates:
        if d.weekday() in [4, 5, 6]:
            for winery in wineries:
                if rng.random() < 0.32:
                    event_type = rng.choice(event_types)
                    attendance = int(max(25, rng.normal(95 if event_type != "Wedding" else 150, 45)))
                    ticket_revenue = attendance * rng.uniform(10, 55)
                    wine_sales = attendance * rng.uniform(18, 65)
                    labor_cost = attendance * rng.uniform(5, 14)
                    vendor_cost = rng.uniform(250, 2800)
                    marketing_cost = rng.uniform(75, 950)
                    event_rows.append([d, winery, event_type, attendance, round(ticket_revenue, 2), round(wine_sales, 2), round(labor_cost, 2), round(vendor_cost, 2), round(marketing_cost, 2)])

    events = pd.DataFrame(event_rows, columns=["date", "winery", "event_type", "attendance", "ticket_revenue", "wine_sales", "labor_cost", "vendor_cost", "marketing_cost"])
    events["total_revenue"] = events["ticket_revenue"] + events["wine_sales"]
    events["total_cost"] = events["labor_cost"] + events["vendor_cost"] + events["marketing_cost"]
    events["profit"] = events["total_revenue"] - events["total_cost"]

    club_rows = []
    for winery in wineries:
        join_dates = pd.to_datetime(rng.choice(dates[:-30], size=250, replace=True))
        for i, join_date in enumerate(join_dates, 1):
            visits_90d = int(rng.poisson(2.1))
            purchases_90d = int(rng.poisson(3.4))
            avg_order_value = round(max(28, rng.normal(118, 48)), 2)
            months_member = max(1, int((dates[-1] - join_date).days / 30))
            emails_opened = int(rng.integers(0, 12))
            skipped_shipments = int(rng.choice([0, 1, 2, 3], p=[.58, .25, .12, .05]))
            days_since_last_purchase = int(rng.integers(1, 180))
            churned = int((skipped_shipments >= 2 and days_since_last_purchase > 75) or rng.random() < 0.08)
            club_rows.append([winery, f"{winery[:3].upper()}-M{i:04d}", join_date.date(), months_member, visits_90d, purchases_90d, avg_order_value, emails_opened, skipped_shipments, days_since_last_purchase, churned])

    club = pd.DataFrame(club_rows, columns=["winery", "member_id", "join_date", "months_member", "visits_90d", "purchases_90d", "avg_order_value", "emails_opened_90d", "skipped_shipments", "days_since_last_purchase", "churned"])

    inventory_rows = []
    for winery in wineries:
        for wine in wines:
            bottles = int(rng.integers(200, 3600))
            forecast = int(rng.integers(90, 620))
            inventory_rows.append([winery, wine, bottles, forecast, cost_map[wine], price_map[wine]])

    inventory = pd.DataFrame(inventory_rows, columns=["winery", "wine", "bottles_on_hand", "monthly_sales_forecast", "unit_cost", "retail_price"])
    inventory["months_of_inventory"] = inventory["bottles_on_hand"] / inventory["monthly_sales_forecast"]
    inventory["inventory_value_cost"] = inventory["bottles_on_hand"] * inventory["unit_cost"]
    inventory["potential_retail_value"] = inventory["bottles_on_hand"] * inventory["retail_price"]

    customers = sales.groupby(["winery", "customer_id"], as_index=False).agg(
        total_revenue=("revenue", "sum"), orders=("date", "count"),
        last_purchase=("date", "max"), bottles=("bottles_sold", "sum"))
    customers["days_since_last_purchase"] = (sales["date"].max() - customers["last_purchase"]).dt.days
    customers["avg_order_value"] = customers["total_revenue"] / customers["orders"]
    customers["segment"] = np.select(
        [(customers["total_revenue"] > customers["total_revenue"].quantile(.85)) & (customers["days_since_last_purchase"] < 60),
         (customers["orders"] >= 6) & (customers["days_since_last_purchase"] < 90),
         (customers["days_since_last_purchase"] > 120),
         (customers["orders"] <= 2)],
        ["VIP", "Loyal", "At Risk", "New / Low Frequency"], default="Regular")

    return sales, events, club, inventory, customers

def calculate_financials(sales, events, club, marketing_spend, overhead, payroll):
    revenue = sales["revenue"].sum()
    cogs = sales["cogs"].sum()
    gross_profit = revenue - cogs
    gross_margin = gross_profit / revenue if revenue else 0
    event_profit = events["profit"].sum() if len(events) and "profit" in events.columns else 0
    operating_expenses = marketing_spend + overhead + payroll
    ebitda = gross_profit + event_profit - operating_expenses
    operating_margin = ebitda / revenue if revenue else 0
    customers_count = sales["customer_id"].nunique() if "customer_id" in sales.columns else max(len(club), 1)
    cac = marketing_spend / max(customers_count, 1)
    avg_order_value = revenue / max(len(sales), 1)
    churn_rate = club["churned"].mean() if len(club) and "churned" in club.columns else 0
    estimated_ltv = avg_order_value * 12 * max(0.15, 1 - churn_rate)
    ltv_cac = estimated_ltv / cac if cac else 0
    return {"revenue": revenue, "cogs": cogs, "gross_profit": gross_profit, "gross_margin": gross_margin,
            "event_profit": event_profit, "marketing_spend": marketing_spend, "overhead": overhead,
            "payroll": payroll, "operating_expenses": operating_expenses, "ebitda": ebitda,
            "operating_margin": operating_margin, "customers": customers_count, "cac": cac,
            "avg_order_value": avg_order_value, "estimated_ltv": estimated_ltv, "ltv_cac": ltv_cac, "churn_rate": churn_rate}

def add_churn_scores(club):
    required = ["months_member", "visits_90d", "purchases_90d", "avg_order_value",
                "emails_opened_90d", "skipped_shipments", "days_since_last_purchase", "churned"]
    if not all(col in club.columns for col in required):
        return club.copy(), None, None
    features = required[:-1]
    X = club[features]
    y = club["churned"]
    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X, y)
    scored = club.copy()
    scored["churn_risk_score"] = model.predict_proba(X)[:, 1]
    scored["risk_level"] = pd.cut(scored["churn_risk_score"], bins=[0, .33, .66, 1],
                                   labels=["Low", "Medium", "High"], include_lowest=True)
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    return scored, model, importance

def build_insights(sales, events, club_scored, inventory, fin):
    insights = []
    if len(sales):
        wine_revenue = sales.groupby("wine")["revenue"].sum().sort_values(ascending=False)
        insights.append(f"{wine_revenue.index[0]} is the top revenue wine at {money_full(wine_revenue.iloc[0])}.")
        insights.append(f"{wine_revenue.index[-1]} is the lowest revenue wine. Consider pairing it with events or a targeted tasting-room promotion.")
        channel_revenue = sales.groupby("channel")["revenue"].sum().sort_values(ascending=False)
        insights.append(f"{channel_revenue.index[0]} is the strongest channel, representing {pct(channel_revenue.iloc[0] / channel_revenue.sum())} of revenue.")
    if len(events):
        event_profit = events.groupby("event_type")["profit"].mean().sort_values(ascending=False)
        insights.append(f"{event_profit.index[0]} has the highest average event profit.")
        insights.append(f"{event_profit.index[-1]} has the weakest average event profit. Review pricing, staffing, vendor costs, and marketing spend.")
    if len(club_scored) and "risk_level" in club_scored.columns:
        insights.append(f"{int((club_scored['risk_level'] == 'High').sum())} wine club members are high-risk. Prioritize personal outreach before the next shipment.")
    if len(inventory) and "months_of_inventory" in inventory.columns:
        low_stock = sorted(inventory[inventory["months_of_inventory"] < 1.5]["wine"].unique().tolist())
        overstock = sorted(inventory[inventory["months_of_inventory"] > 6]["wine"].unique().tolist())
        if low_stock:
            insights.append(f"Low-stock wines: {', '.join(low_stock)}.")
        if overstock:
            insights.append(f"Potential overstock wines: {', '.join(overstock)}.")
    if fin["ebitda"] < 0:
        insights.append("EBITDA is negative under the current cost assumptions. Review operating expenses and margin.")
    else:
        insights.append(f"EBITDA is positive at {money_full(fin['ebitda'])}, with an operating margin of {pct(fin['operating_margin'])}.")
    return insights

def generate_pdf_report(fin, insights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Winery Intelligence Executive Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Financial Snapshot", ln=True)
    pdf.set_font("Helvetica", "", 11)
    rows = [("Revenue", money_full(fin["revenue"])), ("COGS", money_full(fin["cogs"])),
            ("Gross Profit", money_full(fin["gross_profit"])), ("Gross Margin", pct(fin["gross_margin"])),
            ("Event Profit", money_full(fin["event_profit"])), ("Operating Expenses", money_full(fin["operating_expenses"])),
            ("EBITDA Estimate", money_full(fin["ebitda"])), ("Operating Margin", pct(fin["operating_margin"])),
            ("CAC Estimate", money_full(fin["cac"])), ("Estimated LTV", money_full(fin["estimated_ltv"])),
            ("LTV/CAC", f"{fin['ltv_cac']:.1f}x")]
    for label, value in rows:
        pdf.cell(75, 8, label)
        pdf.cell(0, 8, value, ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for item in insights[:10]:
        safe_item = str(item).replace("•", "-").replace("–", "-").replace("—", "-")
        safe_item = safe_item.encode("latin-1", "ignore").decode("latin-1")
        pdf.multi_cell(180, 7, f"- {safe_item[:180]}")
    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)

# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_COLORS = ["#722F37", "#C9A84C", "#9B3A45", "#E8C97A", "#5C1E24", "#F5F0E8"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F5F0E8"),
    title_font=dict(color="#C9A84C", size=14),
    xaxis=dict(gridcolor="rgba(201,168,76,0.1)", linecolor="rgba(201,168,76,0.2)"),
    yaxis=dict(gridcolor="rgba(201,168,76,0.1)", linecolor="rgba(201,168,76,0.2)"),
    colorway=PLOTLY_COLORS,
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

def styled_chart(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ── Load data ────────────────────────────────────────────────────────────────
sample_sales, sample_events, sample_club, sample_inventory, sample_customers = generate_sample_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍷 Winery Intelligence")
    st.divider()

    # ── Grouped Navigation ───────────────────────────────────────────────────
    st.markdown('<p class="nav-section">📊 Overview</p>', unsafe_allow_html=True)
    nav_overview = st.radio("", ["Welcome / How to Use", "Executive Dashboard"],
                             key="nav_overview", label_visibility="collapsed")

    st.markdown('<p class="nav-section">🏪 Operations</p>', unsafe_allow_html=True)
    nav_ops = st.radio("", ["Sales Analytics", "Inventory", "Event ROI"],
                        key="nav_ops", label_visibility="collapsed")

    st.markdown('<p class="nav-section">👥 Customers</p>', unsafe_allow_html=True)
    nav_customers = st.radio("", ["Wine Club Churn", "Customer Intelligence"],
                              key="nav_customers", label_visibility="collapsed")

    st.markdown('<p class="nav-section">💰 Finance</p>', unsafe_allow_html=True)
    nav_finance = st.radio("", ["Financial Dashboard", "Revenue Forecasting", "Multi-Winery Comparison"],
                            key="nav_finance", label_visibility="collapsed")

    st.markdown('<p class="nav-section">🤖 Tools</p>', unsafe_allow_html=True)
    nav_tools = st.radio("", ["AI Winery Chatbot", "PDF Reports", "CSV Templates"],
                          key="nav_tools", label_visibility="collapsed")

    st.divider()

    # Track which group was last changed
    # We use session state to remember active page
    for group, options in [
        ("nav_overview", ["Welcome / How to Use", "Executive Dashboard"]),
        ("nav_ops", ["Sales Analytics", "Inventory", "Event ROI"]),
        ("nav_customers", ["Wine Club Churn", "Customer Intelligence"]),
        ("nav_finance", ["Financial Dashboard", "Revenue Forecasting", "Multi-Winery Comparison"]),
        ("nav_tools", ["AI Winery Chatbot", "PDF Reports", "CSV Templates"]),
    ]:
        if st.session_state.get(group) in options:
            if st.session_state.get("_last_group") != group:
                st.session_state["_active_page"] = st.session_state[group]
                st.session_state["_last_group"] = group

    # Data uploads
    st.markdown('<p class="nav-section">📁 Your Data</p>', unsafe_allow_html=True)
    sales_upload = st.file_uploader("Sales CSV", type=["csv"])
    events_upload = st.file_uploader("Events CSV", type=["csv"])
    club_upload = st.file_uploader("Wine Club CSV", type=["csv"])
    inventory_upload = st.file_uploader("Inventory CSV", type=["csv"])
    customers_upload = st.file_uploader("Customer CSV", type=["csv"])

    st.divider()
    st.markdown('<p class="nav-section">💸 Cost Assumptions</p>', unsafe_allow_html=True)
    marketing_spend = st.number_input("Marketing spend", min_value=0, value=60000, step=5000)
    overhead = st.number_input("Overhead", min_value=0, value=300000, step=10000)
    payroll = st.number_input("Payroll / staffing", min_value=0, value=520000, step=10000)

    st.divider()
    if st.button("🚪 Log out", use_container_width=True):
        st.session_state["logged_in"] = False
        st.rerun()

# Determine active page (default to Welcome)
# Read from the radio groups — whichever one last changed wins
# Simple approach: check all groups, use whichever matches a recently clicked value
def get_active_page():
    groups = {
        "nav_overview": st.session_state.get("nav_overview"),
        "nav_ops": st.session_state.get("nav_ops"),
        "nav_customers": st.session_state.get("nav_customers"),
        "nav_finance": st.session_state.get("nav_finance"),
        "nav_tools": st.session_state.get("nav_tools"),
    }
    return st.session_state.get("_active_page", "Welcome / How to Use")

page = get_active_page()

# ── Load and filter data ──────────────────────────────────────────────────────
sales, sales_source = load_or_sample(sales_upload, sample_sales)
events, events_source = load_or_sample(events_upload, sample_events)
club, club_source = load_or_sample(club_upload, sample_club)
inventory, inventory_source = load_or_sample(inventory_upload, sample_inventory)
customers, customers_source = load_or_sample(customers_upload, sample_customers)

sales = clean_dates(sales)
events = clean_dates(events)
club = clean_dates(club)
customers = clean_dates(customers)

all_wineries = sorted(sales["winery"].unique().tolist()) if "winery" in sales.columns else ["Default Winery"]
if "winery" in sales.columns:
    sales = sales[sales["winery"].isin(all_wineries)]
if "winery" in events.columns:
    events = events[events["winery"].isin(all_wineries)]
if "winery" in club.columns:
    club = club[club["winery"].isin(all_wineries)]
if "winery" in inventory.columns:
    inventory = inventory[inventory["winery"].isin(all_wineries)]
if "winery" in customers.columns:
    customers = customers[customers["winery"].isin(all_wineries)]

club_scored, churn_model, churn_importance = add_churn_scores(club)
fin = calculate_financials(sales, events, club, marketing_spend, overhead, payroll)
insights = build_insights(sales, events, club_scored, inventory, fin)

# Period splits for trend arrows
sales_curr, sales_prev = get_period_split(sales)
fin_curr = calculate_financials(sales_curr, events, club, marketing_spend / 2, overhead / 2, payroll / 2)
fin_prev = calculate_financials(sales_prev, events, club, marketing_spend / 2, overhead / 2, payroll / 2)

# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "Welcome / How to Use":
    st.title("🍷 Winery Intelligence Platform")
    st.markdown("##### Finance, operations, customer & AI analytics for winery owners and managers")
    data_badge(sales_source)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What This Platform Does")
        st.write("Turn raw sales, wine club, event, customer, and inventory data into actionable business decisions — built for owners, general managers, wine club managers, and operations teams.")

        st.markdown("### How to Use It")
        st.markdown("""
1. **Upload your CSVs** in the sidebar (or explore with sample demo data)
2. **Executive Dashboard** — overall performance at a glance
3. **Sales Analytics** — which wines and channels drive revenue
4. **Wine Club Churn** — members at risk of canceling
5. **Event ROI** — which events are actually profitable
6. **Inventory** — prevent stockouts or overstock
7. **Financial Dashboard** — margin, EBITDA, CAC, and LTV
8. **AI Chatbot** — ask business questions in plain English
9. **PDF Reports** — export for owner or investor meetings
        """)

    with col2:
        st.markdown("### Platform Highlights")
        highlights = [
            ("📊", "Executive Dashboard", "Revenue, margin, EBITDA, churn at a glance"),
            ("🍷", "Sales Analytics", "Wine and channel performance with margin"),
            ("👥", "Wine Club Churn", "ML-powered churn risk scores"),
            ("🎪", "Event ROI", "Profit by event type and attendance"),
            ("📦", "Inventory", "Stock levels vs. forecast"),
            ("💰", "Financial Dashboard", "EBITDA, CAC, LTV, profit bridge"),
            ("🤖", "AI Chatbot", "Claude-powered winery analyst"),
            ("📄", "PDF Reports", "One-click executive reports"),
        ]
        for icon, title, desc in highlights:
            st.markdown(f"**{icon} {title}** — {desc}")

    st.divider()
    st.info("Demo login: username `demo` · password `winery123` · The app works with sample data until you upload real CSVs.")

elif page == "Executive Dashboard":
    st.title("📊 Executive Dashboard")
    data_badge(sales_source)

    # KPI metrics with trend arrows
    c1, c2, c3, c4, c5 = st.columns(5)
    rev_delta = trend_arrow(fin_curr["revenue"], fin_prev["revenue"])
    gp_delta = trend_arrow(fin_curr["gross_profit"], fin_prev["gross_profit"])
    gm_delta = trend_arrow(fin_curr["gross_margin"], fin_prev["gross_margin"])
    ebitda_delta = trend_arrow(fin_curr["ebitda"], fin_prev["ebitda"])

    c1.metric("Revenue", money(fin["revenue"]), delta=rev_delta)
    c2.metric("Gross Profit", money(fin["gross_profit"]), delta=gp_delta)
    c3.metric("Gross Margin", pct(fin["gross_margin"]), delta=gm_delta)
    c4.metric("EBITDA", money(fin["ebitda"]), delta=ebitda_delta)
    c5.metric("Club Churn", pct(fin["churn_rate"]))

    st.caption(f"Full revenue: {money_full(fin['revenue'])} · Full EBITDA estimate: {money_full(fin['ebitda'])} · Trend arrows compare first vs second half of period")
    st.divider()

    left, right = st.columns(2)
    with left:
        monthly = sales.groupby(pd.Grouper(key="date", freq="ME"), as_index=False)["revenue"].sum()
        fig = px.line(monthly, x="date", y="revenue", title="Monthly Revenue")
        fig.update_traces(line_color="#C9A84C", line_width=2.5)
        st.plotly_chart(styled_chart(fig), use_container_width=True)
    with right:
        by_channel = sales.groupby("channel", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        fig = px.bar(by_channel, x="channel", y="revenue", title="Revenue by Channel")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    st.subheader("📋 Executive Recommendations")
    for item in insights:
        st.markdown(f"- {item}")

elif page == "Sales Analytics":
    st.title("🍷 Sales Analytics")
    data_badge(sales_source)

    by_wine = sales.groupby("wine", as_index=False).agg(
        revenue=("revenue", "sum"), cogs=("cogs", "sum"), bottles=("bottles_sold", "sum"))
    by_wine["gross_profit"] = by_wine["revenue"] - by_wine["cogs"]
    by_wine["gross_margin"] = by_wine["gross_profit"] / by_wine["revenue"]

    # Trend metrics
    rev_delta = trend_arrow(fin_curr["revenue"], fin_prev["revenue"])
    aov_delta = trend_arrow(fin_curr["avg_order_value"], fin_prev["avg_order_value"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", money(fin["revenue"]), delta=rev_delta)
    c2.metric("Bottles Sold", f"{by_wine['bottles'].sum():,.0f}")
    c3.metric("Avg Bottle Revenue", money(fin["revenue"] / max(by_wine["bottles"].sum(), 1)), delta=aov_delta)

    st.divider()
    left, right = st.columns(2)
    with left:
        fig = px.bar(by_wine.sort_values("revenue", ascending=False), x="wine", y="revenue", title="Revenue by Wine")
        st.plotly_chart(styled_chart(fig), use_container_width=True)
    with right:
        fig = px.bar(by_wine.sort_values("gross_margin", ascending=False), x="wine", y="gross_margin", title="Gross Margin by Wine")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    by_channel = sales.groupby("channel", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    fig = px.bar(by_channel, x="channel", y="revenue", title="Revenue by Sales Channel")
    st.plotly_chart(styled_chart(fig), use_container_width=True)

    st.dataframe(by_wine.sort_values("revenue", ascending=False), use_container_width=True)

elif page == "Wine Club Churn":
    st.title("👥 Wine Club Churn")
    data_badge(club_source)

    if churn_model is None:
        st.error("Wine club data is missing required columns. See CSV Templates for the format.")
    else:
        high_risk_count = int((club_scored["risk_level"] == "High").sum())
        churn_delta = trend_arrow(club_scored["churned"].mean(), 0.12, invert=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Members", f"{len(club_scored):,}")
        c2.metric("Churn Rate", pct(club_scored["churned"].mean()))
        c3.metric("High Risk Members", f"{high_risk_count:,}")
        c4.metric("Avg Order Value", money(club_scored["avg_order_value"].mean()))

        st.divider()
        left, right = st.columns(2)
        with left:
            risk_counts = club_scored["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "members"]
            color_map = {"Low": "#2E7D32", "Medium": "#C9A84C", "High": "#722F37"}
            fig = px.bar(risk_counts, x="risk_level", y="members", title="Risk Distribution",
                         color="risk_level", color_discrete_map=color_map)
            st.plotly_chart(styled_chart(fig), use_container_width=True)
        with right:
            fig = px.bar(churn_importance, x="feature", y="importance", title="Top Churn Drivers")
            st.plotly_chart(styled_chart(fig), use_container_width=True)

        st.subheader("🚨 High-Risk Follow-Up List")
        high_risk = club_scored.sort_values("churn_risk_score", ascending=False).head(50)
        st.dataframe(high_risk, use_container_width=True)
        st.download_button("⬇️ Download High-Risk Member CSV", high_risk.to_csv(index=False),
                           "high_risk_wine_club_members.csv", "text/csv")

elif page == "Customer Intelligence":
    st.title("👤 Customer Intelligence")
    data_badge(customers_source)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{customers['customer_id'].nunique():,}")
    c2.metric("Avg Order Value", money(customers["avg_order_value"].mean()))
    c3.metric("Avg Customer Revenue", money(customers["total_revenue"].mean()))
    c4.metric("At-Risk Customers", f"{(customers['segment'] == 'At Risk').sum():,}")

    st.divider()
    left, right = st.columns(2)
    with left:
        segment_counts = customers["segment"].value_counts().reset_index()
        segment_counts.columns = ["segment", "customers"]
        fig = px.bar(segment_counts, x="segment", y="customers", title="Customer Segments")
        st.plotly_chart(styled_chart(fig), use_container_width=True)
    with right:
        fig = px.scatter(customers, x="orders", y="total_revenue", color="segment",
                         title="Orders vs Revenue by Segment",
                         color_discrete_sequence=PLOTLY_COLORS)
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    st.dataframe(customers.sort_values("total_revenue", ascending=False), use_container_width=True)

elif page == "Event ROI":
    st.title("🎪 Event ROI")
    data_badge(events_source)

    if len(events) == 0:
        st.warning("No event data found.")
    else:
        summary = events.groupby("event_type", as_index=False).agg(
            events=("event_type", "count"), attendance=("attendance", "sum"),
            revenue=("total_revenue", "sum"), cost=("total_cost", "sum"), profit=("profit", "sum"))
        summary["roi"] = summary["profit"] / summary["cost"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Event Revenue", money(summary["revenue"].sum()))
        c2.metric("Event Cost", money(summary["cost"].sum()))
        c3.metric("Event Profit", money(summary["profit"].sum()))
        c4.metric("Avg ROI", pct(summary["profit"].sum() / max(summary["cost"].sum(), 1)))

        st.divider()
        left, right = st.columns(2)
        with left:
            fig = px.bar(summary.sort_values("profit", ascending=False), x="event_type", y="profit",
                         title="Profit by Event Type")
            st.plotly_chart(styled_chart(fig), use_container_width=True)
        with right:
            fig = px.scatter(events, x="attendance", y="profit", color="event_type",
                             title="Attendance vs Profit",
                             color_discrete_sequence=PLOTLY_COLORS)
            st.plotly_chart(styled_chart(fig), use_container_width=True)

        st.dataframe(summary.sort_values("profit", ascending=False), use_container_width=True)

elif page == "Inventory":
    st.title("📦 Inventory Management")
    data_badge(inventory_source)

    inventory["status"] = np.select(
        [inventory["months_of_inventory"] < 1.5, inventory["months_of_inventory"] > 6],
        ["Low Stock", "Overstock"], default="Healthy")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inventory Cost Value", money(inventory["inventory_value_cost"].sum()))
    c2.metric("Potential Retail Value", money(inventory["potential_retail_value"].sum()))
    c3.metric("Low Stock Items", f"{(inventory['status'] == 'Low Stock').sum():,}")
    c4.metric("Overstock Items", f"{(inventory['status'] == 'Overstock').sum():,}")

    st.divider()
    color_map = {"Healthy": "#2E7D32", "Low Stock": "#722F37", "Overstock": "#C9A84C"}
    fig = px.bar(inventory, x="wine", y="months_of_inventory", color="status",
                 title="Months of Inventory by Wine", color_discrete_map=color_map)
    st.plotly_chart(styled_chart(fig), use_container_width=True)
    st.dataframe(inventory.sort_values("months_of_inventory"), use_container_width=True)

elif page == "Financial Dashboard":
    st.title("💰 Financial Dashboard")
    data_badge(sales_source)

    rev_delta = trend_arrow(fin_curr["revenue"], fin_prev["revenue"])
    gm_delta = trend_arrow(fin_curr["gross_margin"], fin_prev["gross_margin"])
    ebitda_delta = trend_arrow(fin_curr["ebitda"], fin_prev["ebitda"])
    om_delta = trend_arrow(fin_curr["operating_margin"], fin_prev["operating_margin"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", money(fin["revenue"]), delta=rev_delta)
    c2.metric("Gross Margin", pct(fin["gross_margin"]), delta=gm_delta)
    c3.metric("EBITDA", money(fin["ebitda"]), delta=ebitda_delta)
    c4.metric("Operating Margin", pct(fin["operating_margin"]), delta=om_delta)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("CAC Estimate", money(fin["cac"]))
    c6.metric("Estimated LTV", money(fin["estimated_ltv"]))
    c7.metric("LTV / CAC", f"{fin['ltv_cac']:.1f}x")
    c8.metric("Event Profit", money(fin["event_profit"]))

    st.divider()
    bridge = pd.DataFrame({"Metric": ["Revenue", "COGS", "Gross Profit", "Event Profit",
                                       "Marketing Spend", "Overhead", "Payroll / Staffing", "EBITDA"],
                            "Amount": [fin["revenue"], -fin["cogs"], fin["gross_profit"], fin["event_profit"],
                                       -fin["marketing_spend"], -fin["overhead"], -fin["payroll"], fin["ebitda"]]})
    bridge["Color"] = bridge["Amount"].apply(lambda x: "#C9A84C" if x >= 0 else "#722F37")
    fig = go.Figure(go.Bar(x=bridge["Metric"], y=bridge["Amount"],
                           marker_color=bridge["Color"]))
    fig.update_layout(title="Simplified Profit Bridge", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(bridge[["Metric", "Amount"]], use_container_width=True)
    st.info("EBITDA here is an estimate: Gross Profit + Event Profit - Marketing Spend - Overhead - Payroll. In a real deployment, this would connect to actual accounting data.")

elif page == "Revenue Forecasting":
    st.title("📈 Revenue Forecasting")
    data_badge(sales_source)

    daily = sales.groupby("date", as_index=False)["revenue"].sum()
    daily["day_num"] = np.arange(len(daily))
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month

    model = LinearRegression()
    model.fit(daily[["day_num", "day_of_week", "month"]], daily["revenue"])

    days_forward = st.slider("Forecast days ahead", 7, 180, 60)
    future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=days_forward)
    future = pd.DataFrame({"date": future_dates,
                           "day_num": np.arange(len(daily), len(daily) + days_forward),
                           "day_of_week": future_dates.dayofweek,
                           "month": future_dates.month})
    future["forecast_revenue"] = model.predict(future[["day_num", "day_of_week", "month"]]).clip(0)

    combined = pd.concat([
        daily[["date", "revenue"]].rename(columns={"revenue": "amount"}).assign(type="Actual"),
        future[["date", "forecast_revenue"]].rename(columns={"forecast_revenue": "amount"}).assign(type="Forecast")
    ])

    fig = px.line(combined, x="date", y="amount", color="type", title="Actual vs Forecast Revenue",
                  color_discrete_map={"Actual": "#C9A84C", "Forecast": "#9B3A45"})
    st.plotly_chart(styled_chart(fig), use_container_width=True)

    st.metric("Forecasted Revenue Total", money(future["forecast_revenue"].sum()))
    st.dataframe(future[["date", "forecast_revenue"]], use_container_width=True)

elif page == "Multi-Winery Comparison":
    st.title("🏘️ Multi-Winery Comparison")
    st.info("This view compares all winery locations using full sample data.")

    comp_sales = sample_sales.groupby("winery", as_index=False).agg(
        revenue=("revenue", "sum"), cogs=("cogs", "sum"),
        bottles=("bottles_sold", "sum"), customers=("customer_id", "nunique"))
    comp_sales["gross_profit"] = comp_sales["revenue"] - comp_sales["cogs"]
    comp_sales["gross_margin"] = comp_sales["gross_profit"] / comp_sales["revenue"]
    comp_events = sample_events.groupby("winery", as_index=False)["profit"].sum().rename(columns={"profit": "event_profit"})
    comp_club = sample_club.groupby("winery", as_index=False)["churned"].mean().rename(columns={"churned": "churn_rate"})
    comp = comp_sales.merge(comp_events, on="winery", how="left").merge(comp_club, on="winery", how="left")

    left, right = st.columns(2)
    with left:
        fig = px.bar(comp, x="winery", y="revenue", title="Revenue by Winery")
        st.plotly_chart(styled_chart(fig), use_container_width=True)
    with right:
        fig = px.bar(comp, x="winery", y="gross_margin", title="Gross Margin by Winery")
        st.plotly_chart(styled_chart(fig), use_container_width=True)

    st.dataframe(comp.sort_values("revenue", ascending=False), use_container_width=True)

elif page == "AI Winery Chatbot":
    st.title("🤖 AI Winery Chatbot")
    st.caption("Powered by Claude · Ask anything about your winery's sales, events, wine club, inventory, and financial performance.")

    data_badge(sales_source)

    # Chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    data_summary = {
        "revenue": money_full(fin["revenue"]),
        "gross_profit": money_full(fin["gross_profit"]),
        "gross_margin": pct(fin["gross_margin"]),
        "ebitda": money_full(fin["ebitda"]),
        "club_churn": pct(fin["churn_rate"]),
        "top_wine": sales.groupby("wine")["revenue"].sum().idxmax() if len(sales) else "N/A",
        "top_channel": sales.groupby("channel")["revenue"].sum().idxmax() if len(sales) else "N/A",
        "high_risk_members": int((club_scored["risk_level"] == "High").sum()) if "risk_level" in club_scored.columns else 0,
        "recommendations": insights,
    }

    system_prompt = f"""You are a senior winery finance and operations analyst embedded in the Winery Intelligence Platform. 
You have access to the following current business data summary:
{data_summary}

Your job is to give practical, specific, and concise answers to questions from winery owners and general managers. 
Be direct and actionable. Use bullet points when listing recommendations. Keep answers under 200 words unless a detailed breakdown is specifically requested."""

    # Display chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"], avatar="🍷" if msg["role"] == "assistant" else None):
            st.write(msg["content"])

    # Example prompts
    if not st.session_state["chat_history"]:
        st.markdown("**Try asking:**")
        examples = [
            "Which wine should we promote this quarter?",
            "Why might our churn rate be high?",
            "Which event type should we focus on?",
            "What should management focus on this month?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}"):
                st.session_state["chat_history"].append({"role": "user", "content": ex})
                st.rerun()

    # Chat input
    question = st.chat_input("Ask your winery analyst anything...")

    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant", avatar="🍷"):
            with st.spinner("Analyzing your winery data..."):
                try:
                    # Try Anthropic API key from secrets or env
                    api_key = None
                    try:
                        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
                    except Exception:
                        pass
                    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

                    if api_key:
                        client = anthropic.Anthropic(api_key=api_key)
                        messages_payload = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state["chat_history"]
                        ]
                        response = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=500,
                            system=system_prompt,
                            messages=messages_payload,
                        )
                        answer = response.content[0].text
                    else:
                        # Fallback built-in insights
                        answer = "**No API key found.** Add `ANTHROPIC_API_KEY` to your Streamlit secrets to enable the AI chatbot.\n\n**Built-in insights based on your data:**\n"
                        answer += "\n".join(f"- {item}" for item in insights[:7])

                    st.write(answer)
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})

                except Exception as e:
                    err = f"Chatbot error: {e}"
                    st.error(err)
                    st.session_state["chat_history"].append({"role": "assistant", "content": err})

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear chat history"):
            st.session_state["chat_history"] = []
            st.rerun()

elif page == "PDF Reports":
    st.title("📄 PDF Executive Reports")
    st.write("Generate a one-click executive report for owners, managers, or investors.")

    st.subheader("Current Recommendations")
    for item in insights:
        st.markdown(f"- {item}")

    st.divider()
    pdf_bytes = generate_pdf_report(fin, insights)
    st.download_button("⬇️ Download Executive PDF Report", data=pdf_bytes,
                       file_name="winery_executive_report.pdf", mime="application/pdf")

elif page == "CSV Templates":
    st.title("📋 CSV Templates")
    st.write("Download these templates to upload your real winery data.")

    templates = {
        "sales_template.csv": sample_sales.head(100),
        "events_template.csv": sample_events.head(100),
        "wine_club_template.csv": sample_club.head(100),
        "inventory_template.csv": sample_inventory.head(100),
        "customers_template.csv": sample_customers.head(100),
    }

    for filename, df in templates.items():
        with st.expander(f"📄 {filename}"):
            st.dataframe(df.head(5), use_container_width=True)
            st.download_button(f"⬇️ Download {filename}", data=df.to_csv(index=False),
                               file_name=filename, mime="text/csv")
