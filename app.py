# app.py
# Winery Intelligence Platform - polished version
# Requirements: streamlit pandas numpy plotly scikit-learn fpdf2 openai

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from fpdf import FPDF

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="Winery Intelligence Platform", page_icon="🍷", layout="wide")

# Hide Streamlit's default top-left navigation and clean up layout
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
.block-container {padding-top: 2rem;}
div[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 16px;
    border-radius: 16px;
}
div[data-testid="metric-container"] label {font-size: 0.95rem !important;}
</style>
""", unsafe_allow_html=True)


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
        st.success("Data Source: Uploaded Winery Data")
    else:
        st.info("Data Source: Sample Demo Data")


def login_screen():
    st.title("🍷 Winery Intelligence Platform")
    st.caption("Demo login for winery owners, managers, and analysts.")
    with st.form("login"):
        username = st.text_input("Username", value="demo")
        password = st.text_input("Password", value="winery123", type="password")
        submitted = st.form_submit_button("Login")
    st.info("Demo login: username `demo`, password `winery123`")
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

    customers = sales.groupby(["winery", "customer_id"], as_index=False).agg(total_revenue=("revenue", "sum"), orders=("date", "count"), last_purchase=("date", "max"), bottles=("bottles_sold", "sum"))
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
    return {"revenue": revenue, "cogs": cogs, "gross_profit": gross_profit, "gross_margin": gross_margin, "event_profit": event_profit, "marketing_spend": marketing_spend, "overhead": overhead, "payroll": payroll, "operating_expenses": operating_expenses, "ebitda": ebitda, "operating_margin": operating_margin, "customers": customers_count, "cac": cac, "avg_order_value": avg_order_value, "estimated_ltv": estimated_ltv, "ltv_cac": ltv_cac, "churn_rate": churn_rate}


def add_churn_scores(club):
    required = ["months_member", "visits_90d", "purchases_90d", "avg_order_value", "emails_opened_90d", "skipped_shipments", "days_since_last_purchase", "churned"]
    if not all(col in club.columns for col in required):
        return club.copy(), None, None
    features = required[:-1]
    X = club[features]
    y = club["churned"]
    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X, y)
    scored = club.copy()
    scored["churn_risk_score"] = model.predict_proba(X)[:, 1]
    scored["risk_level"] = pd.cut(scored["churn_risk_score"], bins=[0, .33, .66, 1], labels=["Low", "Medium", "High"], include_lowest=True)
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
        insights.append("EBITDA is negative under the current cost assumptions, so management should review operating expenses and margin.")
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
    rows = [("Revenue", money_full(fin["revenue"])), ("COGS", money_full(fin["cogs"])), ("Gross Profit", money_full(fin["gross_profit"])), ("Gross Margin", pct(fin["gross_margin"])), ("Event Profit", money_full(fin["event_profit"])), ("Operating Expenses", money_full(fin["operating_expenses"])), ("EBITDA Estimate", money_full(fin["ebitda"])), ("Operating Margin", pct(fin["operating_margin"])), ("CAC Estimate", money_full(fin["cac"])), ("Estimated LTV", money_full(fin["estimated_ltv"])), ("LTV/CAC", f"{fin['ltv_cac']:.1f}x")]
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


sample_sales, sample_events, sample_club, sample_inventory, sample_customers = generate_sample_data()

st.sidebar.title("🍷 Winery Intelligence")
if st.sidebar.button("Log out"):
    st.session_state["logged_in"] = False
    st.rerun()

sales_upload = st.sidebar.file_uploader("Sales CSV", type=["csv"])
events_upload = st.sidebar.file_uploader("Events CSV", type=["csv"])
club_upload = st.sidebar.file_uploader("Wine Club CSV", type=["csv"])
inventory_upload = st.sidebar.file_uploader("Inventory CSV", type=["csv"])
customers_upload = st.sidebar.file_uploader("Customer CSV", type=["csv"])

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

if len(all_wineries) > 1:
    selected_wineries = st.sidebar.multiselect(
        "Location Filter",
        all_wineries,
        default=all_wineries
    )
else:
    selected_wineries = all_wineries

if "winery" in sales.columns:
    sales = sales[sales["winery"].isin(selected_wineries)]
if "winery" in events.columns:
    events = events[events["winery"].isin(selected_wineries)]
if "winery" in club.columns:
    club = club[club["winery"].isin(selected_wineries)]
if "winery" in inventory.columns:
    inventory = inventory[inventory["winery"].isin(selected_wineries)]
if "winery" in customers.columns:
    customers = customers[customers["winery"].isin(selected_wineries)]

st.sidebar.divider()
st.sidebar.caption("Cost assumptions for financial dashboard")
marketing_spend = st.sidebar.number_input("Marketing spend", min_value=0, value=60000, step=5000)
overhead = st.sidebar.number_input("Overhead", min_value=0, value=300000, step=10000)
payroll = st.sidebar.number_input("Payroll / staffing", min_value=0, value=520000, step=10000)

page = st.sidebar.radio("Choose dashboard", ["Welcome / How to Use", "Executive Dashboard", "Sales Analytics", "Wine Club Churn", "Customer Intelligence", "Event ROI", "Inventory", "Financial Dashboard", "Revenue Forecasting", "Multi-Winery Comparison", "AI Winery Chatbot", "PDF Reports", "CSV Templates"])

club_scored, churn_model, churn_importance = add_churn_scores(club)
fin = calculate_financials(sales, events, club, marketing_spend, overhead, payroll)
insights = build_insights(sales, events, club_scored, inventory, fin)

st.title("Winery Intelligence Platform")
st.caption("Finance, operations, customer, and AI analytics for winery owners and managers.")

if page == "Welcome / How to Use":
    st.subheader("What This Platform Does")
    st.write("This app helps a winery turn sales, wine club, event, customer, and inventory data into business decisions. It is designed for owners, general managers, wine club managers, and operations teams.")
    data_badge(sales_source)
    st.markdown("""
### How a winery would use it
1. Upload CSV files for sales, events, wine club members, inventory, and customers.
2. Use the Executive Dashboard to see overall performance.
3. Use Sales Analytics to see which wines and channels drive revenue.
4. Use Wine Club Churn to identify members at risk of canceling.
5. Use Event ROI to see which events are actually profitable.
6. Use Inventory to prevent stockouts or overstock.
7. Use the Financial Dashboard to review margin, EBITDA, CAC, and LTV.
8. Use the AI Chatbot to ask business questions in plain English.
9. Export a PDF report for meetings.
""")
    st.info("Demo login: username demo / password winery123. The app works with sample data until a winery uploads real CSVs.")
    st.subheader("Best way to explain this to a winery")
    st.write("“I built a winery analytics platform that combines sales, event, wine club, customer, and inventory data into one dashboard. The goal is to help management understand what is making money, which customers are at risk, which events are profitable, and where there are inventory or margin issues.”")

elif page == "Executive Dashboard":
    data_badge(sales_source)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Revenue", money(fin["revenue"]))
    c2.metric("Gross Profit", money(fin["gross_profit"]))
    c3.metric("Gross Margin", pct(fin["gross_margin"]))
    c4.metric("EBITDA", money(fin["ebitda"]))
    c5.metric("Club Churn", pct(fin["churn_rate"]))
    st.caption(f"Full revenue: {money_full(fin['revenue'])} | Full EBITDA estimate: {money_full(fin['ebitda'])}")
    left, right = st.columns(2)
    with left:
        monthly = sales.groupby(pd.Grouper(key="date", freq="ME"), as_index=False)["revenue"].sum()
        st.plotly_chart(px.line(monthly, x="date", y="revenue", title="Monthly Revenue"), use_container_width=True)
    with right:
        by_channel = sales.groupby("channel", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        st.plotly_chart(px.bar(by_channel, x="channel", y="revenue", title="Revenue by Channel"), use_container_width=True)
    st.subheader("Executive Recommendations")
    for item in insights:
        st.markdown(f"- {item}")

elif page == "Sales Analytics":
    data_badge(sales_source)
    st.subheader("Sales Analytics")
    by_wine = sales.groupby("wine", as_index=False).agg(revenue=("revenue", "sum"), cogs=("cogs", "sum"), bottles=("bottles_sold", "sum"))
    by_wine["gross_profit"] = by_wine["revenue"] - by_wine["cogs"]
    by_wine["gross_margin"] = by_wine["gross_profit"] / by_wine["revenue"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", money(fin["revenue"]))
    c2.metric("Bottles Sold", f"{by_wine['bottles'].sum():,.0f}")
    c3.metric("Avg Bottle Revenue", money(fin["revenue"] / max(by_wine["bottles"].sum(), 1)))
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(by_wine.sort_values("revenue", ascending=False), x="wine", y="revenue", title="Revenue by Wine"), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(by_wine.sort_values("gross_margin", ascending=False), x="wine", y="gross_margin", title="Gross Margin by Wine"), use_container_width=True)
    st.dataframe(by_wine.sort_values("revenue", ascending=False), use_container_width=True)

elif page == "Wine Club Churn":
    data_badge(club_source)
    st.subheader("Wine Club Churn Predictions")
    if churn_model is None:
        st.error("Wine club data is missing required columns. Go to CSV Templates to see the format.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Members", f"{len(club_scored):,}")
        c2.metric("Churn Rate", pct(club_scored["churned"].mean()))
        c3.metric("High Risk Members", f"{(club_scored['risk_level'] == 'High').sum():,}")
        c4.metric("Avg Order Value", money(club_scored["avg_order_value"].mean()))
        left, right = st.columns(2)
        with left:
            risk_counts = club_scored["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "members"]
            st.plotly_chart(px.bar(risk_counts, x="risk_level", y="members", title="Risk Distribution"), use_container_width=True)
        with right:
            st.plotly_chart(px.bar(churn_importance, x="feature", y="importance", title="Churn Drivers"), use_container_width=True)
        st.subheader("High-Risk Follow-Up List")
        high_risk = club_scored.sort_values("churn_risk_score", ascending=False).head(50)
        st.dataframe(high_risk, use_container_width=True)
        st.download_button("Download High-Risk Member CSV", high_risk.to_csv(index=False), "high_risk_wine_club_members.csv", "text/csv")

elif page == "Customer Intelligence":
    data_badge(customers_source)
    st.subheader("Customer Intelligence")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{customers['customer_id'].nunique():,}")
    c2.metric("Avg Order Value", money(customers["avg_order_value"].mean()))
    c3.metric("Avg Customer Revenue", money(customers["total_revenue"].mean()))
    c4.metric("At-Risk Customers", f"{(customers['segment'] == 'At Risk').sum():,}")
    left, right = st.columns(2)
    with left:
        segment_counts = customers["segment"].value_counts().reset_index()
        segment_counts.columns = ["segment", "customers"]
        st.plotly_chart(px.bar(segment_counts, x="segment", y="customers", title="Customer Segments"), use_container_width=True)
    with right:
        st.plotly_chart(px.scatter(customers, x="orders", y="total_revenue", color="segment", title="Orders vs Revenue"), use_container_width=True)
    st.dataframe(customers.sort_values("total_revenue", ascending=False), use_container_width=True)

elif page == "Event ROI":
    data_badge(events_source)
    st.subheader("Event ROI")
    if len(events) == 0:
        st.warning("No event data found.")
    else:
        summary = events.groupby("event_type", as_index=False).agg(events=("event_type", "count"), attendance=("attendance", "sum"), revenue=("total_revenue", "sum"), cost=("total_cost", "sum"), profit=("profit", "sum"))
        summary["roi"] = summary["profit"] / summary["cost"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Event Revenue", money(summary["revenue"].sum()))
        c2.metric("Event Cost", money(summary["cost"].sum()))
        c3.metric("Event Profit", money(summary["profit"].sum()))
        c4.metric("Event ROI", pct(summary["profit"].sum() / max(summary["cost"].sum(), 1)))
        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.bar(summary.sort_values("profit", ascending=False), x="event_type", y="profit", title="Profit by Event Type"), use_container_width=True)
        with right:
            st.plotly_chart(px.scatter(events, x="attendance", y="profit", color="event_type", title="Attendance vs Profit"), use_container_width=True)
        st.dataframe(summary.sort_values("profit", ascending=False), use_container_width=True)

elif page == "Inventory":
    data_badge(inventory_source)
    st.subheader("Inventory Management")
    inventory["status"] = np.select([inventory["months_of_inventory"] < 1.5, inventory["months_of_inventory"] > 6], ["Low Stock", "Overstock"], default="Healthy")
    c1, c2, c3 = st.columns(3)
    c1.metric("Inventory Cost Value", money(inventory["inventory_value_cost"].sum()))
    c2.metric("Potential Retail Value", money(inventory["potential_retail_value"].sum()))
    c3.metric("Low Stock Items", f"{(inventory['status'] == 'Low Stock').sum():,}")
    st.plotly_chart(px.bar(inventory, x="wine", y="months_of_inventory", color="status", title="Months of Inventory"), use_container_width=True)
    st.dataframe(inventory.sort_values("months_of_inventory"), use_container_width=True)

elif page == "Financial Dashboard":
    data_badge(sales_source)
    st.subheader("Financial Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", money(fin["revenue"]))
    c2.metric("Gross Margin", pct(fin["gross_margin"]))
    c3.metric("EBITDA", money(fin["ebitda"]))
    c4.metric("Operating Margin", pct(fin["operating_margin"]))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("CAC Estimate", money(fin["cac"]))
    c6.metric("Estimated LTV", money(fin["estimated_ltv"]))
    c7.metric("LTV / CAC", f"{fin['ltv_cac']:.1f}x")
    c8.metric("Event Profit", money(fin["event_profit"]))
    bridge = pd.DataFrame({"Metric": ["Revenue", "COGS", "Gross Profit", "Event Profit", "Marketing Spend", "Overhead", "Payroll / Staffing", "EBITDA"], "Amount": [fin["revenue"], -fin["cogs"], fin["gross_profit"], fin["event_profit"], -fin["marketing_spend"], -fin["overhead"], -fin["payroll"], fin["ebitda"]]})
    st.plotly_chart(px.bar(bridge, x="Metric", y="Amount", title="Simplified Profit Bridge"), use_container_width=True)
    st.dataframe(bridge, use_container_width=True)
    st.info("EBITDA here is an estimate: Gross Profit + Event Profit - Marketing Spend - Overhead - Payroll. In a real winery, this would be replaced with actual accounting data.")

elif page == "Revenue Forecasting":
    data_badge(sales_source)
    st.subheader("Revenue Forecasting")
    daily = sales.groupby("date", as_index=False)["revenue"].sum()
    daily["day_num"] = np.arange(len(daily))
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    model = LinearRegression()
    model.fit(daily[["day_num", "day_of_week", "month"]], daily["revenue"])
    days_forward = st.slider("Forecast days", 7, 180, 60)
    future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=days_forward)
    future = pd.DataFrame({"date": future_dates, "day_num": np.arange(len(daily), len(daily) + days_forward), "day_of_week": future_dates.dayofweek, "month": future_dates.month})
    future["forecast_revenue"] = model.predict(future[["day_num", "day_of_week", "month"]])
    future["forecast_revenue"] = future["forecast_revenue"].clip(lower=0)
    combined = pd.concat([daily[["date", "revenue"]].rename(columns={"revenue": "amount"}).assign(type="Actual"), future[["date", "forecast_revenue"]].rename(columns={"forecast_revenue": "amount"}).assign(type="Forecast")])
    st.plotly_chart(px.line(combined, x="date", y="amount", color="type", title="Actual vs Forecast Revenue"), use_container_width=True)
    st.metric("Forecasted Revenue Total", money(future["forecast_revenue"].sum()))
    st.dataframe(future, use_container_width=True)

elif page == "Multi-Winery Comparison":
    st.subheader("Multi-Winery Comparison")
    st.info("This view compares all sample locations, even if filters are selected.")
    comp_sales = sample_sales.groupby("winery", as_index=False).agg(revenue=("revenue", "sum"), cogs=("cogs", "sum"), bottles=("bottles_sold", "sum"), customers=("customer_id", "nunique"))
    comp_sales["gross_profit"] = comp_sales["revenue"] - comp_sales["cogs"]
    comp_sales["gross_margin"] = comp_sales["gross_profit"] / comp_sales["revenue"]
    comp_events = sample_events.groupby("winery", as_index=False)["profit"].sum().rename(columns={"profit": "event_profit"})
    comp_club = sample_club.groupby("winery", as_index=False)["churned"].mean().rename(columns={"churned": "churn_rate"})
    comp = comp_sales.merge(comp_events, on="winery", how="left").merge(comp_club, on="winery", how="left")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(comp, x="winery", y="revenue", title="Revenue by Winery"), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(comp, x="winery", y="gross_margin", title="Gross Margin by Winery"), use_container_width=True)
    st.dataframe(comp.sort_values("revenue", ascending=False), use_container_width=True)

elif page == "AI Winery Chatbot":
    data_badge(sales_source)
    st.subheader("AI Winery Chatbot")
    st.caption("Ask questions about sales, events, wine club, customers, inventory, and financial performance.")
    st.code("Example questions:\nWhy did revenue decline?\nWhich event type is most profitable?\nWhich wine club members should we contact?\nWhat should management focus on this month?")
    question = st.text_area("Ask a question", placeholder="What should we do to improve profitability?")
    data_summary = {"revenue": money_full(fin["revenue"]), "gross_profit": money_full(fin["gross_profit"]), "gross_margin": pct(fin["gross_margin"]), "ebitda": money_full(fin["ebitda"]), "club_churn": pct(fin["churn_rate"]), "top_wine": sales.groupby("wine")["revenue"].sum().idxmax() if len(sales) else "N/A", "top_channel": sales.groupby("channel")["revenue"].sum().idxmax() if len(sales) else "N/A", "recommendations": insights}
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("Type a question first.")
        elif OpenAI is not None and api_key:
            try:
                client = OpenAI(api_key=api_key)
                prompt = f"""You are a winery finance and operations analyst. Give a practical answer for a winery owner or general manager.\n\nBusiness summary:\n{data_summary}\n\nUser question:\n{question}"""
                response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are a concise winery business analyst."}, {"role": "user", "content": prompt}], temperature=0.3)
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI call failed: {e}")
        else:
            st.warning("No OpenAI API key found. Showing built-in analytics answer instead.")
            st.write("Based on the dashboard, management should focus on:")
            for item in insights[:7]:
                st.markdown(f"- {item}")

elif page == "PDF Reports":
    st.subheader("PDF Executive Reports")
    st.write("Generate a one-click executive report for managers, owners, or investors.")
    for item in insights:
        st.markdown(f"- {item}")
    pdf_bytes = generate_pdf_report(fin, insights)
    st.download_button("Download Executive PDF Report", data=pdf_bytes, file_name="winery_executive_report.pdf", mime="application/pdf")

elif page == "CSV Templates":
    st.subheader("CSV Templates")
    st.write("Download these templates if a winery wants to test the app with real data.")
    templates = {"sales_template.csv": sample_sales.head(100), "events_template.csv": sample_events.head(100), "wine_club_template.csv": sample_club.head(100), "inventory_template.csv": sample_inventory.head(100), "customers_template.csv": sample_customers.head(100)}
    for filename, df in templates.items():
        st.markdown(f"### {filename}")
        st.dataframe(df.head(), use_container_width=True)
        st.download_button(f"Download {filename}", data=df.to_csv(index=False), file_name=filename, mime="text/csv")
