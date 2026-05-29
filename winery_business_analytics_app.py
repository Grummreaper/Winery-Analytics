
# app.py
# Winery Business Analytics Platform
# Run locally:
#   pip install streamlit pandas numpy plotly scikit-learn
#   streamlit run app.py

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Winery Business Analytics Platform",
    page_icon="🍷",
    layout="wide"
)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data
def generate_sample_data(n_days: int = 365, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")

    wines = ["Cabernet Sauvignon", "Chardonnay", "Merlot", "Rosé", "Sauvignon Blanc", "Pinot Noir"]
    channels = ["Tasting Room", "Wine Club", "Event", "Online", "Wholesale"]
    events = ["None", "Live Music", "Food Truck", "Release Party", "Wedding", "Corporate Event"]

    sales_rows = []
    for d in dates:
        weekend = d.weekday() >= 5
        season_boost = 1.25 if d.month in [5, 6, 7, 8, 9, 10] else 0.9
        for _ in range(rng.integers(8, 25 if weekend else 14)):
            wine = rng.choice(wines)
            channel = rng.choice(channels, p=[0.38, 0.25, 0.17, 0.12, 0.08])
            bottles = int(max(1, rng.normal(3.2 if channel != "Wholesale" else 18, 1.8)))
            price = {
                "Cabernet Sauvignon": 48,
                "Chardonnay": 34,
                "Merlot": 39,
                "Rosé": 29,
                "Sauvignon Blanc": 31,
                "Pinot Noir": 44
            }[wine]
            discount = rng.choice([0, .05, .10, .15, .20], p=[.55, .18, .15, .08, .04])
            revenue = bottles * price * (1 - discount) * season_boost
            cogs = revenue * rng.uniform(.28, .42)
            sales_rows.append([d, wine, channel, bottles, round(revenue, 2), round(cogs, 2)])

    sales = pd.DataFrame(
        sales_rows,
        columns=["date", "wine", "channel", "bottles_sold", "revenue", "cogs"]
    )

    event_rows = []
    for d in dates:
        if d.weekday() in [4, 5, 6] and rng.random() < 0.42:
            event_type = rng.choice(events[1:])
            attendance = int(max(25, rng.normal(95 if event_type != "Wedding" else 145, 45)))
            ticket_revenue = attendance * rng.uniform(10, 55)
            wine_sales = attendance * rng.uniform(18, 65)
            labor_cost = attendance * rng.uniform(4, 11)
            vendor_cost = rng.uniform(250, 2600)
            marketing_cost = rng.uniform(75, 950)
            event_rows.append([
                d, event_type, attendance, round(ticket_revenue, 2), round(wine_sales, 2),
                round(labor_cost, 2), round(vendor_cost, 2), round(marketing_cost, 2)
            ])

    events_df = pd.DataFrame(
        event_rows,
        columns=["date", "event_type", "attendance", "ticket_revenue", "wine_sales",
                 "labor_cost", "vendor_cost", "marketing_cost"]
    )
    if len(events_df):
        events_df["total_revenue"] = events_df["ticket_revenue"] + events_df["wine_sales"]
        events_df["total_cost"] = events_df["labor_cost"] + events_df["vendor_cost"] + events_df["marketing_cost"]
        events_df["profit"] = events_df["total_revenue"] - events_df["total_cost"]

    members = []
    start_dates = pd.to_datetime(rng.choice(dates[:-30], size=450, replace=True))
    for i, sd in enumerate(start_dates, 1):
        visits_90d = int(rng.poisson(2.1))
        purchases_90d = int(rng.poisson(3.4))
        avg_order_value = round(max(28, rng.normal(118, 48)), 2)
        months_member = max(1, int((dates[-1] - sd).days / 30))
        emails_opened = int(rng.integers(0, 12))
        skipped_shipments = int(rng.choice([0, 1, 2, 3], p=[.58, .25, .12, .05]))
        last_purchase_days = int(rng.integers(1, 180))
        churned = int((skipped_shipments >= 2 and last_purchase_days > 75) or rng.random() < 0.08)
        members.append([
            f"M{i:04d}", sd.date(), months_member, visits_90d, purchases_90d,
            avg_order_value, emails_opened, skipped_shipments, last_purchase_days, churned
        ])

    club = pd.DataFrame(
        members,
        columns=["member_id", "join_date", "months_member", "visits_90d", "purchases_90d",
                 "avg_order_value", "emails_opened_90d", "skipped_shipments",
                 "days_since_last_purchase", "churned"]
    )

    inventory = pd.DataFrame({
        "wine": wines,
        "bottles_on_hand": rng.integers(200, 3600, size=len(wines)),
        "monthly_sales_forecast": rng.integers(90, 620, size=len(wines)),
        "unit_cost": [18, 12, 14, 10, 11, 16],
        "retail_price": [48, 34, 39, 29, 31, 44]
    })
    inventory["months_of_inventory"] = inventory["bottles_on_hand"] / inventory["monthly_sales_forecast"]
    inventory["inventory_value_cost"] = inventory["bottles_on_hand"] * inventory["unit_cost"]
    inventory["potential_retail_value"] = inventory["bottles_on_hand"] * inventory["retail_price"]

    return sales, events_df, club, inventory


def load_or_sample(upload, sample_df):
    if upload is None:
        return sample_df.copy()
    return pd.read_csv(upload)


def clean_dates(df, date_col="date"):
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
    return df


def format_money(x):
    return f"${x:,.0f}"


def download_template(df, name):
    csv = df.head(25).to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Download {name} CSV template",
        csv,
        file_name=f"{name.lower().replace(' ', '_')}_template.csv",
        mime="text/csv"
    )


def kpi_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🍷 Winery Analytics")
st.sidebar.caption("Upload winery data or use built-in sample data.")

sample_sales, sample_events, sample_club, sample_inventory = generate_sample_data()

sales_upload = st.sidebar.file_uploader("Sales CSV", type=["csv"])
events_upload = st.sidebar.file_uploader("Events CSV", type=["csv"])
club_upload = st.sidebar.file_uploader("Wine Club CSV", type=["csv"])
inventory_upload = st.sidebar.file_uploader("Inventory CSV", type=["csv"])

sales = clean_dates(load_or_sample(sales_upload, sample_sales))
events_df = clean_dates(load_or_sample(events_upload, sample_events))
club = load_or_sample(club_upload, sample_club)
inventory = load_or_sample(inventory_upload, sample_inventory)

page = st.sidebar.radio(
    "Choose dashboard",
    [
        "Executive Summary",
        "Sales Analytics",
        "Wine Club Retention",
        "Event ROI",
        "Inventory",
        "Forecasting",
        "AI Insights",
        "CSV Templates"
    ]
)

# ----------------------------
# Header
# ----------------------------
st.title("Winery Business Analytics Platform")
st.caption("A finance + operations dashboard for tasting room sales, wine club retention, event profitability, inventory, and AI-style business insights.")

# ----------------------------
# Executive Summary
# ----------------------------
if page == "Executive Summary":
    total_revenue = sales["revenue"].sum()
    total_cogs = sales["cogs"].sum()
    gross_profit = total_revenue - total_cogs
    gross_margin = gross_profit / total_revenue if total_revenue else 0
    total_bottles = sales["bottles_sold"].sum()
    event_profit = events_df["profit"].sum() if "profit" in events_df.columns else 0
    churn_rate = club["churned"].mean() if "churned" in club.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Revenue", format_money(total_revenue))
    with c2: kpi_card("Gross Profit", format_money(gross_profit))
    with c3: kpi_card("Gross Margin", f"{gross_margin:.1%}")
    with c4: kpi_card("Bottles Sold", f"{total_bottles:,.0f}")
    with c5: kpi_card("Club Churn", f"{churn_rate:.1%}")

    st.divider()

    left, right = st.columns(2)

    with left:
        monthly = sales.groupby(pd.Grouper(key="date", freq="ME"), as_index=False)["revenue"].sum()
        fig = px.line(monthly, x="date", y="revenue", title="Monthly Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        by_channel = sales.groupby("channel", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        fig = px.bar(by_channel, x="channel", y="revenue", title="Revenue by Channel")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Business Snapshot")
    best_wine = sales.groupby("wine")["revenue"].sum().idxmax()
    best_channel = sales.groupby("channel")["revenue"].sum().idxmax()
    worst_inventory = inventory.sort_values("months_of_inventory").head(1)

    st.write(
        f"""
        **Top wine by revenue:** {best_wine}  
        **Top sales channel:** {best_channel}  
        **Event profit impact:** {format_money(event_profit)}  
        **Lowest inventory coverage:** {worst_inventory.iloc[0]['wine']} has about {worst_inventory.iloc[0]['months_of_inventory']:.1f} months of inventory.
        """
    )

# ----------------------------
# Sales Analytics
# ----------------------------
elif page == "Sales Analytics":
    st.subheader("Sales Analytics")

    min_date, max_date = sales["date"].min(), sales["date"].max()
    date_range = st.date_input("Date range", [min_date, max_date])
    if len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        sales_filtered = sales[(sales["date"] >= start) & (sales["date"] <= end)]
    else:
        sales_filtered = sales

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Revenue", format_money(sales_filtered["revenue"].sum()))
    with c2: kpi_card("COGS", format_money(sales_filtered["cogs"].sum()))
    with c3: kpi_card("Gross Profit", format_money((sales_filtered["revenue"] - sales_filtered["cogs"]).sum()))
    with c4: kpi_card("Avg. Bottle Price", format_money(sales_filtered["revenue"].sum() / max(sales_filtered["bottles_sold"].sum(), 1)))

    by_wine = sales_filtered.groupby("wine", as_index=False).agg(
        revenue=("revenue", "sum"),
        bottles=("bottles_sold", "sum"),
        cogs=("cogs", "sum")
    )
    by_wine["gross_margin"] = (by_wine["revenue"] - by_wine["cogs"]) / by_wine["revenue"]

    left, right = st.columns(2)
    with left:
        fig = px.bar(by_wine.sort_values("revenue", ascending=False), x="wine", y="revenue", title="Revenue by Wine")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        by_channel = sales_filtered.groupby("channel", as_index=False)["revenue"].sum()
        fig = px.pie(by_channel, names="channel", values="revenue", title="Channel Mix")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(by_wine.sort_values("revenue", ascending=False), use_container_width=True)

# ----------------------------
# Wine Club Retention
# ----------------------------
elif page == "Wine Club Retention":
    st.subheader("Wine Club Retention")

    required = ["months_member", "visits_90d", "purchases_90d", "avg_order_value",
                "emails_opened_90d", "skipped_shipments", "days_since_last_purchase", "churned"]

    if all(col in club.columns for col in required):
        features = required[:-1]
        X = club[features]
        y = club["churned"]

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)
        club["churn_risk_score"] = model.predict_proba(X)[:, 1]
        club["risk_level"] = pd.cut(
            club["churn_risk_score"],
            bins=[0, .33, .66, 1],
            labels=["Low", "Medium", "High"]
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Members", f"{len(club):,}")
        with c2: kpi_card("Churn Rate", f"{club['churned'].mean():.1%}")
        with c3: kpi_card("High-Risk Members", f"{(club['risk_level'] == 'High').sum():,}")
        with c4: kpi_card("Avg Order Value", format_money(club["avg_order_value"].mean()))

        left, right = st.columns(2)
        with left:
            risk_counts = club["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "members"]
            fig = px.bar(risk_counts, x="risk_level", y="members", title="Wine Club Risk Levels")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            importance = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            fig = px.bar(importance, x="feature", y="importance", title="Churn Driver Importance")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Members to Follow Up With First")
        high_risk = club.sort_values("churn_risk_score", ascending=False).head(25)
        st.dataframe(high_risk, use_container_width=True)

        st.info(
            "Suggested action: offer high-risk members a personalized tasting invite, shipment flexibility, or a limited-release bottle preview."
        )
    else:
        st.error("Wine club CSV is missing required columns. Open CSV Templates to see the format.")

# ----------------------------
# Event ROI
# ----------------------------
elif page == "Event ROI":
    st.subheader("Event ROI")

    if len(events_df) == 0:
        st.warning("No event data found.")
    else:
        total_event_revenue = events_df["total_revenue"].sum()
        total_event_cost = events_df["total_cost"].sum()
        total_event_profit = events_df["profit"].sum()
        roi = total_event_profit / total_event_cost if total_event_cost else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Event Revenue", format_money(total_event_revenue))
        with c2: kpi_card("Event Cost", format_money(total_event_cost))
        with c3: kpi_card("Event Profit", format_money(total_event_profit))
        with c4: kpi_card("Event ROI", f"{roi:.1%}")

        event_summary = events_df.groupby("event_type", as_index=False).agg(
            events=("event_type", "count"),
            attendance=("attendance", "sum"),
            revenue=("total_revenue", "sum"),
            cost=("total_cost", "sum"),
            profit=("profit", "sum")
        )
        event_summary["roi"] = event_summary["profit"] / event_summary["cost"]

        left, right = st.columns(2)
        with left:
            fig = px.bar(event_summary.sort_values("profit", ascending=False), x="event_type", y="profit", title="Profit by Event Type")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.scatter(events_df, x="attendance", y="profit", color="event_type", title="Attendance vs Profit")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(event_summary.sort_values("profit", ascending=False), use_container_width=True)

# ----------------------------
# Inventory
# ----------------------------
elif page == "Inventory":
    st.subheader("Inventory Management")

    inventory["months_of_inventory"] = inventory["bottles_on_hand"] / inventory["monthly_sales_forecast"]
    inventory["status"] = np.select(
        [
            inventory["months_of_inventory"] < 1.5,
            inventory["months_of_inventory"] > 6
        ],
        [
            "Low Stock",
            "Overstock"
        ],
        default="Healthy"
    )

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Inventory Cost Value", format_money(inventory["inventory_value_cost"].sum()))
    with c2: kpi_card("Potential Retail Value", format_money(inventory["potential_retail_value"].sum()))
    with c3: kpi_card("Low Stock Wines", f"{(inventory['status'] == 'Low Stock').sum()}")

    fig = px.bar(inventory, x="wine", y="months_of_inventory", color="status", title="Months of Inventory by Wine")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(inventory.sort_values("months_of_inventory"), use_container_width=True)

    st.info(
        "Suggested action: low-stock wines may need production/purchasing attention. Overstock wines may need tasting room promotions, club offers, or event features."
    )

# ----------------------------
# Forecasting
# ----------------------------
elif page == "Forecasting":
    st.subheader("Revenue Forecasting")

    daily = sales.groupby("date", as_index=False)["revenue"].sum()
    daily["day_num"] = np.arange(len(daily))
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month

    X = daily[["day_num", "day_of_week", "month"]]
    y = daily["revenue"]

    model = LinearRegression()
    model.fit(X, y)

    days_forward = st.slider("Forecast days forward", 7, 120, 30)
    future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=days_forward)
    future = pd.DataFrame({
        "date": future_dates,
        "day_num": np.arange(len(daily), len(daily) + days_forward),
        "day_of_week": future_dates.dayofweek,
        "month": future_dates.month
    })
    future["forecast_revenue"] = model.predict(future[["day_num", "day_of_week", "month"]])
    future["forecast_revenue"] = future["forecast_revenue"].clip(lower=0)

    combined = pd.concat([
        daily[["date", "revenue"]].rename(columns={"revenue": "amount"}).assign(type="Actual"),
        future[["date", "forecast_revenue"]].rename(columns={"forecast_revenue": "amount"}).assign(type="Forecast")
    ])

    fig = px.line(combined, x="date", y="amount", color="type", title="Actual + Forecast Revenue")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(future, use_container_width=True)

# ----------------------------
# AI Insights
# ----------------------------
elif page == "AI Insights":
    st.subheader("AI-Style Business Insights")

    insights = []

    wine_rev = sales.groupby("wine")["revenue"].sum().sort_values(ascending=False)
    top_wine = wine_rev.index[0]
    bottom_wine = wine_rev.index[-1]
    insights.append(f"**Sales:** {top_wine} is the top revenue wine. Consider featuring it in premium tastings or club shipments.")
    insights.append(f"**Sales risk:** {bottom_wine} is the lowest revenue wine. Consider a promotion, pairing event, or tasting room script improvement.")

    channel_rev = sales.groupby("channel")["revenue"].sum().sort_values(ascending=False)
    insights.append(f"**Channel mix:** {channel_rev.index[0]} is the strongest channel. Try to convert more of those buyers into wine club members.")

    if len(events_df):
        event_profit = events_df.groupby("event_type")["profit"].mean().sort_values(ascending=False)
        insights.append(f"**Events:** {event_profit.index[0]} has the highest average profit per event. Consider scheduling more of these.")
        insights.append(f"**Events:** {event_profit.index[-1]} has the weakest average profit. Review pricing, labor, vendors, and marketing spend.")

    if "skipped_shipments" in club.columns:
        risk_proxy = club[(club["skipped_shipments"] >= 2) | (club["days_since_last_purchase"] > 90)]
        insights.append(f"**Wine club:** {len(risk_proxy)} members show churn warning signs. Prioritize personal outreach before the next shipment.")

    low_stock = inventory[inventory["months_of_inventory"] < 1.5]
    overstock = inventory[inventory["months_of_inventory"] > 6]
    if len(low_stock):
        insights.append(f"**Inventory:** {', '.join(low_stock['wine'].tolist())} may run low soon.")
    if len(overstock):
        insights.append(f"**Inventory:** {', '.join(overstock['wine'].tolist())} may be overstocked. Use targeted discounts or event pours.")

    for item in insights:
        st.markdown(f"- {item}")

    st.divider()
    st.subheader("Manager Talking Points")
    st.write(
        """
        Use these as interview/conversation points:

        1. “I built a dashboard that shows which wines and events drive the most profit.”
        2. “It can flag wine club members who may cancel based on purchase behavior.”
        3. “It helps management see inventory risk before it becomes a cash flow problem.”
        4. “My goal is to connect hospitality data with finance decisions.”
        """
    )

# ----------------------------
# CSV Templates
# ----------------------------
elif page == "CSV Templates":
    st.subheader("CSV Templates")
    st.write("Use these formats if a winery wants to test the app with its own data.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Sales CSV")
        st.dataframe(sample_sales.head(), use_container_width=True)
        download_template(sample_sales, "Sales")

        st.markdown("### Wine Club CSV")
        st.dataframe(sample_club.head(), use_container_width=True)
        download_template(sample_club, "Wine Club")

    with c2:
        st.markdown("### Events CSV")
        st.dataframe(sample_events.head(), use_container_width=True)
        download_template(sample_events, "Events")

        st.markdown("### Inventory CSV")
        st.dataframe(sample_inventory.head(), use_container_width=True)
        download_template(sample_inventory, "Inventory")

st.sidebar.divider()
st.sidebar.caption("Built for wineries that want better visibility into revenue, retention, events, and inventory.")
