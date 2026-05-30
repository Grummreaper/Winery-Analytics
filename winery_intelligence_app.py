# app.py
# Winery Intelligence Platform
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

# ------------------------- LOGIN -------------------------
def login():
    st.title("🍷 Winery Intelligence Platform")
    st.caption("Demo login for winery owners, managers, and analysts.")
    with st.form("login"):
        u = st.text_input("Username", value="demo")
        p = st.text_input("Password", value="winery123", type="password")
        ok = st.form_submit_button("Login")
    st.info("Demo login: username `demo`, password `winery123`")
    if ok:
        if u == "demo" and p == "winery123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Wrong login. Use demo / winery123.")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    login()
    st.stop()

# ------------------------- SAMPLE DATA -------------------------
@st.cache_data
def sample_data(seed=42, n_days=365):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    wineries = ["Livermore Estate", "Napa Tasting Room", "Sonoma Reserve"]
    wines = ["Cabernet Sauvignon", "Chardonnay", "Merlot", "Rosé", "Sauvignon Blanc", "Pinot Noir"]
    channels = ["Tasting Room", "Wine Club", "Event", "Online", "Wholesale"]
    events_list = ["Live Music", "Food Truck", "Release Party", "Wedding", "Corporate Event"]
    price = {"Cabernet Sauvignon":48,"Chardonnay":34,"Merlot":39,"Rosé":29,"Sauvignon Blanc":31,"Pinot Noir":44}
    cost = {"Cabernet Sauvignon":18,"Chardonnay":12,"Merlot":14,"Rosé":10,"Sauvignon Blanc":11,"Pinot Noir":16}

    sales_rows = []
    for d in dates:
        weekend = d.weekday() >= 5
        season = 1.25 if d.month in [5,6,7,8,9,10] else .9
        for wny in wineries:
            boost = {"Livermore Estate":1.0,"Napa Tasting Room":1.18,"Sonoma Reserve":.92}[wny]
            for _ in range(rng.integers(8, 25 if weekend else 14)):
                wine = rng.choice(wines)
                channel = rng.choice(channels, p=[.38,.25,.17,.12,.08])
                bottles = int(max(1, rng.normal(3.2 if channel != "Wholesale" else 18, 1.8)))
                discount = rng.choice([0,.05,.10,.15,.20], p=[.55,.18,.15,.08,.04])
                revenue = bottles * price[wine] * (1 - discount) * season * boost
                cogs = revenue * rng.uniform(.28,.42)
                sales_rows.append([d, wny, f"C{rng.integers(1000,1999)}", wine, channel, bottles, round(revenue,2), round(cogs,2)])
    sales = pd.DataFrame(sales_rows, columns=["date","winery","customer_id","wine","channel","bottles_sold","revenue","cogs"])

    event_rows = []
    for d in dates:
        if d.weekday() in [4,5,6]:
            for wny in wineries:
                if rng.random() < .31:
                    et = rng.choice(events_list)
                    attendance = int(max(25, rng.normal(95 if et != "Wedding" else 150, 45)))
                    ticket = attendance * rng.uniform(10,55)
                    wine_sales = attendance * rng.uniform(18,65)
                    labor = attendance * rng.uniform(4,11)
                    vendor = rng.uniform(250,2600)
                    marketing = rng.uniform(75,950)
                    event_rows.append([d, wny, et, attendance, round(ticket,2), round(wine_sales,2), round(labor,2), round(vendor,2), round(marketing,2)])
    events = pd.DataFrame(event_rows, columns=["date","winery","event_type","attendance","ticket_revenue","wine_sales","labor_cost","vendor_cost","marketing_cost"])
    events["total_revenue"] = events["ticket_revenue"] + events["wine_sales"]
    events["total_cost"] = events["labor_cost"] + events["vendor_cost"] + events["marketing_cost"]
    events["profit"] = events["total_revenue"] - events["total_cost"]

    club_rows = []
    for wny in wineries:
        starts = pd.to_datetime(rng.choice(dates[:-30], size=250, replace=True))
        for i, sd in enumerate(starts, 1):
            visits = int(rng.poisson(2.1))
            purchases = int(rng.poisson(3.4))
            aov = round(max(28, rng.normal(118,48)), 2)
            months = max(1, int((dates[-1] - sd).days / 30))
            emails = int(rng.integers(0,12))
            skips = int(rng.choice([0,1,2,3], p=[.58,.25,.12,.05]))
            days_last = int(rng.integers(1,180))
            churned = int((skips >= 2 and days_last > 75) or rng.random() < .08)
            club_rows.append([wny, f"{wny[:3].upper()}-M{i:04d}", sd.date(), months, visits, purchases, aov, emails, skips, days_last, churned])
    club = pd.DataFrame(club_rows, columns=["winery","member_id","join_date","months_member","visits_90d","purchases_90d","avg_order_value","emails_opened_90d","skipped_shipments","days_since_last_purchase","churned"])

    inv_rows = []
    for wny in wineries:
        for wine in wines:
            bottles = int(rng.integers(200,3600))
            forecast = int(rng.integers(90,620))
            inv_rows.append([wny, wine, bottles, forecast, cost[wine], price[wine]])
    inventory = pd.DataFrame(inv_rows, columns=["winery","wine","bottles_on_hand","monthly_sales_forecast","unit_cost","retail_price"])
    inventory["months_of_inventory"] = inventory["bottles_on_hand"] / inventory["monthly_sales_forecast"]
    inventory["inventory_value_cost"] = inventory["bottles_on_hand"] * inventory["unit_cost"]
    inventory["potential_retail_value"] = inventory["bottles_on_hand"] * inventory["retail_price"]

    customers = sales.groupby(["winery","customer_id"], as_index=False).agg(total_revenue=("revenue","sum"), orders=("date","count"), last_purchase=("date","max"), bottles=("bottles_sold","sum"))
    customers["days_since_last_purchase"] = (sales["date"].max() - customers["last_purchase"]).dt.days
    customers["avg_order_value"] = customers["total_revenue"] / customers["orders"]
    customers["segment"] = np.select(
        [(customers.total_revenue > customers.total_revenue.quantile(.85)) & (customers.days_since_last_purchase < 60),
         (customers.orders >= 6) & (customers.days_since_last_purchase < 90),
         customers.days_since_last_purchase > 120,
         customers.orders <= 2],
        ["VIP", "Loyal", "At Risk", "New / Low Frequency"], default="Regular")
    return sales, events, club, inventory, customers

# ------------------------- HELPERS -------------------------
def clean_dates(df):
    for c in ["date", "join_date", "last_purchase"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.dropna(subset=["date"]) if "date" in df.columns else df

def load(upload, sample):
    return sample.copy() if upload is None else pd.read_csv(upload)

def money(x): return f"${x:,.0f}"
def pct(x): return f"{x:.1%}"

def churn_scores(df):
    req = ["months_member","visits_90d","purchases_90d","avg_order_value","emails_opened_90d","skipped_shipments","days_since_last_purchase","churned"]
    if not all(c in df.columns for c in req):
        return df, None, None
    X, y = df[req[:-1]], df["churned"]
    model = RandomForestClassifier(n_estimators=250, random_state=42).fit(X, y)
    out = df.copy()
    out["churn_risk_score"] = model.predict_proba(X)[:,1]
    out["risk_level"] = pd.cut(out.churn_risk_score, [0,.33,.66,1], labels=["Low","Medium","High"])
    imp = pd.DataFrame({"feature":req[:-1], "importance":model.feature_importances_}).sort_values("importance", ascending=False)
    return out, model, imp

def financials(sales, events, club, marketing, overhead):
    revenue = sales.revenue.sum(); cogs = sales.cogs.sum(); gp = revenue - cogs
    event_profit = events.profit.sum() if len(events) else 0
    ebitda = gp + event_profit - marketing - overhead
    customers = sales.customer_id.nunique() if "customer_id" in sales.columns else max(len(club),1)
    cac = marketing / max(customers, 1)
    avg_order = revenue / max(len(sales), 1)
    churn = club.churned.mean() if "churned" in club.columns and len(club) else 0
    ltv = avg_order * 4 * 3 * (1 - churn)
    return {"revenue":revenue,"cogs":cogs,"gross_profit":gp,"gross_margin":gp/revenue if revenue else 0,"event_profit":event_profit,"marketing":marketing,"overhead":overhead,"ebitda":ebitda,"operating_margin":ebitda/revenue if revenue else 0,"cac":cac,"ltv":ltv,"payback_orders":cac/max(avg_order*(gp/revenue if revenue else 0),1)}

def pdf_report(fin, insights):
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Helvetica", "B", 18); pdf.cell(0, 12, "Winery Intelligence Executive Report", ln=True)
    pdf.set_font("Helvetica", "", 10); pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 9, "Financial Snapshot", ln=True)
    pdf.set_font("Helvetica", "", 11)
    rows = [("Revenue",money(fin['revenue'])),("Gross Profit",money(fin['gross_profit'])),("Gross Margin",pct(fin['gross_margin'])),("Event Profit",money(fin['event_profit'])),("EBITDA Estimate",money(fin['ebitda'])),("Operating Margin",pct(fin['operating_margin'])),("CAC Estimate",money(fin['cac'])),("LTV Estimate",money(fin['ltv']))]
    for k,v in rows:
        pdf.cell(70,8,k); pdf.cell(0,8,v,ln=True)
    pdf.ln(5); pdf.set_font("Helvetica", "B", 14); pdf.cell(0, 9, "Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for item in insights[:8]:
        pdf.multi_cell(0, 7, "- " + item.replace("**", ""))
    out = pdf.output(dest="S")
    return out.encode("latin-1") if isinstance(out, str) else bytes(out)

# ------------------------- LOAD DATA -------------------------
s_sales, s_events, s_club, s_inventory, s_customers = sample_data()
st.sidebar.title("🍷 Winery Intelligence")
if st.sidebar.button("Log out"):
    st.session_state.logged_in = False; st.rerun()

sales = clean_dates(load(st.sidebar.file_uploader("Sales CSV", type="csv"), s_sales))
events = clean_dates(load(st.sidebar.file_uploader("Events CSV", type="csv"), s_events))
club = clean_dates(load(st.sidebar.file_uploader("Wine Club CSV", type="csv"), s_club))
inventory = load(st.sidebar.file_uploader("Inventory CSV", type="csv"), s_inventory)
customers = clean_dates(load(st.sidebar.file_uploader("Customer CSV", type="csv"), s_customers))

locations = sorted(sales.winery.unique()) if "winery" in sales.columns else ["All"]
selected = st.sidebar.multiselect("Select winery location(s)", locations, default=locations)
for name in ["sales","events","club","inventory","customers"]:
    df = globals()[name]
    if "winery" in df.columns:
        globals()[name] = df[df.winery.isin(selected)]

marketing = st.sidebar.number_input("Estimated marketing spend", 0, value=25000, step=5000)
overhead = st.sidebar.number_input("Estimated overhead", 0, value=90000, step=5000)
page = st.sidebar.radio("Choose dashboard", ["Executive Dashboard","Sales Analytics","Wine Club Churn","Customer Intelligence","Event ROI","Inventory","Financial Dashboard","Revenue Forecasting","Multi-Winery Comparison","AI Winery Chatbot","PDF Reports","CSV Templates"])

club_scored, model, importance = churn_scores(club)
fin = financials(sales, events, club, marketing, overhead)

def insights():
    out = []
    if len(sales):
        wr = sales.groupby("wine").revenue.sum().sort_values(ascending=False)
        cr = sales.groupby("channel").revenue.sum().sort_values(ascending=False)
        out.append(f"Top wine by revenue is {wr.index[0]} with {money(wr.iloc[0])} in sales.")
        out.append(f"Lowest revenue wine is {wr.index[-1]}; consider a targeted tasting room promotion or pairing event.")
        out.append(f"Strongest channel is {cr.index[0]}, representing {pct(cr.iloc[0]/cr.sum())} of revenue.")
    if len(events):
        ep = events.groupby("event_type").profit.mean().sort_values(ascending=False)
        out.append(f"Most profitable event type on average is {ep.index[0]}; consider scheduling more of these.")
        out.append(f"Weakest event type is {ep.index[-1]}; review ticket price, labor, vendor, and marketing costs.")
    if len(club_scored) and "risk_level" in club_scored.columns:
        out.append(f"{(club_scored.risk_level == 'High').sum()} wine club members are high-risk and should get personal follow-up.")
    low = inventory[inventory.months_of_inventory < 1.5].wine.unique().tolist() if len(inventory) else []
    over = inventory[inventory.months_of_inventory > 6].wine.unique().tolist() if len(inventory) else []
    if low: out.append("Low-stock wines: " + ", ".join(low) + ".")
    if over: out.append("Possible overstock wines: " + ", ".join(over) + ".")
    out.append("Operating margin is positive." if fin["operating_margin"] >= 0 else "Operating margin is negative under the current cost assumptions.")
    return out

recos = insights()
st.title("Winery Intelligence Platform")
st.caption("Finance, operations, customer, and AI analytics for winery owners and managers.")

# ------------------------- PAGES -------------------------
if page == "Executive Dashboard":
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Revenue", money(fin['revenue'])); c2.metric("Gross Profit", money(fin['gross_profit'])); c3.metric("Gross Margin", pct(fin['gross_margin'])); c4.metric("EBITDA", money(fin['ebitda'])); c5.metric("Club Churn", pct(club.churned.mean()))
    left,right = st.columns(2)
    with left:
        monthly = sales.groupby(pd.Grouper(key="date", freq="ME"), as_index=False).revenue.sum()
        st.plotly_chart(px.line(monthly, x="date", y="revenue", title="Monthly Revenue"), use_container_width=True)
    with right:
        by_channel = sales.groupby("channel", as_index=False).revenue.sum().sort_values("revenue", ascending=False)
        st.plotly_chart(px.bar(by_channel, x="channel", y="revenue", title="Revenue by Channel"), use_container_width=True)
    st.subheader("Executive Recommendations")
    for r in recos: st.markdown(f"- {r}")

elif page == "Sales Analytics":
    by = sales.groupby("wine", as_index=False).agg(revenue=("revenue","sum"), cogs=("cogs","sum"), bottles=("bottles_sold","sum"))
    by["gross_profit"] = by.revenue - by.cogs; by["gross_margin"] = by.gross_profit / by.revenue
    st.metric("Average Bottle Revenue", money(by.revenue.sum()/max(by.bottles.sum(),1)))
    a,b = st.columns(2)
    a.plotly_chart(px.bar(by.sort_values("revenue", ascending=False), x="wine", y="revenue", title="Revenue by Wine"), use_container_width=True)
    b.plotly_chart(px.bar(by.sort_values("gross_margin", ascending=False), x="wine", y="gross_margin", title="Gross Margin by Wine"), use_container_width=True)
    st.dataframe(by.sort_values("revenue", ascending=False), use_container_width=True)

elif page == "Wine Club Churn":
    if model is None: st.error("Wine club CSV is missing required columns.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Members", f"{len(club_scored):,}"); c2.metric("Churn Rate", pct(club_scored.churned.mean())); c3.metric("High Risk", f"{(club_scored.risk_level=='High').sum():,}"); c4.metric("Avg Order", money(club_scored.avg_order_value.mean()))
        a,b = st.columns(2)
        counts = club_scored.risk_level.value_counts().reset_index(); counts.columns=["risk_level","members"]
        a.plotly_chart(px.bar(counts, x="risk_level", y="members", title="Risk Distribution"), use_container_width=True)
        b.plotly_chart(px.bar(importance, x="feature", y="importance", title="Main Churn Drivers"), use_container_width=True)
        high = club_scored.sort_values("churn_risk_score", ascending=False).head(50)
        st.dataframe(high, use_container_width=True)
        st.download_button("Download High-Risk Member List", high.to_csv(index=False), "high_risk_members.csv", "text/csv")

elif page == "Customer Intelligence":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers", f"{customers.customer_id.nunique():,}"); c2.metric("Avg Order", money(customers.avg_order_value.mean())); c3.metric("Avg Customer Revenue", money(customers.total_revenue.mean())); c4.metric("At Risk", f"{(customers.segment=='At Risk').sum():,}")
    counts = customers.segment.value_counts().reset_index(); counts.columns=["segment","customers"]
    a,b = st.columns(2)
    a.plotly_chart(px.bar(counts, x="segment", y="customers", title="Customer Segments"), use_container_width=True)
    b.plotly_chart(px.scatter(customers, x="orders", y="total_revenue", color="segment", title="Orders vs Revenue"), use_container_width=True)
    st.dataframe(customers.sort_values("total_revenue", ascending=False), use_container_width=True)

elif page == "Event ROI":
    summary = events.groupby("event_type", as_index=False).agg(events=("event_type","count"), attendance=("attendance","sum"), revenue=("total_revenue","sum"), cost=("total_cost","sum"), profit=("profit","sum"))
    summary["roi"] = summary.profit / summary.cost
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Event Revenue", money(summary.revenue.sum())); c2.metric("Event Cost", money(summary.cost.sum())); c3.metric("Event Profit", money(summary.profit.sum())); c4.metric("Event ROI", pct(summary.profit.sum()/max(summary.cost.sum(),1)))
    a,b = st.columns(2)
    a.plotly_chart(px.bar(summary.sort_values("profit", ascending=False), x="event_type", y="profit", title="Profit by Event Type"), use_container_width=True)
    b.plotly_chart(px.scatter(events, x="attendance", y="profit", color="event_type", title="Attendance vs Profit"), use_container_width=True)
    st.dataframe(summary.sort_values("profit", ascending=False), use_container_width=True)

elif page == "Inventory":
    inventory["status"] = np.select([inventory.months_of_inventory < 1.5, inventory.months_of_inventory > 6], ["Low Stock","Overstock"], default="Healthy")
    c1,c2,c3 = st.columns(3)
    c1.metric("Inventory Cost Value", money(inventory.inventory_value_cost.sum())); c2.metric("Potential Retail Value", money(inventory.potential_retail_value.sum())); c3.metric("Low Stock Items", f"{(inventory.status=='Low Stock').sum():,}")
    st.plotly_chart(px.bar(inventory, x="wine", y="months_of_inventory", color="status", title="Months of Inventory"), use_container_width=True)
    st.dataframe(inventory.sort_values("months_of_inventory"), use_container_width=True)

elif page == "Financial Dashboard":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Revenue", money(fin['revenue'])); c2.metric("Gross Margin", pct(fin['gross_margin'])); c3.metric("EBITDA", money(fin['ebitda'])); c4.metric("Operating Margin", pct(fin['operating_margin']))
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("CAC", money(fin['cac'])); c6.metric("Estimated LTV", money(fin['ltv'])); c7.metric("Payback Orders", f"{fin['payback_orders']:.1f}"); c8.metric("Event Profit", money(fin['event_profit']))
    bridge = pd.DataFrame({"metric":["Revenue","COGS","Gross Profit","Event Profit","Marketing","Overhead","EBITDA"],"amount":[fin['revenue'],-fin['cogs'],fin['gross_profit'],fin['event_profit'],-fin['marketing'],-fin['overhead'],fin['ebitda']]})
    st.plotly_chart(px.bar(bridge, x="metric", y="amount", title="Profit Bridge"), use_container_width=True)
    st.dataframe(bridge, use_container_width=True)

elif page == "Revenue Forecasting":
    daily = sales.groupby("date", as_index=False).revenue.sum()
    daily["day_num"] = np.arange(len(daily)); daily["day_of_week"] = daily.date.dt.dayofweek; daily["month"] = daily.date.dt.month
    lm = LinearRegression().fit(daily[["day_num","day_of_week","month"]], daily.revenue)
    days = st.slider("Forecast days", 7, 180, 60)
    dates = pd.date_range(daily.date.max() + pd.Timedelta(days=1), periods=days)
    future = pd.DataFrame({"date":dates,"day_num":np.arange(len(daily), len(daily)+days),"day_of_week":dates.dayofweek,"month":dates.month})
    future["forecast_revenue"] = lm.predict(future[["day_num","day_of_week","month"]]).clip(min=0)
    combined = pd.concat([daily[["date","revenue"]].rename(columns={"revenue":"amount"}).assign(type="Actual"), future[["date","forecast_revenue"]].rename(columns={"forecast_revenue":"amount"}).assign(type="Forecast")])
    st.metric("Forecasted Revenue", money(future.forecast_revenue.sum()))
    st.plotly_chart(px.line(combined, x="date", y="amount", color="type", title="Actual vs Forecast Revenue"), use_container_width=True)
    st.dataframe(future, use_container_width=True)

elif page == "Multi-Winery Comparison":
    comp = s_sales.groupby("winery", as_index=False).agg(revenue=("revenue","sum"), cogs=("cogs","sum"), bottles=("bottles_sold","sum"), customers=("customer_id","nunique"))
    comp["gross_profit"] = comp.revenue - comp.cogs; comp["gross_margin"] = comp.gross_profit / comp.revenue
    ev = s_events.groupby("winery", as_index=False).profit.sum().rename(columns={"profit":"event_profit"})
    ch = s_club.groupby("winery", as_index=False).churned.mean().rename(columns={"churned":"churn_rate"})
    comp = comp.merge(ev, on="winery", how="left").merge(ch, on="winery", how="left")
    st.plotly_chart(px.bar(comp, x="winery", y="revenue", title="Revenue by Winery"), use_container_width=True)
    st.plotly_chart(px.bar(comp, x="winery", y="gross_margin", title="Gross Margin by Winery"), use_container_width=True)
    st.dataframe(comp.sort_values("revenue", ascending=False), use_container_width=True)

elif page == "AI Winery Chatbot":
    st.subheader("AI Winery Chatbot")
    st.code("Examples:\nWhy did sales decline?\nWhich event is most profitable?\nWhat should management focus on this month?")
    q = st.text_area("Ask a business question")
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if st.button("Generate AI Answer"):
        if not q.strip(): st.warning("Type a question first.")
        elif OpenAI and api_key:
            client = OpenAI(api_key=api_key)
            summary = {"revenue":money(fin['revenue']),"gross_margin":pct(fin['gross_margin']),"ebitda":money(fin['ebitda']),"recommendations":recos}
            try:
                resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"You are a concise winery finance and operations analyst."},{"role":"user","content":f"Use this winery data summary: {summary}\nQuestion: {q}"}], temperature=.3)
                st.write(resp.choices[0].message.content)
            except Exception as e: st.error(f"OpenAI call failed: {e}")
        else:
            st.info("No OpenAI API key found. Using built-in analytics instead. Add OPENAI_API_KEY in Streamlit secrets to enable the chatbot.")
            for r in recos: st.markdown(f"- {r}")

elif page == "PDF Reports":
    st.subheader("PDF Executive Reports")
    for r in recos: st.markdown(f"- {r}")
    st.download_button("Download Executive PDF Report", pdf_report(fin, recos), "winery_executive_report.pdf", "application/pdf")

elif page == "CSV Templates":
    templates = {"sales_template.csv":s_sales.head(100), "events_template.csv":s_events.head(100), "wine_club_template.csv":s_club.head(100), "inventory_template.csv":s_inventory.head(100), "customers_template.csv":s_customers.head(100)}
    for fn, df in templates.items():
        st.markdown(f"### {fn}")
        st.dataframe(df.head(), use_container_width=True)
        st.download_button(f"Download {fn}", df.to_csv(index=False), fn, "text/csv")
