# Winery Intelligence Platform

A Streamlit analytics platform for wineries that turns sales, event, wine club, customer, and inventory CSV data into business dashboards and recommendations.

## Features

- Demo login screen
- Clean upload validation and messy CSV column mapping
- Data Cleaning Report page
- Executive dashboard
- Sales analytics
- Wine club churn prediction
- Customer intelligence and segmentation
- Event ROI dashboard
- Inventory dashboard
- Financial dashboard with EBITDA, CAC, LTV, and operating margin
- Revenue forecasting
- Multi-winery comparison
- AI winery chatbot
- PDF executive reports
- CSV templates

## Demo Login

Username: `demo`  
Password: `winery123`

## Required Files for GitHub / Streamlit Cloud

```text
app.py
requirements.txt
README.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Setup

1. Push `app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Open Streamlit Cloud.
3. Connect the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.

## Optional OpenAI Chatbot Setup

Go to Streamlit Cloud → Manage App → Settings → Secrets and add:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

The app still works without the OpenAI API key. It will use built-in analytics insights instead.

## CSV Cleaning Logic

The app can accept imperfect CSVs and automatically map common column names.

Examples:

- `Sale Date`, `Transaction Date`, `Order Date` → `date`
- `Sales Amount`, `Gross Sales`, `Net Sales` → `revenue`
- `Qty`, `Quantity`, `Units` → `bottles_sold`
- `Product`, `Wine Name`, `Item`, `Varietal` → `wine`
- `Location`, `Site`, `Store` → `winery`

The Data Cleaning Report page shows what was changed, removed, or filled in.
