# Winery Intelligence Platform

A Streamlit analytics platform for wineries.

## Features

- Clean login screen
- Hidden default Streamlit navigation
- Welcome / how-to-use page
- Data source badges: sample data vs uploaded data
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

Username: demo  
Password: winery123

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Required files:

```text
app.py
requirements.txt
README.md
```

Optional AI chatbot setup:

Go to Streamlit Cloud → Manage App → Settings → Secrets and add:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

The app still works without the API key. It will use built-in analytics insights instead.
