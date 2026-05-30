# Winery Intelligence Platform

A Streamlit SaaS-style analytics platform for wineries.

## Features

- Demo login page
- Executive dashboard
- Sales analytics
- Wine club churn prediction
- Customer intelligence and segmentation
- Event ROI dashboard
- Inventory dashboard
- Financial dashboard with EBITDA, CAC, LTV, and operating margin
- Revenue forecasting
- Multi-winery comparison
- OpenAI-powered winery chatbot with fallback built-in insights
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

Upload these files:

```text
app.py
requirements.txt
README.md
```

Optional for the AI chatbot:

In Streamlit Cloud, go to App settings → Secrets and add:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

The app still works without the API key; it will use built-in analytics insights instead.
