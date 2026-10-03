# 🚀 StockWise: Deployment & Publishing Checklist

This guide provides step-by-step instructions to publish the repository to GitHub and deploy a live interactive demo for recruiters on **Streamlit Community Cloud** or **Render**.

---

## 1. Local Pre-Flight Verification Checklist

Before pushing to GitHub, verify that all local checks pass:

- [x] Run automated test suite:
  ```bash
  py -3 -m pytest
  ```
  *(Expected: 10 passed)*

- [x] Run linting check:
  ```bash
  py -3 -m flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
  ```

- [x] Verify database generation & pipeline execution:
  ```bash
  py -3 -m src.data.pipeline --generate-synthetic
  ```

- [x] Verify Streamlit app runs locally:
  ```bash
  py -3 -m streamlit run app.py
  ```

---

## 2. Pushing Repository to GitHub

1. Initialize Git repository (if not already initialized):
   ```bash
   git init
   git add .
   git commit -m "feat: complete StockWise explainable demand forecasting & inventory planning project"
   ```

2. Create a public repository on GitHub named `stockwise-demand-forecasting`.

3. Link local repository and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/stockwise-demand-forecasting.git
   git branch -M main
   git push -u origin main
   ```

---

## 3. Deploying to Streamlit Community Cloud (Free)

1. Log into [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Click **New App**.
3. Select your repository: `YOUR_USERNAME/stockwise-demand-forecasting`.
4. Branch: `main`
5. Main file path: `app.py`
6. Click **Deploy!**

> **Note on Free Hosting Limitations**: Streamlit Cloud free tier provides 1GB RAM. SQLite database file (`data/stockwise.db`) will automatically initialize on app start.

---

## 4. Deploying to Render / Hugging Face Spaces (Alternative)

- **Render Web Service**:
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
