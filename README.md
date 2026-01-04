# Experiments

Contains various streamlit applications for exploring topics I find interesting in quantitative finance.

View the streamlit applications here:
- [Binomial Options Pricing Model](https://qf-experiments-american-8cc3fyrekoq6tepubug2sk.streamlit.app/)
- [Portfolio Hedging](https://qf-experiments-hedge-5kaxtafgdgjappbx3ufxfhn.streamlit.app/)

## Setup

1. Clone the repository
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate the environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Running the Applications

**American vs European Options Visualiser:**
```bash
streamlit run src/american/app.py
```

**Portfolio Hedging Application:**
```bash
streamlit run src/hedge/app.py
```

Both applications will launch at `http://localhost:8501`
