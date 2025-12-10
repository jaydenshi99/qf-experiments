# Experiments

Contains various streamlit applications for exploring topics I find interesting in quantitative finance.

## Launching the Application

1. Clone the repository
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate the environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

### Running the Applications

**American vs European Options Visualizer:**
```bash
source .venv/bin/activate
streamlit run src/american/app.py
```

This will launch the Options Visualizer at `http://localhost:8501`

**Portfolio Hedging Application:**
```bash
source .venv/bin/activate
streamlit run src/hedge/app.py
```

This will launch the Portfolio Hedging app at `http://localhost:8501`
