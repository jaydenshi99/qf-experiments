# Options Playground 🎮

An interactive web application for exploring and experimenting with options pricing models and strategies.

## Features

- 🌳 **Interactive Binomial Trees** - Visualize European vs American options
- ⚡ **Real-time Parameter Adjustment** - See changes instantly
- 📊 **Multiple Option Types** - Calls and puts with early exercise analysis
- 🎯 **Model Parameters Display** - View intermediate calculations (u, d, p, dt)
- 🚀 **Extensible Platform** - Ready for new features and analysis tools

## Quick Start

1. Clone the repository
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate the environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

### Running the Application

**Web Interface (Recommended):**
```bash
source .venv/bin/activate
streamlit run src/visualisation.py
```

This will launch the Options Playground at `http://localhost:8501`

**Command Line Interface:**
```bash
python src/cli.py --help
python src/cli.py --S0 100 --K 105 --T 0.25 --r 0.05 --sigma 0.2 --steps 10 --type call
```

To run the development server:
```bash
streamlit run src/app.py
```

## License

MIT License - see LICENSE file for details.
