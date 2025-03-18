"""
main.py
🚀 Entry point for OKXsignal's Streamlit Dashboard
"""

import streamlit as st
import os
import sys

# 1) Adjust Python path to ensure we can import backend & dashboard modules
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "backend"))
sys.path.append(os.path.join(BASE_DIR, "dashboard"))

# 2) Optional: import custom CSS
def load_custom_css():
    css_file = os.path.join(BASE_DIR, "dashboard", "assets", "custom.css")
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3) Import your layout / navigation if desired
from dashboard.layout.navigation import build_sidebar  # example
# from dashboard.layout.theme import get_theme_settings  # example theming

# 4) Import pages (show them in a multi-page style or custom routing)
from dashboard.pages.home import show_home
from dashboard.pages.market_analysis.overview import show_market_overview
from dashboard.pages.portfolio.holdings import show_holdings
from dashboard.pages.trade_execution import show_trade_execution
# etc.

# 5) If you need to fetch market data (like from fetch_market.py):
# from backend.execution.fetch_market import main as fetch_market_data

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="OKXsignal", layout="wide")

def main():
    # Load CSS
    load_custom_css()

    # Optional: build a sidebar or top bar
    # E.g. "build_sidebar()" could return the selected page
    st.sidebar.title("OKXsignal Navigation")
    pages = {
        "🏠 Home": "home",
        "📈 Market Analysis": "market_overview",
        "💰 Portfolio": "holdings",
        "⚡ Trade Execution": "trade_execution",
    }
    chosen_page = st.sidebar.radio("Go to:", list(pages.keys()))

    # Show whichever page user picked
    if pages[chosen_page] == "home":
        show_home()
    elif pages[chosen_page] == "market_overview":
        show_market_overview()
    elif pages[chosen_page] == "holdings":
        show_holdings()
    elif pages[chosen_page] == "trade_execution":
        show_trade_execution()
    else:
        show_home()  # default fallback

if __name__ == "__main__":
    main()
