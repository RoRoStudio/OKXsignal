"""
main.py
🚀 Entry point for OKXsignal Dashboard (Streamlit).
"""

import streamlit as st

# ✅ First Streamlit command (MUST be first)
st.set_page_config(
    page_title="OKXsignal",
    page_icon="⚡",
    layout="wide",
)

import os
import sys

# Ensure correct imports from dashboard/
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "dashboard"))

# Import layout & theming
from dashboard.layout.theme import inject_global_css
from dashboard.layout.base_page import apply_base_layout
from dashboard.layout.navigation import build_sidebar
from dashboard.components.forms.filter_form import filter_form

# Inject global CSS (ensure styling is applied correctly)
inject_global_css()

# Load navigation sidebar
selected_page_name = build_sidebar()

# ✅ **Fix: Ensure module names match exactly**
valid_pages = {
    "home": "dashboard.pages.home",
    "market_analysis.overview": "dashboard.pages.market_analysis.overview",  
    "portfolio.holdings": "dashboard.pages.portfolio.holdings",
    "trade_execution": "dashboard.pages.trade_execution",
    "settings.user_prefs": "dashboard.pages.settings.user_prefs",
}

if selected_page_name in valid_pages:
    try:
        page_module = __import__(valid_pages[selected_page_name], fromlist=["show_page"])
        
        # Ensure the module has `show_page` function
        if hasattr(page_module, "show_page"):
            page_module.show_page()
        else:
            st.error(f"❌ Page '{selected_page_name}' does not have a `show_page()` function.")
    except ImportError as e:
        st.error(f"❌ Page '{selected_page_name}' not found. Error: {e}")
else:
    st.error(f"❌ Invalid page selection: {selected_page_name}")
