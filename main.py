"""
main.py
🚀 Entry point for OKXsignal Dashboard (Streamlit).
"""

import streamlit as st
import os
import sys

# Ensure correct imports from dashboard/
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "dashboard"))

# Import layout & theming
from dashboard.layout.theme import get_theme, inject_global_css
from dashboard.layout.base_page import apply_base_layout
from dashboard.layout.navigation import build_sidebar

# Configure Streamlit
st.set_page_config(
    page_title="OKXsignal",
    page_icon="⚡",
    layout="wide",
)

# Inject global CSS (ensure styling is applied on startup)
inject_global_css()

# Load navigation sidebar
selected_page_name = build_sidebar()

# ✅ Dynamically load the correct page module
try:
    page_module = __import__(f"dashboard.pages.{selected_page_name}", fromlist=["show_page"])
    
    # Ensure the module has `show_page` function
    if hasattr(page_module, "show_page"):
        apply_base_layout(page_module.show_page)
    else:
        st.error(f"❌ Page '{selected_page_name}' does not have a `show_page()` function.")
except ImportError:
    st.error(f"❌ Page '{selected_page_name}' not found.")

