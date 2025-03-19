"""
theme.py
🎨 Defines the color theme and global styling for OKXsignal.
"""

import streamlit as st

def inject_global_css():
    """
    Injects global CSS styles for consistent theming across all pages.
    """
    primary_color = "#03fcb6"
    background_light = "#ffffff"
    background_dark = "#1e1e1e"
    text_light = "#222222"
    text_dark = "#ffffff"

    css = f"""
    /* 🌟 Global Theme Variables */
    :root {{
        --primary-color: {primary_color};
        --background-light: {background_light};
        --background-dark: {background_dark};
        --text-light: {text_light};
        --text-dark: {text_dark};
        --box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }}

    /* ☀️ Light Mode */
    [data-theme="light"] {{
        background-color: var(--background-light);
        color: var(--text-light);
    }}

    /* 🌙 Dark Mode */
    [data-theme="dark"] {{
        background-color: var(--background-dark);
        color: var(--text-dark);
    }}

    /* 🎨 Sidebar Styling */
    .stSidebar {{
        background-color: var(--background-light);
        box-shadow: var(--box-shadow);
        transition: background-color 0.3s ease-in-out;
    }}
    [data-theme="dark"] .stSidebar {{
        background-color: var(--background-dark);
    }}

    /* 🔘 Sidebar Radio Buttons */
    .stRadio label {{
        color: var(--primary-color) !important;
    }}
    div[role='radiogroup'] label[data-baseweb='radio'] > div:first-child {{
        background-color: var(--primary-color) !important;
    }}

    /* 🔘 Buttons */
    .stButton > button {{
        background-color: var(--primary-color) !important;
        border-radius: 8px;
        color: var(--text-light);
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.2s ease-in-out;
    }}
    .stButton > button:hover {{
        background-color: #02e3a5 !important;
    }}
    [data-theme="dark"] .stButton > button {{
        color: var(--text-dark);
    }}

    /* 📦 Containers */
    .stContainer {{
        background-color: var(--background-light);
        border-radius: 10px;
        padding: 20px;
        box-shadow: var(--box-shadow);
    }}
    [data-theme="dark"] .stContainer {{
        background-color: var(--background-dark);
    }}

    /* 📋 Dropdowns & Selects */
    .stSelectbox, .stMultiSelect {{
        border: 1px solid var(--primary-color) !important;
    }}

    /* ✨ Elevated Containers */
    .st-emotion-cache-16txtl3 {{
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15);
        border-radius: 8px;
    }}
    """

    # Inject the CSS into Streamlit
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def get_theme():
    """
    Returns the current theme settings for use in Python functions.
    """
    dark_mode = st.session_state.get("dark_mode", False)
    
    theme = {
        "primary": "#03fcb6",
        "background": "#1e1e1e" if dark_mode else "#ffffff",
        "text": "#ffffff" if dark_mode else "#222222",
        "box_shadow": "0px 4px 10px rgba(0, 0, 0, 0.1)",
    }
    return theme
