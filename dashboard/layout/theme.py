"""
theme.py
🎨 Defines global theming & styling for OKXsignal's Streamlit Dashboard.
"""

import streamlit as st

def inject_global_css():
    """Injects global CSS for consistent UI styling inspired by InsightBig."""
    custom_css = """
    <style>
        /* 📌 Global Styling */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        /* 🌍 Dark Theme Colors */
        :root {
            --primary-color: #03fcb6;
            --background-dark: #131722;
            --text-dark: #f6f6f6;
            --secondary-background: #0c0e15;
            --box-shadow: -6px 8px 20px 1px rgba(0, 0, 0, 0.52);
        }

        /* 🌙 Dark Mode Only */
        body {
            background-color: var(--background-dark);
            color: var(--text-dark);
        }

        /* 🎨 Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--secondary-background);
            padding: 20px;
            box-shadow: var(--box-shadow);
            border-radius: 10px;
        }

        /* 📦 Containers */
        div[data-testid="stVerticalBlock"] {
            background-color: var(--secondary-background);
            border-radius: 12px;
            padding: 20px;
            box-shadow: var(--box-shadow);
        }

        /* 🔘 Custom Radio Buttons */
        div[data-testid="stRadio"] label {
            color: var(--text-dark) !important;
            font-weight: bold !important;
        }

        /* 🚀 Custom Buttons */
        .stButton > button {
            background-color: var(--primary-color) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            font-weight: bold !important;
            transition: all 0.2s ease-in-out;
            box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.15);
        }
        .stButton > button:hover {
            background-color: #02e3a5 !important;
        }

        /* 🛠️ Forms & Dropdowns */
        .stSelectbox, .stMultiSelect {
            border: 0px solid var(--primary-color) !important;
            background-color: var(--secondary-background);
            color: var(--text-dark);
            border-radius: 5px;
            padding: 8px;
        }

        /* 📢 Custom Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-dark) !important;
            font-weight: bold !important;
        }

        /* 📝 Tables */
        table {
            border: 1px solid var(--primary-color) !important;
            border-radius: 10px !important;
        }

        /* 📦 Custom Containers */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            box-shadow: var(--box-shadow);
            border-radius: 10px;
        }

        /* 📏 Dashboard Padding */
        div.block-container {
            padding-left: 5rem !important;
            padding-right: 5rem !important;
            padding-top: 15px !important;
            padding-bottom: 40px !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
