"""
base_page.py
📌 Provides a base layout for all pages.
"""

import streamlit as st
from dashboard.layout.theme import get_theme

def apply_base_layout(page_function):
    """
    Applies the base layout to the page.
    :param page_function: The function that renders the actual page content.
    """
    theme = get_theme()

    st.markdown(
        f"""
        <style>
            body {{
                background-color: {theme["background"]};
                color: {theme["text"]};
            }}
        </style>
        """, unsafe_allow_html=True
    )

    st.container()
    page_function()  # Render the selected page
