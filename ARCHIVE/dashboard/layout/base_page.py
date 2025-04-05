"""
base_page.py
📌 Provides a base layout for all pages.
"""

import streamlit as st

def apply_base_layout(page_function):
    """
    Applies the base layout to the page.
    :param page_function: The function that renders the actual page content.
    """

    st.container()
    page_function()  # Render the selected page
