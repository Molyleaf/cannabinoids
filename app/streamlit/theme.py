# -*- coding: utf-8 -*-
import streamlit as st
from app.i18n import get_text as _


def render_hero_section(title: str, subtitle: str, steps: list):
    """
    Render top title and workflow instructions using native Streamlit components.
    """
    st.title(f"🧪 {title}")
    st.caption(subtitle)

    workflow_label = _("Analysis Workflow Instructions")
    steps_text = "\n".join([f"- {step}" for step in steps])
    st.info(f"**{workflow_label}:**\n\n{steps_text}")

