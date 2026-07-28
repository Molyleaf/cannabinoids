# -*- coding: utf-8 -*-
import streamlit as st
from app.i18n import get_text

def inject_beihang_theme():
    """
    原生模式：不注入任何自定义 CSS 样式，完全保留 Streamlit 纯原生渲染体系。
    """
    pass

def render_hero_section(title: str, subtitle: str, steps: list):
    """
    使用纯 Streamlit 原生组件渲染顶部标题与流程说明。
    """
    st.title(f"🧪 {title}")
    st.caption(subtitle)
    
    workflow_label = get_text("分析流程说明")
    steps_text = "\n".join([f"- {step}" for step in steps])
    st.info(f"**{workflow_label}：**\n\n{steps_text}")
