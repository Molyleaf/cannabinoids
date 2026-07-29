# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 确保项目根目录在 sys.path 首位
current_dir = Path(__file__).resolve().parent
app_dir = current_dir.parent
project_root = app_dir.parent

if str(current_dir) in sys.path:
    sys.path.remove(str(current_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.pipeline import run_pipeline
from app.privacy import record_sample_if_authorized, get_queue_stats
from app.i18n import init_i18n, get_text as _

# 1. 优先读取并确定 session_state 中的语言选择，初始化 i18n
if "language" not in st.session_state:
    st.session_state.language = "en_US"

_ = init_i18n()

st.set_page_config(
    page_title=_("NPS Spectrum Platform"),
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.header("🌐 " + _("Language"))
lang_options = {"English": "en_US", "简体中文": "zh_CN"}
current_idx = 0 if st.session_state.language == "en_US" else 1

selected_lang_label = st.sidebar.selectbox(
    _("Language Selector"),
    options=list(lang_options.keys()),
    index=current_idx,
    label_visibility="collapsed"
)

selected_lang_code = lang_options[selected_lang_label]
if st.session_state.language != selected_lang_code:
    st.session_state.language = selected_lang_code
    st.rerun()

# 渲染顶部标题与流程说明
st.title("🧪 " + _("NPS Spectrum Detection & Analysis Platform"))
st.caption(_("Upload mass spectrum files to quickly perform model inference and known structure retrieval."))

workflow_label = _("Analysis Workflow Instructions")
steps = [
    _("1. Upload File: Supports .msp and .mgf formats."),
    _("2. Select Model: Binary Model (Risk Evaluation) or Multi-class Model (Structure Classification)."),
    _("3. View Results: Automatic positive/negative inference; returns SMILES for similarity > 0.90.")
]
steps_text = "\n".join([f"- {step}" for step in steps])
st.info(f"**{workflow_label}：**\n\n{steps_text}")

# ==================== 侧边栏配置 ====================
st.sidebar.markdown("---")
st.sidebar.header("⚙️ " + _("Analysis Settings"))

model_choice_label = st.sidebar.radio(
    _("Select Model"),
    options=[_("Binary Model (Risk Evaluation)"), _("Multi-class Model (Structure Classification)")],
    index=0
)

model_type_key = "multi" if _("Multi-class Model (Structure Classification)") in model_choice_label else "binary"

min_similarity = st.sidebar.slider(
    _("Library Match Threshold (Similarity)"),
    min_value=0.50,
    max_value=0.99,
    value=0.90,
    step=0.01
)

st.sidebar.markdown("---")
st.sidebar.header("🔒 " + _("Data Sharing & Privacy Settings"))

user_consent = st.sidebar.checkbox(
    _("Consent to share spectrum data to help improve model performance"),
    value=False
)

user_name = ""
user_email = ""

if user_consent:
    st.sidebar.success(_("Thank you for supporting model optimization!"))
    user_name = st.sidebar.text_input(
        _("How should we address you"),
        placeholder="e.g. Chemist_A"
    )
    user_email = st.sidebar.text_input(
        _("Email Address"),
        placeholder="name@example.com"
    )
    if st.sidebar.button(_("Submit Information")):
        st.sidebar.success(_("Information submitted successfully!"))
else:
    st.sidebar.info("🛡️ **" + _("Privacy Protection") + "**：" + _("When data sharing is not selected, the system only displays detection results on the current page and will not save or upload any of your data."))

with st.sidebar.expander("ℹ️ " + _("About Data Sharing & Acknowledgments")):
    st.markdown(_("Data sharing directly helps improve model detection accuracy:"))
    st.markdown("- " + _("Model Update Notifications: Priority email notifications with update details when a new version is released."))
    st.markdown("- " + _("Contributor Acknowledgments: System acknowledgments in model update notes will address you as provided."))

# ==================== 主界面上传与分析 ====================
uploaded_file = st.file_uploader(
    _("Upload Mass Spectrum File (.msp or .mgf)"),
    type=["msp", "mgf"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    with st.spinner(f"Processing {filename}..."):
        try:
            result = run_pipeline(
                file_bytes=file_bytes,
                filename=filename,
                model_type=model_type_key,
                min_similarity=min_similarity
            )

            model_inf = result["model_inference"]
            entropy_match = result["entropy_match"]
            pred_prob = model_inf["risk_probability"] if model_type_key == "binary" else model_inf["confidence"]

            # 执行数据共享与复核逻辑
            privacy_res = record_sample_if_authorized(
                file_name=filename,
                peaks=result["peaks"],
                model_type=model_type_key,
                probability=pred_prob,
                predicted_result=model_inf["status_text"],
                user_consent=user_consent,
                user_name=user_name,
                user_email=user_email
            )

            st.success("🎉 " + _("Analysis complete!"))

            # 隐私与样本复核状态提示
            if not privacy_res["authorized"]:
                st.info("🔒 **" + _("Privacy Mode Active") + "**：" + _("Only current detection results are displayed; no data is retained in the system."))
            elif privacy_res["is_uncertain"]:
                st.warning("⚠️ **" + _("Boundary Sample Review Notice") + f"**：{_('Predicted probability is')} **{pred_prob:.2f}**，{_('falling in model uncertainty interval [0.3, 0.7]. Saved to pending review queue for expert review.')}")
            else:
                st.success("✅ **" + _("Detection Complete") + f"**：{_('Model prediction confidence is high')} ({pred_prob:.2f})，{_('results archived for model performance improvement.')}")

            st.markdown("---")

            # 结果摘要卡片
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(_("Analyzed File"), result["filename"])

            with col2:
                st.metric(_("Cleaned Peak Count"), result["num_cleaned_peaks"])

            with col3:
                status_disp = _(model_inf["status_text"]) if model_type_key == "binary" else f"{_('Positive 🎯')} ({_('Category')}: {_(model_inf['pred_class'])})"
                if result["is_positive"]:
                    st.metric(_("Model Determination"), status_disp, delta=_("Positive"), delta_color="inverse")
                else:
                    st.metric(_("Model Determination"), status_disp, delta=_("Negative"), delta_color="normal")

            with col4:
                if entropy_match["is_matched"]:
                    st.metric(_("Known Molecule Similarity"), f"{entropy_match['similarity_score']:.4f}", delta=_("Matched SMILES"), delta_color="normal")
                elif entropy_match["is_triggered"]:
                    st.metric(_("Known Molecule Similarity"), f"{entropy_match['similarity_score']:.4f}", delta=f"{_('Below Threshold')} ({min_similarity:.2f})", delta_color="off")
                else:
                    st.metric(_("Known Molecule Similarity"), _("Not Triggered"), delta=_("Negative Sample (Skipped)"), delta_color="off")

            st.markdown("---")

            # 详细报告与图表展示 Tab 页
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🧬 " + _("Results & Molecule Structure"),
                "📊 " + _("Spectrum Profile"),
                "📈 " + _("Prediction Probabilities"),
                "🔒 " + _("Data Sharing & Review Queue"),
                "📄 " + _("Raw Peak Data")
            ])

            with tab1:
                st.subheader(_("1. Model Inference Result"))
                if model_type_key == "binary":
                    st.write(f"- **{_('Evaluated Model')}**：{_(model_inf['model_name'])}")
                    st.write(f"- **{_('Positive/High Risk Probability')}**：`{model_inf['risk_probability']}` ({model_inf['risk_percentage']})")
                    st.write(f"- **{_('Determination Result')}**：**{_(model_inf['risk_level'])}**")
                else:
                    st.write(f"- **{_('Evaluated Model')}**：{_(model_inf['model_name'])}")
                    st.write(f"- **{_('Predicted Category')}**：**{_(model_inf['pred_class'])}**")
                    st.write(f"- **{_('Category Confidence')}**：`{model_inf['confidence']}` ({model_inf['confidence_percentage']})")

                st.markdown("---")
                st.subheader(_("2. Known Library Match Result"))
                if not entropy_match["is_triggered"]:
                    st.info(_("Current sample evaluated as negative by model, library search not triggered."))
                else:
                    st.write(f"- **{_('Highest Similarity Score')}**：`{entropy_match['similarity_score']:.4f}`")
                    st.write(f"- **{_('Matching Threshold')}**：`{min_similarity:.2f}`")

                    if entropy_match["is_matched"]:
                        st.success(f"{_('Successfully matched known molecule')} ({_('Similarity')} {entropy_match['similarity_score']:.4f} > {min_similarity:.2f})")
                        if entropy_match["matched_name"]:
                            st.write(f"- **{_('Compound Name')}**：`{entropy_match['matched_name']}`")
                        st.write(f"- **{_('Known Molecule SMILES Structure')}**：")
                        st.code(entropy_match["matched_smiles"], language="text")
                    else:
                        st.warning(f"{_('Sample predicted positive, but max similarity is')} `{entropy_match['similarity_score']:.4f}` ({_('<= threshold')} {min_similarity:.2f})。{_('No SMILES returned. Sample may be a novel positive derivative.')}")

            with tab2:
                st.subheader(_("Spectrum Profile") + " (m/z vs Relative Intensity %)")
                peaks = np.array(result["peaks"])
                if len(peaks) > 0:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    mzs = peaks[:, 0]
                    intensities = peaks[:, 1]
                    max_int = max(intensities) if len(intensities) > 0 else 1.0
                    norm_int = (intensities / max_int) * 100.0

                    ax.vlines(x=mzs, ymin=0, ymax=norm_int, linewidth=1.5)
                    ax.scatter(mzs, norm_int, s=10)
                    ax.set_xlabel(_("m/z"))
                    ax.set_ylabel(_("Relative Intensity (%)"))
                    ax.set_title(f"{_('Cleaned Spectrum Profile')}: {filename} ({len(peaks)} {_('peaks')})")
                    ax.grid(True, linestyle="--", alpha=0.5)

                    st.pyplot(fig)

            with tab3:
                st.subheader(_("Prediction Probabilities"))
                if model_type_key == "multi":
                    probs_dict = model_inf["probabilities"]
                    classes = list(probs_dict.keys())
                    translated_classes = [_(c) for c in classes]
                    probs_vals = [probs_dict[c] * 100 for c in classes]

                    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
                    bars = ax_bar.barh(translated_classes, probs_vals)
                    ax_bar.set_xlabel(_("Probability (%)"))
                    ax_bar.set_title(_("Multi-Classifier Prediction Probabilities"))
                    ax_bar.set_xlim(0, 100)
                    for bar in bars:
                        w = bar.get_width()
                        ax_bar.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9)
                    ax_bar.grid(True, linestyle=":", alpha=0.6)
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                else:
                    risk_p = model_inf["risk_probability"] * 100
                    st.write(f"**{_('Positive Probability')}**：`{risk_p:.2f}%`")
                    st.progress(int(min(max(risk_p, 0), 100)))

            with tab4:
                st.subheader(_("Data Sharing & Sample Management Details"))
                st.write(f"- **{_('Data Sharing Authorization')}**：`{_('Authorized') if user_consent else _('Not Authorized (Privacy Protected)')}`")
                if user_consent:
                    st.write(f"- **{_('Contributor Signature')}**：`{user_name if user_name else _('Anonymous Contributor')}`")
                    st.write(f"- **{_('Notification Email')}**：`{user_email if user_email else _('Not Provided')}`")
                st.write(f"- **{_('Model Prediction Probability ($P$)')}**：`{pred_prob:.4f}`")
                st.write(f"- **{_('Uncertainty Interval')}**：`[0.3, 0.7]`")
                st.write(f"- **{_('Sample Classification')}**：`{_('Uncertainty Interval (Pending Manual Review)') if privacy_res['is_uncertain'] else _('High Confidence Interval')}`")

                stats = get_queue_stats()
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric(_("Total Archived Contribution Samples"), stats["total_contributions"])
                with col_s2:
                    st.metric(_("Total Pending Review Samples"), stats["total_uncertain"])

            with tab5:
                st.subheader(_("Cleaned Raw Peak Data"))
                st.dataframe(
                    peaks,
                    column_config={"0": "m/z", "1": _("Intensity")},
                    width="stretch"
                )

        except Exception as e:
            st.error("❌ " + _("Error during analysis:") + f" {str(e)}")
