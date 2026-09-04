# -*- coding: utf-8 -*-
from typing import Optional
import streamlit as st

# English is the default source language; define English -> Chinese mapping dictionary
ZH_CN_TRANSLATIONS = {
    "NPS Spectral Intelligence Platform": "新精神活性物质质谱智能分析平台",
    "NPS Spectrum Platform": "NPS 质谱智能平台",
    "Upload mass spectrum files to quickly perform model inference and known structure retrieval.": "上传质谱文件，快速完成模型推断与已知分子结构检索。",
    "Analysis Workflow Instructions": "分析流程说明",
    "1. Select Model: binary model (SC versus NSC) or multi-class model (nine NPS categories).": "1. 选择模型：二分类模型 (合成大麻素 SC 与非合成大麻素 NSC) 或多分类模型 (九大类 NPS)。",
    "2. Sign the data sharing consent: consent to share spectrum data or not.": "2. 签署数据共享知情同意：同意共享谱图数据或选择仅本地分析。",
    "3. Upload File: Supports .msp and .mgf formats.": "3. 上传文件：支持 .msp 和 .mgf 格式。",
    "Analysis Settings": "检测配置",
    "Language": "语言选择",
    "Language Selector": "选择语言",
    "Select Model": "选择分析模型",
    "Binary Model (SC versus NSC)": "二分类模型 (SC vs NSC)",
    "Multi-class Model (nine NPS categories)": "多分类模型 (九大类 NPS)",
    "Binary Model (Risk Evaluation)": "二分类模型 (风险评估)",
    "Multi-class Model (Structure Classification)": "多分类模型 (结构分类)",
    "Library Match Threshold (Similarity)": "已知库匹配阈值 (相似度)",
    "Data Sharing & Privacy Settings": "数据共享与隐私配置",
    "Consent to share spectrum data to help improve model performance": "同意共享谱图数据，帮助改进模型性能",
    "How should we address you": "要怎么称呼您",
    "Email Address": "电子邮箱",
    "Thank you for supporting model optimization!": "感谢您支持模型优化！",
    "Submit Information": "提交信息",
    "Information submitted successfully!": "信息提交成功！",
    "Privacy Protection": "隐私保障",
    "When data sharing is not selected, the system only displays detection results on the current page and will not save or upload any of your data.": "未勾选共享选项时，系统仅在当前页面显示检测结果，不会保存或上传您的任何数据。",
    "About Data Sharing & Acknowledgments": "关于数据共享与致谢",
    "Data sharing directly helps improve model detection accuracy:": "数据共享将直接帮助提升模型的检测准确率：",
    "Model Update Notifications: Priority email notifications with update details when a new version is released.": "模型更新通知：模型发布新版本时，您将优先收到通知邮件，了解改进详情。",
    "Contributor Acknowledgments: System acknowledgments in model update notes will address you as provided.": "贡献致谢：在模型升级公告中，系统将以您提供的称呼表达致谢。",
    "OOD": "OOD",
    "Upload Mass Spectrum File (.msp or .mgf)": "上传质谱文件 (.msp 或 .mgf)",
    "Analyzed File": "检测文件",
    "Cleaned Peak Count": "清洗后有效峰数",
    "Model Determination": "模型检测判定",
    "Known Molecule Similarity": "已知分子相似度",
    "Positive": "阳性",
    "Negative": "阴性",
    "Positive (Positive)": "阳性 (Positive)",
    "Negative (Negative)": "阴性 (Negative)",
    "Matched SMILES": "已匹配 SMILES",
    "Below Threshold": "低于阈值",
    "Below 0.80 Threshold": "低于 0.80 阈值",
    "Below 0.90 Threshold": "低于 0.90 阈值",
    "Not Triggered": "未触发",
    "Negative Sample (Skipped)": "阴性样本免检索",
    "Results & Molecule Structure": "检测结果与分子结构",
    "Spectrum Profile": "质谱峰图",
    "Prediction Probabilities": "模型预测概率",
    "Data Sharing & Review Queue": "数据共享与复核状态",
    "Raw Peak Data": "原始峰数据",
    "1. Model Inference Result": "1. 模型推断结果",
    "Evaluated Model": "评估模型",
    "Positive/High Risk Probability": "阳性/高风险概率",
    "Determination Result": "判定结果",
    "Predicted Category": "预测分类",
    "Category Confidence": "分类置信度",
    "2. Known Library Match Result": "2. 已知分子库匹配结果",
    "Highest Similarity Score": "最高相似度得分",
    "Matching Threshold": "匹配阈值",
    "Compound Name": "化合物名称",
    "Known Molecule SMILES Structure": "已知分子 SMILES 结构式",
    "Data Sharing & Sample Management Details": "数据共享与样本数据管理明细",
    "Data Sharing Authorization": "数据共享授权",
    "Contributor Signature": "称呼署名",
    "Notification Email": "通知邮箱",
    "Uncertainty Interval": "不确定区间",
    "Sample Classification": "样本分类",
    "Total Archived Contribution Samples": "归档共享样本总数",
    "Total Pending Review Samples": "待复核不确定样本总数",
    "Cleaned Raw Peak Data": "清洗后原始峰数据",
    "UI Theme": "界面主题",
    "Dark Mode": "深色模式",
    "Analysis complete!": "分析完成！",
    "Privacy Mode Active": "隐私模式已生效",
    "Only current detection results are displayed; no data is retained in the system.": "当前仅显示本次检测结论，未在系统内留存任何数据。",
    "Boundary Sample Review Notice": "边界样本复核提示",
    "Predicted probability is": "本次预测概率为",
    "falling in model uncertainty interval [0.3, 0.7]. Saved to pending review queue for expert review.": "处于模型的不确定区间 [0.3, 0.7]。已为您存入待标注队列，并将由专家进行后续复核。",
    "Detection Complete": "检测完成",
    "Sample categorized as out-of-distribution (OOD).": "样本判定为分布外样本 (OOD)。",
    "Model prediction confidence is high": "模型预测置信度较高",
    "results archived for model performance improvement.": "结果已归档用于模型性能改进。",
    "Positive 🎯 (High Risk)": "阳性 🎯 (高风险)",
    "Negative 🛡️ (Low Risk)": "阴性 🛡️ (低风险)",
    "Positive (High Risk)": "阳性 🎯 (高风险)",
    "Negative (Low Risk)": "阴性 🛡️ (低风险)",
    "Positive 🎯": "阳性 🎯",
    "Category": "类别",
    "High Risk": "高风险阳性",
    "Low Risk": "低风险阴性",
    "Binary Model (@finetune 5-Fold Ensemble)": "二分类模型 (@finetune 5-Fold 集成)",
    "Multi-class Model": "多分类模型",
    "Multi-class Model (@multi_classifier)": "多分类模型",
    "Current sample evaluated as negative by model, library search not triggered.": "当前样本经模型判定为阴性，未触发已知库检索。",
    "Successfully matched known molecule": "成功匹配到已知分子",
    "Similarity": "相似度",
    "Sample predicted positive, but max similarity is": "样本表征为阳性，但最高相似度为",
    "<= threshold": "小于等于阈值",
    "No SMILES returned. Sample may be a novel positive derivative.": "未返回 SMILES 结构。此样本可能为新型阳性衍生化合物。",
    "Cleaned Spectrum Profile": "清洗质谱峰图",
    "peaks": "个峰",
    "m/z": "质荷比 (m/z)",
    "Relative Intensity (%)": "相对强度 (%)",
    "Probability (%)": "概率 (%)",
    "Multi-Classifier Prediction Probabilities": "多分类模型预测概率",
    "Positive Probability": "阳性概率",
    "Authorized": "已同意共享",
    "Not Authorized (Privacy Protected)": "未授权 (隐私保护)",
    "Anonymous Contributor": "匿名贡献者",
    "Not Provided": "未提供",
    "Model Prediction Probability ($P$)": "模型预测概率 ($P$)",
    "Uncertainty Interval (Pending Manual Review)": "不确定区间 (待人工复核)",
    "Pending Manual Review": "待人工复核",
    "High Confidence Interval": "高置信度区间",
    "Intensity": "响应强度",
    "Error during analysis:": "分析过程发生错误：",
    "Fentanyls": "芬太尼类",
    "Cathinones": "卡西酮类",
    "Synthetic Cathinones": "合成卡西酮类",
    "Synthetic Cannabinoids": "合成大麻素类",
    "Cannabinoids": "合成大麻素类",
    "Arylcyclohexylamines": "芳基环己胺类",
    "Benzodiazepines": "苯二氮䓬类",
    "Nitazenes": "硝嗪类",
    "Opiates": "阿片类",
    "Phenethylamines": "苯乙胺类",
    "Tryptamines": "色胺类",
}

_default_language = "en_US"


def _has_active_streamlit_session() -> bool:
    """Safely check if running within an active Streamlit execution runtime."""
    try:
        return hasattr(st, "runtime") and hasattr(st.runtime, "exists") and st.runtime.exists()
    except Exception:
        return False


def set_language(lang: str) -> None:
    """Set the active language."""
    global _default_language
    _default_language = lang
    if _has_active_streamlit_session():
        try:
            st.session_state.language = lang
        except Exception:
            pass


def get_current_language() -> str:
    """Get the active language from session_state if available, else default fallback."""
    if _has_active_streamlit_session():
        try:
            if "language" in st.session_state:
                return st.session_state.language
        except Exception:
            pass
    return _default_language


def init_i18n():
    """
    Initialize i18n module and synchronize with st.session_state.
    Default language: en_US.
    """
    if _has_active_streamlit_session():
        try:
            if "language" not in st.session_state:
                st.session_state.language = "en_US"
            st.session_state._ = get_text
        except Exception:
            pass
    return get_text


def get_text(message: str, lang: Optional[str] = None) -> str:
    """
    Retrieve translated text according to active language:
    - en_US: returns the original English text
    - zh_CN: returns the Chinese translation from ZH_CN_TRANSLATIONS
    """
    if lang is None:
        lang = get_current_language()
    if lang == "zh_CN":
        return ZH_CN_TRANSLATIONS.get(message, message)
    return message
