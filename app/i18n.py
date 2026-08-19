import gettext
import os
import struct
from pathlib import Path
import streamlit as st

LOCALES_DIR = Path(__file__).resolve().parent / "locales"

# 以英文为默认源码语言，定义 英文 -> 中文 映射表
ZH_CN_TRANSLATIONS = {
    "": "Project-Id-Version: 1.0\nContent-Type: text/plain; charset=UTF-8\n",
    "NPS Spectrum Detection & Analysis Platform": "新精神活性物质质谱检测与分析平台",
    "NPS Spectrum Platform": "NPS 质谱检测平台",
    "Upload mass spectrum files to quickly perform model inference and known structure retrieval.": "上传质谱文件，快速完成模型推断与已知分子结构检索。",
    "Analysis Workflow Instructions": "分析流程说明",
    "1. Upload File: Supports .msp and .mgf formats.": "1. 上传文件：支持 .msp 和 .mgf 格式。",
    "2. Select Model: Binary Model (Risk Evaluation) or Multi-class Model (Structure Classification).": "2. 选择模型：提供二分类模型 (风险评估) 和多分类模型 (结构分类)。",
    "3. View Results: Automatic positive/negative inference; returns SMILES for similarity > 0.90.": "3. 查看结果：模型自动推断阳性/阴性，并在高相似度 (>0.90) 时返回已知分子的 SMILES 结构式。",
    "Analysis Settings": "检测配置",
    "Language": "语言选择",
    "Language Selector": "选择语言",
    "Select Model": "选择分析模型",
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
    "Model prediction confidence is high": "模型预测置信度较高",
    "results archived for model performance improvement.": "结果已归档用于模型性能改进。",
    "Positive 🎯 (High Risk)": "阳性 🎯 (高风险)",
    "Negative 🛡️ (Low Risk)": "阴性 🛡️ (低风险)",
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
    "High Confidence Interval": "高置信度区间",
    "Intensity": "响应强度",
    "Error during analysis:": "分析过程发生错误：",
    "芬太尼": "芬太尼类",
    "卡西酮": "卡西酮类",
    "大麻素": "合成大麻素类",
    "Arylcyclohexylamines": "芳基环己胺类",
    "Benzodiazepines": "苯二氮䓬类",
    "Nitazenes": "硝嗪类",
    "Opiates": "阿片类",
    "Phenethylamines": "苯乙胺类",
    "Tryptamines": "色胺类"
}


def create_mo_file(mo_path: Path, translations: dict):
    """使用纯 Python 构建符合 GNU gettext 标准二进制规范的 .mo 文件"""
    keys = sorted(list(translations.keys()))
    N = len(keys)

    orig_table = []
    trans_table = []
    orig_bytes = b""
    trans_bytes = b""

    for k in keys:
        k_b = k.encode('utf-8')
        v_b = translations[k].encode('utf-8')

        orig_table.append((len(k_b), len(orig_bytes)))
        orig_bytes += k_b + b"\x00"

        trans_table.append((len(v_b), len(trans_bytes)))
        trans_bytes += v_b + b"\x00"

    header_size = 28
    orig_table_size = N * 8
    trans_table_size = N * 8
    start_strings = header_size + orig_table_size + trans_table_size

    table_orig_bytes = b""
    for length, rel_off in orig_table:
        table_orig_bytes += struct.pack("<II", length, start_strings + rel_off)

    table_trans_bytes = b""
    start_trans_strings = start_strings + len(orig_bytes)
    for length, rel_off in trans_table:
        table_trans_bytes += struct.pack("<II", length, start_trans_strings + rel_off)

    header = struct.pack("<IIIIIII", 0x950412de, 0, N, 28, 28 + orig_table_size, 0, 0)
    mo_data = header + table_orig_bytes + table_trans_bytes + orig_bytes + trans_bytes

    mo_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mo_path, "wb") as f:
        f.write(mo_data)


def build_all_locales():
    """编译 zh_CN 的 .mo 翻译包"""
    zh_mo = LOCALES_DIR / "zh_CN" / "LC_MESSAGES" / "messages.mo"
    create_mo_file(zh_mo, ZH_CN_TRANSLATIONS)


def init_i18n():
    """
    初始化 i18n 模块并与 st.session_state 协同管理语言切换 (源码默认语言：en_US 英语)
    """
    build_all_locales()

    if "language" not in st.session_state:
        st.session_state.language = "en_US"

    if hasattr(gettext, "_translations"):
        gettext._translations.clear()

    # 导出可直接调用的 get_text 函数
    st.session_state._ = get_text
    return get_text


def get_text(message: str) -> str:
    """
    根据 st.session_state.language 优先检索字典映射：
    - en_US: 直接返回英文原文 message
    - zh_CN: 返回 ZH_CN_TRANSLATIONS 中对应的中文翻译
    """
    lang = st.session_state.get("language", "en_US")
    if lang == "zh_CN":
        return ZH_CN_TRANSLATIONS.get(message, message)
    return message
