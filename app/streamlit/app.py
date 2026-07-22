import sys
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 确保 app 所在目录可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import run_pipeline

st.set_page_config(
    page_title="合成大麻素质谱测样风险评估系统",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 合成大麻素质谱测样与风险评估平台")
st.markdown("快速导入质谱数据 (`.msp` / `.mgf`)，在线清洗、已知库比对与深度学习风险推理。")

st.sidebar.header("⚙️ 测样参数配置")
min_similarity = st.sidebar.slider(
    "已知库熵检索匹配阈值",
    min_value=0.50,
    max_value=0.99,
    value=0.75,
    step=0.01,
    help="用于判断质谱是否在已知阳性库内的最小熵相似度阈值"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📋 使用说明
1. 选择单个 `.msp` 或 `.mgf` 格式的质谱文件；
2. 系统自动调用 `ms_entropy` 洗净质谱；
3. 执行已建 FlashEntropySearch 索引已知库快速匹配；
4. 运行 `Safetensors` 模型推断阳性/高风险概率。
""")

uploaded_file = st.file_uploader(
    "上传单样质谱文件 (.msp 或 .mgf)",
    type=["msp", "mgf"],
    help="支持 NIST MSP 或 MGF 格式文本质谱文件"
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    with st.spinner(f"正在对 {filename} 进行测样分析..."):
        try:
            result = run_pipeline(
                file_bytes=file_bytes,
                filename=filename,
                min_similarity=min_similarity
            )
            
            st.success("🎉 质谱解析与模型推断完成！")

            # 结果摘要卡片
            col1, col2, col3, col4 = st.columns(4)

            model_inf = result["model_inference"]
            known_match = result["known_library_match"]

            with col1:
                st.metric("测样文件名", result["filename"])

            with col2:
                st.metric("清洗后质谱峰数", result["num_cleaned_peaks"])

            with col3:
                if known_match["is_matched"]:
                    st.metric("已知库匹配状态", "已命中已知库 🎯")
                else:
                    st.metric("已知库匹配状态", "未命中已知库 ❓")

            with col4:
                prob_pct = model_inf["risk_percentage"]
                if model_inf["is_high_risk"]:
                    st.metric("合成大麻素风险预测", prob_pct, delta="高风险 High Risk", delta_color="inverse")
                else:
                    st.metric("合成大麻素风险预测", prob_pct, delta="低风险 Low Risk", delta_color="normal")

            st.markdown("---")

            # 详细信息选项卡
            tab1, tab2, tab3 = st.tabs(["📊 质谱可视化", "🧬 比对与风险详细报告", "📄 原始 Peaks 数据"])

            with tab1:
                st.subheader("质谱峰图 (m/z vs. Intensity)")
                peaks = np.array(result["peaks"])
                if len(peaks) > 0:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    mzs = peaks[:, 0]
                    intensities = peaks[:, 1]
                    # 归一化强度用于作图
                    max_int = max(intensities) if len(intensities) > 0 else 1.0
                    norm_int = (intensities / max_int) * 100.0

                    ax.vlines(x=mzs, ymin=0, ymax=norm_int, color='#1f77b4', linewidth=1.5)
                    ax.scatter(mzs, norm_int, color='#1f77b4', s=10)
                    ax.set_xlabel("m/z")
                    ax.set_ylabel("Relative Intensity (%)")
                    ax.set_title(f"Cleaned Spectrum Profile: {filename}")
                    ax.grid(True, linestyle="--", alpha=0.5)

                    st.pyplot(fig)

            with tab2:
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("1. 已知库熵检索结果")
                    st.write(f"- **是否命中已知库**: {'是' if known_match['is_matched'] else '否'}")
                    st.write(f"- **匹配阈值 (Similarity Threshold)**: {known_match['min_similarity_threshold']}")
                    if known_match['matched_smiles']:
                        st.write(f"- **匹配 SMILES 结构**: `{known_match['matched_smiles']}`")

                with col_right:
                    st.subheader("2. 深度学习 Safetensors 模型推断")
                    st.write(f"- **高风险概率值**: `{model_inf['risk_probability']}` ({model_inf['risk_percentage']})")
                    st.write(f"- **风险等级判定**: **{model_inf['risk_level']}**")

            with tab3:
                st.subheader("清洗后 Peak 数据明细")
                st.dataframe(peaks, column_config={"0": "m/z", "1": "Intensity"}, use_container_width=True)

        except Exception as e:
            st.error(f"❌ 测样过程发生错误: {str(e)}")
