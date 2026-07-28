import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import streamlit as st
from app.i18n import init_i18n

st.session_state.language = "en_US"
_ = init_i18n()

print("Current language in session_state:", st.session_state.language)

items = [
    "同意共享谱图数据，帮助改进模型性能",
    "感谢您支持模型优化！",
    "要怎么称呼您",
    "电子邮箱",
    "提交信息",
    "信息提交成功！",
    "关于数据共享与致谢",
    "数据共享将直接帮助提升模型的检测准确率：",
    "模型更新通知：模型发布新版本时，您将优先收到通知邮件，了解改进详情。",
    "贡献致谢：在模型升级公告中，系统将以您提供的称呼表达致谢。",
]

for item in items:
    res = _(item)
    print(f"[{item}] => [{res}]")
