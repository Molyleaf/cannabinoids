import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import streamlit as st
from app.i18n import init_i18n

st.session_state.language = "en_US"
_ = init_i18n()

test_strings = [
    "感谢您支持模型优化！",
    "信息提交成功！",
    "关于数据共享与致谢",
    "数据共享将直接帮助提升模型的检测准确率：",
    "模型更新通知：模型发布新版本时，您将优先收到通知邮件，了解改进详情。",
    "贡献致谢：在模型升级公告中，系统将以您提供的称呼表达致谢。",
]

for s in test_strings:
    translated = _(s)
    print(f"Original: '{s}'")
    print(f"Result:   '{translated}'")
    print("-" * 50)
