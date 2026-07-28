import json
import os
from datetime import datetime
from pathlib import Path

# 队列与隐私数据存取目录 (必须在 app 内部)
QUEUE_DIR = Path(__file__).resolve().parent / "data_queue"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

UNCERTAIN_QUEUE_FILE = QUEUE_DIR / "uncertain_samples.json"
CONTRIBUTIONS_FILE = QUEUE_DIR / "authorized_contributions.json"


def record_sample_if_authorized(
    file_name: str,
    peaks: list,
    model_type: str,
    probability: float,
    predicted_result: str,
    user_consent: bool,
    user_name: str = "",
    user_email: str = ""
) -> dict:
    """
    第1阶段：知情同意与隐私检查。若未授权，绝对不执行数据留存或上传。
    第2阶段：在获得授权前提下，判定预测概率是否落在 [0.3, 0.7] 不确定区间。
    若是，自动保存至本地待标注队列，并触发后台人工复核标志。
    """
    # 未授权状态：直接返回，不做任何留存
    if not user_consent:
        return {
            "authorized": False,
            "status": "未授权数据共享。系统仅输出鉴定结果，未执行任何数据留存或上传操作。",
            "is_uncertain": False,
            "review_flag": False
        }

    # 已授权状态：记录贡献
    contribution_record = {
        "timestamp": datetime.now().isoformat(),
        "file_name": file_name,
        "user_name": user_name.strip() if user_name else "匿名贡献者",
        "user_email": user_email.strip() if user_email else "",
        "model_type": model_type,
        "probability": probability,
        "predicted_result": predicted_result
    }
    _append_json_record(CONTRIBUTIONS_FILE, contribution_record)

    # 第2阶段：认知边界不确定区间判断 [0.3, 0.7]
    is_uncertain = 0.3 <= probability <= 0.7
    review_flag = False

    if is_uncertain:
        review_flag = True
        queue_item = {
            "id": f"UNC_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "timestamp": datetime.now().isoformat(),
            "file_name": file_name,
            "model_type": model_type,
            "probability": round(probability, 4),
            "predicted_result": predicted_result,
            "user_name": user_name.strip() if user_name else "匿名贡献者",
            "user_email": user_email.strip() if user_email else "",
            "status": "待人工复核 (Pending Manual Review)",
            "peaks_count": len(peaks),
            "peaks": peaks
        }
        _append_json_record(UNCERTAIN_QUEUE_FILE, queue_item)
        status_msg = "已获得授权！样本概率落在不确定区间 [0.3, 0.7]，处于认知边界，已自动存入本地待标注队列并触发后台人工复核请求。"
    else:
        status_msg = "已获得授权！样本处于高置信度区间，直接输出鉴定结论。数据已记录归档用于未来模型升级。"

    return {
        "authorized": True,
        "status": status_msg,
        "is_uncertain": is_uncertain,
        "review_flag": review_flag
    }


def _append_json_record(filepath: Path, record: dict):
    records = []
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []
    records.append(record)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def get_queue_stats() -> dict:
    """获取待复核队列与贡献统计"""
    total_uncertain = 0
    total_contributions = 0
    if UNCERTAIN_QUEUE_FILE.exists():
        try:
            with open(UNCERTAIN_QUEUE_FILE, "r", encoding="utf-8") as f:
                total_uncertain = len(json.load(f))
        except Exception:
            pass
    if CONTRIBUTIONS_FILE.exists():
        try:
            with open(CONTRIBUTIONS_FILE, "r", encoding="utf-8") as f:
                total_contributions = len(json.load(f))
        except Exception:
            pass
    return {
        "total_uncertain": total_uncertain,
        "total_contributions": total_contributions
    }
