import os
import httpx
from datetime import datetime
from loguru import logger


def send_slack_alert(
    image_path: str,
    score: float,
    is_anomaly: bool,
    heatmap_path: str,
    threshold: float,
    webhook_url: str = None,
) -> bool:
    if not webhook_url:
        webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL이 설정되지 않았습니다.")
        return False

    status_emoji = "🔴" if is_anomaly else "🟢"
    status_text  = "이상" if is_anomaly else "정상"
    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = {
        "text": "🚨 *[PRP] 이상 탐지 알림*",
        "attachments": [
            {
                "color": "#FF0000" if is_anomaly else "#00FF00",
                "fields": [
                    {"title": "📁 이미지",        "value": image_path,                                    "short": False},
                    {"title": "📊 Anomaly Score", "value": f"`{score:.6f}`",                              "short": True},
                    {"title": "판정",              "value": f"{status_emoji} {status_text} (threshold: {threshold})", "short": True},
                    {"title": "🕐 시각",           "value": timestamp,                                    "short": True},
                    {"title": "🖼️ 히트맵 경로",   "value": heatmap_path,                                 "short": False},
                ],
            }
        ],
    }

    try:
        response = httpx.post(webhook_url, json=message, timeout=10)
        response.raise_for_status()
        logger.info(f"Slack 알림 전송 완료 | score={score:.6f} | {status_text}")
        return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Slack 알림 실패 (HTTP {e.response.status_code}): {e}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Slack 알림 실패 (네트워크 오류): {e}")
        return False


def should_alert(score: float, alert_threshold: float) -> bool:
    return score >= alert_threshold