"""
utils/notify.py
===============
역할
----
이상 탐지 결과를 Slack으로 알림을 보내는 유틸리티 파일입니다.

왜 Slack Webhook인가?
---------------------
- 이메일보다 구현이 훨씬 간단 (HTTP POST 요청 하나)
- 실무에서 모니터링 알림으로 많이 사용
- 무료로 사용 가능

Slack Webhook 설정 방법
-----------------------
1. https://api.slack.com/apps 접속
2. Create New App → From scratch
3. Incoming Webhooks → Activate
4. Add New Webhook to Workspace → 채널 선택
5. Webhook URL 복사 → .env 파일의 SLACK_WEBHOOK_URL에 붙여넣기

알림이 전송되는 조건
--------------------
config.yaml의 slack.alert_threshold 값 이상일 때만 전송합니다.
  예: alert_threshold=0.7이면 score >= 0.7일 때만 Slack 알림

알림 메시지 예시
----------------
  🚨 [PRP] 이상 탐지 알림
  ─────────────────────────
  📁 이미지: bottle/test/broken_large/000.png
  📊 Anomaly Score: 0.8234
  🔴 판정: 이상 (threshold: 0.7)
  🕐 시각: 2026-06-01 09:47:00
  🖼️ 히트맵: results/heatmaps/000_heatmap.png
"""

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
    """
    이상 탐지 결과를 Slack으로 전송합니다.

    Args:
        image_path  : 추론한 이미지 경로
        score       : Anomaly Score
        is_anomaly  : 이상 여부
        heatmap_path: 생성된 히트맵 경로
        threshold   : 판정 기준값 (config.yaml의 model.threshold)
        webhook_url : Slack Webhook URL (.env의 SLACK_WEBHOOK_URL)

    Returns:
        전송 성공 여부 (bool)
    """
    # Webhook URL이 없으면 .env에서 가져오기
    if not webhook_url:
        webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL이 설정되지 않았습니다. .env 파일을 확인하세요.")
        return False

    # 메시지 구성
    status_emoji = "🔴" if is_anomaly else "🟢"
    status_text  = "이상" if is_anomaly else "정상"
    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = {
        "text": f"🚨 *[PRP] 이상 탐지 알림*",
        "attachments": [
            {
                # 이상이면 빨간색, 정상이면 초록색 사이드바
                "color": "#FF0000" if is_anomaly else "#00FF00",
                "fields": [
                    {
                        "title": "📁 이미지",
                        "value": image_path,
                        "short": False,
                    },
                    {
                        "title": "📊 Anomaly Score",
                        "value": f"`{score:.6f}`",
                        "short": True,
                    },
                    {
                        "title": "판정",
                        "value": f"{status_emoji} {status_text} (threshold: {threshold})",
                        "short": True,
                    },
                    {
                        "title": "🕐 시각",
                        "value": timestamp,
                        "short": True,
                    },
                    {
                        "title": "🖼️ 히트맵 경로",
                        "value": heatmap_path,
                        "short": False,
                    },
                ],
            }
        ],
    }

    # HTTP POST 전송
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
    """
    알림을 보내야 하는지 판단합니다.

    model.threshold (탐지 기준)와 slack.alert_threshold (알림 기준)를 분리한 이유:
    - 탐지는 민감하게 (낮은 threshold) 하고
    - 알림은 확실한 이상만 (높은 threshold) 보낼 수 있도록

    예: model.threshold=0.5, slack.alert_threshold=0.7
        → score 0.6: 이상으로 탐지는 하지만 Slack 알림은 안 보냄
        → score 0.8: 이상으로 탐지 + Slack 알림 전송

    Args:
        score          : Anomaly Score
        alert_threshold: Slack 알림 기준 (config.yaml의 slack.alert_threshold)

    Returns:
        알림 전송 여부 (bool)
    """
    return score >= alert_threshold
