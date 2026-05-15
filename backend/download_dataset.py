#!/usr/bin/env python3
# download_dataset.py - 간단한 데이터셋 다운로드
"""
Roboflow에서 화재 데이터셋을 다운로드합니다.
사전 조건: pip install roboflow (또는 pip install -r requirements.txt)
"""

import roboflow


def download_fire_dataset(api_key):
    """화재 데이터셋 다운로드"""
    print("🔥 화재 데이터셋 다운로드 시작...")

    try:
        rf = roboflow.Roboflow(api_key=api_key)
        project = rf.workspace("custom-thxhn").project("fire-wrpgm")
        dataset = project.version(8).download("yolov8")
        print(f"✅ 다운로드 완료: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return None


if __name__ == "__main__":
    api_key = input("Roboflow API 키를 입력하세요: ").strip()
    if api_key:
        download_fire_dataset(api_key)
