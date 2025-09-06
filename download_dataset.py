#!/usr/bin/env python3
# download_dataset.py - 간단한 데이터셋 다운로드
import os

def download_fire_dataset(api_key):
    """화재 데이터셋 다운로드"""
    try:
        import subprocess
        import sys
        
        # roboflow 설치
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'roboflow'])
        
        import roboflow
        
        print("🔥 화재 데이터셋 다운로드 시작...")
        
        rf = roboflow.Roboflow(api_key=api_key)
        project = rf.workspace("custom-thxhn").project("fire-wrpgm")
        dataset = project.version(8).download("yolov8")
        
        print(f"✅ 다운로드 완료: {dataset.location}")
        return dataset.location
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return None

if __name__ == "__main__":
    api_key = input("Roboflow API 키를 입력하세요: ")
    if api_key:
        download_fire_dataset(api_key)
