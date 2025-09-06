#!/usr/bin/env python3
# simple_validation.py - 간단한 모델 검증
"""
훈련된 모델의 간단한 검증
"""

import os
import torch
from ultralytics import YOLO

def validate_model():
    """모델 검증"""
    model_path = 'runs/detect/fire_smoke_detection/weights/best.pt'
    
    if not os.path.exists(model_path):
        print("❌ 훈련된 모델을 찾을 수 없습니다.")
        print("먼저 모델을 훈련하세요:")
        print("python main.py --mode train --epochs 50")
        return False
    
    try:
        print("🔍 모델 검증 시작...")
        
        # 모델 로드
        model = YOLO(model_path)
        
        # 검증 실행
        results = model.val(
            data='data.yaml',
            split='val',
            imgsz=640,
            batch=1,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        print("✅ 모델 검증 완료!")
        print(f"mAP@0.5: {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 검증 실패: {e}")
        return False

if __name__ == "__main__":
    validate_model()
