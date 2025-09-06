# model_trainer.py - 모델 훈련
"""
YOLOv8 화재 및 연기 감지 모델 훈련 모듈
"""

import os
import time
import torch
from pathlib import Path
from ultralytics import YOLO
from config import Config

class ModelTrainer:
    """모델 훈련 클래스"""
    
    def __init__(self, model_size='yolov8n.pt'):
        self.model_size = model_size
        self.model = None
        self.training_results = None
        self.best_model_path = None
        
    def load_model(self, pretrained=True):
        """모델 로드"""
        print(f"=== {self.model_size} 모델 로드 ===")
        
        try:
            if pretrained:
                # 사전 훈련된 모델 로드
                self.model = YOLO(self.model_size)
                print(f"✅ 사전 훈련된 {self.model_size} 모델 로드 완료")
            else:
                # 빈 모델 로드 (처음부터 훈련)
                model_config = self.model_size.replace('.pt', '.yaml')
                self.model = YOLO(model_config)
                print(f"✅ 빈 {model_config} 모델 로드 완료")
            
            # 모델 정보 출력
            print(f"모델 파라미터 수: {sum(p.numel() for p in self.model.model.parameters()):,}")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
        
        return True
    
    def train(self, 
              data_yaml='data.yaml',
              epochs=100,
              batch_size=16,
              image_size=640,
              project_name='fire_smoke_detection',
              patience=50,
              save_period=10,
              **kwargs):
        """모델 훈련"""
        
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
            return None
        
        print("=== 모델 훈련 시작 ===")
        print(f"데이터: {data_yaml}")
        print(f"에포크: {epochs}")
        print(f"배치 크기: {batch_size}")
        print(f"이미지 크기: {image_size}")
        print(f"디바이스: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # 훈련 시작 시간 기록
        start_time = time.time()
        
        try:
            # 훈련 실행
            self.training_results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=image_size,
                batch=batch_size,
                name=project_name,
                patience=patience,
                save=True,
                save_period=save_period,
                device=0 if torch.cuda.is_available() else 'cpu',
                workers=4,  # 데이터 로딩 워커 수
                optimizer='AdamW',  # 옵티마이저
                lr0=0.01,  # 초기 학습률
                lrf=0.1,  # 최종 학습률 (lr0 * lrf)
                momentum=0.937,  # SGD 모멘텀
                weight_decay=0.0005,  # 가중치 감쇠
                warmup_epochs=3,  # 워밍업 에포크
                warmup_momentum=0.8,  # 워밍업 모멘텀
                warmup_bias_lr=0.1,  # 워밍업 바이어스 학습률
                box=7.5,  # 박스 손실 가중치
                cls=0.5,  # 클래스 손실 가중치
                dfl=1.5,  # DFL 손실 가중치
                pose=12.0,  # 포즈 손실 가중치 (사용 안 함)
                kobj=2.0,  # 키포인트 객체 손실 가중치 (사용 안 함)
                label_smoothing=0.0,  # 라벨 스무딩
                nbs=64,  # 정규화 배치 크기
                hsv_h=0.015,  # HSV-Hue 증강
                hsv_s=0.7,  # HSV-Saturation 증강
                hsv_v=0.4,  # HSV-Value 증강
                degrees=0.0,  # 이미지 회전 (+/- deg)
                translate=0.1,  # 이미지 이동 (+/- fraction)
                scale=0.5,  # 이미지 스케일 (+/- gain)
                shear=0.0,  # 이미지 기울기 (+/- deg)
                perspective=0.0,  # 이미지 원근 (+/- fraction), range 0-0.001
                flipud=0.0,  # 이미지 상하 뒤집기 (확률)
                fliplr=0.5,  # 이미지 좌우 뒤집기 (확률)
                mosaic=1.0,  # 이미지 모자이크 (확률)
                mixup=0.0,  # 이미지 믹스업 (확률)
                copy_paste=0.0,  # 세그먼트 복사-붙여넣기 (확률)
                **kwargs
            )
            
            # 훈련 시간 계산
            training_time = time.time() - start_time
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            seconds = int(training_time % 60)
            
            print(f"\n✅ 훈련 완료! 소요 시간: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # 최고 성능 모델 경로 저장
            self.best_model_path = os.path.join(
                Config.HOME, 'runs', 'detect', project_name, 'weights', 'best.pt'
            )
            
            if os.path.exists(self.best_model_path):
                print(f"최고 성능 모델 저장: {self.best_model_path}")
            
            # 훈련 결과 요약
            self._print_training_summary()
            
        except Exception as e:
            print(f"❌ 훈련 중 오류 발생: {e}")
            return None
        
        return self.training_results
    
    def _print_training_summary(self):
        """훈련 결과 요약 출력"""
        if self.training_results is None:
            return
        
        print("\n=== 훈련 결과 요약 ===")
        try:
            # 최종 메트릭 출력
            metrics = self.training_results.results_dict
            if metrics:
                print(f"최종 mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
                print(f"최종 mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
                print(f"최종 Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
                print(f"최종 Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
        except:
            print("메트릭 정보를 가져올 수 없습니다.")
    
    def resume_training(self, checkpoint_path, epochs=None):
        """체크포인트에서 훈련 재개"""
        print(f"=== 훈련 재개: {checkpoint_path} ===")
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
            return None
        
        try:
            # 체크포인트에서 모델 로드
            self.model = YOLO(checkpoint_path)
            
            # 훈련 재개
            resume_results = self.model.train(
                resume=True,
                epochs=epochs  # None이면 원래 설정된 에포크까지 계속
            )
            
            print("✅ 훈련 재개 완료!")
            return resume_results
            
        except Exception as e:
            print(f"❌ 훈련 재개 실패: {e}")
            return None
    
    def hyperparameter_tuning(self, space=None, iterations=100):
        """하이퍼파라미터 튜닝"""
        print("=== 하이퍼파라미터 튜닝 시작 ===")
        
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        try:
            # 기본 하이퍼파라미터 검색 공간
            if space is None:
                space = {
                    'lr0': (1e-5, 1e-1),  # 초기 학습률
                    'lrf': (0.01, 1.0),   # 최종 학습률 비율
                    'momentum': (0.6, 0.98),  # 모멘텀
                    'weight_decay': (0.0, 0.001),  # 가중치 감쇠
                    'warmup_epochs': (0.0, 5.0),  # 워밍업 에포크
                    'box': (0.02, 0.2),   # 박스 손실 가중치
                    'cls': (0.2, 4.0),    # 클래스 손실 가중치
                    'dfl': (0.4, 6.0),    # DFL 손실 가중치
                    'hsv_h': (0.0, 0.1),  # HSV-Hue 증강
                    'hsv_s': (0.0, 0.9),  # HSV-Saturation 증강
                    'hsv_v': (0.0, 0.9),  # HSV-Value 증강
                    'degrees': (0.0, 45.0),  # 회전 각도
                    'translate': (0.0, 0.9),  # 이동
                    'scale': (0.0, 0.9),  # 스케일
                    'fliplr': (0.0, 1.0)  # 좌우 뒤집기 확률
                }
            
            # 하이퍼파라미터 튜닝 실행
            tuning_results = self.model.tune(
                data='data.yaml',
                space=space,
                iterations=iterations,
                optimizer='AdamW',
                epochs=50,  # 튜닝용 짧은 에포크
                imgsz=640,
                device=0 if torch.cuda.is_available() else 'cpu'
            )
            
            print("✅ 하이퍼파라미터 튜닝 완료!")
            return tuning_results
            
        except Exception as e:
            print(f"❌ 하이퍼파라미터 튜닝 실패: {e}")
            return None
    
    def export_model(self, format='onnx', **kwargs):
        """훈련된 모델 내보내기"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            model = YOLO(self.best_model_path)
        elif self.model:
            model = self.model
        else:
            print("❌ 내보낼 모델이 없습니다.")
            return None
        
        print(f"=== 모델을 {format} 형식으로 내보내기 ===")
        
        try:
            export_path = model.export(
                format=format,
                imgsz=640,
                **kwargs
            )
            print(f"✅ 모델 내보내기 완료: {export_path}")
            return export_path
            
        except Exception as e:
            print(f"❌ 모델 내보내기 실패: {e}")
            return None
    
    def get_model_info(self):
        """모델 정보 조회"""
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return None
        
        print("=== 모델 정보 ===")
        try:
            # 모델 구조 정보
            total_params = sum(p.numel() for p in self.model.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            
            print(f"총 파라미터 수: {total_params:,}")
            print(f"훈련 가능한 파라미터 수: {trainable_params:,}")
            print(f"모델 크기: {self.model_size}")
            
            # 레이어 정보
            print(f"레이어 수: {len(list(self.model.model.modules()))}")
            
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size': self.model_size
            }
            
        except Exception as e:
            print(f"모델 정보 조회 실패: {e}")
            return None
    
    def benchmark_model(self):
        """모델 성능 벤치마크"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            model = YOLO(self.best_model_path)
        elif self.model:
            model = self.model
        else:
            print("❌ 벤치마크할 모델이 없습니다.")
            return None
        
        print("=== 모델 성능 벤치마크 ===")
        
        try:
            # 벤치마크 실행
            benchmark_results = model.benchmark(
                data='coco128.yaml',  # 표준 벤치마크 데이터
                imgsz=640,
                half=False,  # FP16 사용 안 함
                device=0 if torch.cuda.is_available() else 'cpu'
            )
            
            print("✅ 벤치마크 완료!")
            return benchmark_results
            
        except Exception as e:
            print(f"❌ 벤치마크 실패: {e}")
            return None

class TrainingCallback:
    """훈련 콜백 클래스"""
    
    def __init__(self):
        self.best_fitness = 0
        self.training_log = []
    
    def on_epoch_end(self, trainer):
        """에포크 종료 시 호출"""
        epoch = trainer.epoch
        fitness = trainer.fitness
        
        # 로그 저장
        log_entry = {
            'epoch': epoch,
            'fitness': fitness,
            'lr': trainer.optimizer.param_groups[0]['lr']
        }
        self.training_log.append(log_entry)
        
        # 최고 성능 업데이트
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            print(f"🎉 새로운 최고 성능! Epoch {epoch}, Fitness: {fitness:.4f}")
    
    def on_train_end(self, trainer):
        """훈련 종료 시 호출"""
        print(f"훈련 완료! 최고 Fitness: {self.best_fitness:.4f}")

def main():
    """모델 훈련 테스트"""
    # 훈련기 초기화
    trainer = ModelTrainer('yolov8n.pt')
    
    # 모델 로드
    if trainer.load_model():
        # 모델 정보 출력
        trainer.get_model_info()
        
        # 훈련 실행 (예시)
        print("\n훈련을 시작하려면 trainer.train()을 호출하세요.")
        print("예: trainer.train(epochs=50, batch_size=8)")

if __name__ == "__main__":
    main()