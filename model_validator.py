# model_validator.py - 모델 검증
"""
YOLOv8 화재 및 연기 감지 모델 검증 모듈
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from ultralytics import YOLO
from config import Config

class ModelValidator:
    """모델 검증 클래스"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.validation_results = None
        
    def load_model(self, model_path=None):
        """모델 로드"""
        if model_path:
            self.model_path = model_path
        
        if not self.model_path or not os.path.exists(self.model_path):
            # 기본 경로에서 최고 성능 모델 찾기
            default_path = os.path.join(Config.HOME, 'runs', 'detect', 'fire_smoke_detection', 'weights', 'best.pt')
            if os.path.exists(default_path):
                self.model_path = default_path
            else:
                print("❌ 모델 파일을 찾을 수 없습니다.")
                return False
        
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ 모델 로드 완료: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def validate(self, data_yaml='data.yaml', split='val', save_json=True):
        """모델 검증 실행"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        print(f"=== 모델 검증 시작 ({split} 데이터) ===")
        
        try:
            # 검증 실행
            self.validation_results = self.model.val(
                data=data_yaml,
                split=split,
                imgsz=640,
                batch=1,  # 검증 시 배치 크기 1로 고정
                save_json=save_json,
                save_hybrid=False,  # 하이브리드 라벨 저장 안 함
                conf=0.001,  # 낮은 신뢰도 임계값으로 모든 예측 포함
                iou=0.6,    # IoU 임계값
                max_det=300,  # 최대 감지 수
                half=False,   # FP16 사용 안 함
                device=0 if torch.cuda.is_available() else 'cpu',
                dnn=False,    # ONNX DNN 백엔드 사용 안 함
                plots=True,   # 플롯 생성
                verbose=True
            )
            
            # 검증 결과 출력
            self._print_validation_results()
            
            print("✅ 모델 검증 완료!")
            return self.validation_results
            
        except Exception as e:
            print(f"❌ 모델 검증 실패: {e}")
            return None
    
    def _print_validation_results(self):
        """검증 결과 출력"""
        if self.validation_results is None:
            return
        
        try:
            results = self.validation_results
            
            print("\n=== 검증 결과 ===")
            print(f"전체 이미지 수: {results.seen}")
            print(f"전체 라벨 수: {results.nt}")
            
            # 박스 메트릭
            if hasattr(results, 'box') and results.box:
                box_metrics = results.box
                print(f"\n📦 Detection Metrics:")
                print(f"  Precision: {box_metrics.p[0]:.4f}")
                print(f"  Recall: {box_metrics.r[0]:.4f}")
                print(f"  mAP@0.5: {box_metrics.map50:.4f}")
                print(f"  mAP@0.5:0.95: {box_metrics.map:.4f}")
                
                # 클래스별 성능
                if len(box_metrics.ap_class_index) > 0:
                    print(f"\n📊 클래스별 mAP@0.5:")
                    for i, class_idx in enumerate(box_metrics.ap_class_index):
                        if class_idx < len(Config.CLASS_NAMES):
                            class_name = Config.CLASS_NAMES[class_idx]
                            ap50 = box_metrics.ap50[i] if i < len(box_metrics.ap50) else 0
                            print(f"  {class_name:10}: {ap50:.4f}")
            
            # 속도 정보
            if hasattr(results, 'speed'):
                speed = results.speed
                print(f"\n⚡ 처리 속도:")
                print(f"  전처리: {speed['preprocess']:.1f}ms")
                print(f"  추론: {speed['inference']:.1f}ms")
                print(f"  NMS: {speed['postprocess']:.1f}ms")
                
                total_time = sum(speed.values())
                fps = 1000 / total_time if total_time > 0 else 0
                print(f"  총 처리시간: {total_time:.1f}ms")
                print(f"  FPS: {fps:.1f}")
                
        except Exception as e:
            print(f"검증 결과 출력 중 오류: {e}")
    
    def test_on_dataset(self, split='test'):
        """테스트 데이터셋으로 평가"""
        return self.validate(split=split)
    
    def confusion_matrix_analysis(self):
        """혼동 행렬 분석"""
        if self.validation_results is None:
            print("❌ 검증 결과가 없습니다. validate()를 먼저 실행하세요.")
            return None
        
        print("=== 혼동 행렬 분석 ===")
        
        try:
            # 혼동 행렬 데이터 가져오기
            if hasattr(self.validation_results, 'confusion_matrix'):
                cm = self.validation_results.confusion_matrix.matrix
            else:
                print("혼동 행렬 데이터를 찾을 수 없습니다.")
                return None
            
            # 시각화
            plt.figure(figsize=(10, 8))
            
            # 클래스 이름 (배경 클래스 포함)
            class_names = Config.CLASS_NAMES + ['background']
            
            # 혼동 행렬 히트맵
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.show()
            
            # 클래스별 성능 분석
            print("\n클래스별 성능 분석:")
            for i, class_name in enumerate(Config.CLASS_NAMES):
                if i < len(cm):
                    tp = cm[i, i]  # True Positive
                    fp = sum(cm[:, i]) - tp  # False Positive
                    fn = sum(cm[i, :]) - tp  # False Negative
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f"{class_name:10} - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")
            
            return cm
            
        except Exception as e:
            print(f"혼동 행렬 분석 실패: {e}")
            return None
    
    def precision_recall_analysis(self):
        """정밀도-재현율 분석"""
        if self.validation_results is None:
            print("❌ 검증 결과가 없습니다.")
            return None
        
        print("=== 정밀도-재현율 분석 ===")
        
        try:
            # PR 곡선 데이터 가져오기
            if hasattr(self.validation_results.box, 'p') and hasattr(self.validation_results.box, 'r'):
                precision = self.validation_results.box.p
                recall = self.validation_results.box.r
            else:
                print("PR 데이터를 찾을 수 없습니다.")
                return None
            
            # PR 곡선 그리기
            plt.figure(figsize=(12, 5))
            
            # 전체 PR 곡선
            plt.subplot(1, 2, 1)
            plt.plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            plt.legend()
            
            # F1 스코어 분포
            plt.subplot(1, 2, 2)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = f1_scores[~np.isnan(f1_scores)]  # NaN 제거
            
            plt.hist(f1_scores, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('F1 Score')
            plt.ylabel('Frequency')
            plt.title('F1 Score Distribution')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # 최적 임계값 찾기
            best_f1_idx = np.argmax(f1_scores)
            best_f1 = f1_scores[best_f1_idx]
            
            print(f"최고 F1 스코어: {best_f1:.4f}")
            print(f"해당 Precision: {precision[best_f1_idx]:.4f}")
            print(f"해당 Recall: {recall[best_f1_idx]:.4f}")
            
            return {
                'precision': precision,
                'recall': recall,
                'f1_scores': f1_scores,
                'best_f1': best_f1
            }
            
        except Exception as e:
            print(f"PR 분석 실패: {e}")
            return None
    
    def speed_benchmark(self, num_images=100):
        """속도 벤치마크"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        print(f"=== 속도 벤치마크 ({num_images}개 이미지) ===")
        
        try:
            import time
            
            # 더미 이미지 생성
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 워밍업
            for _ in range(10):
                self.model(dummy_image, verbose=False)
            
            # 실제 벤치마크
            times = []
            for i in range(num_images):
                start_time = time.time()
                results = self.model(dummy_image, verbose=False)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # ms로 변환
                
                if (i + 1) % 20 == 0:
                    print(f"진행률: {i + 1}/{num_images}")
            
            # 통계 계산
            times = np.array(times)
            
            print(f"\n⚡ 속도 벤치마크 결과:")
            print(f"평균 처리 시간: {np.mean(times):.2f}ms")
            print(f"최소 처리 시간: {np.min(times):.2f}ms")
            print(f"최대 처리 시간: {np.max(times):.2f}ms")
            print(f"표준 편차: {np.std(times):.2f}ms")
            print(f"평균 FPS: {1000/np.mean(times):.1f}")
            
            # 시간 분포 히스토그램
            plt.figure(figsize=(10, 6))
            plt.hist(times, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(times), color='red', linestyle='--', 
                       label=f'평균: {np.mean(times):.2f}ms')
            plt.xlabel('처리 시간 (ms)')
            plt.ylabel('빈도')
            plt.title('처리 시간 분포')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            return {
                'times': times,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'fps': 1000/np.mean(times)
            }
            
        except Exception as e:
            print(f"속도 벤치마크 실패: {e}")
            return None
    
    def validate_specific_classes(self, target_classes=None):
        """특정 클래스에 대한 상세 검증"""
        if target_classes is None:
            target_classes = ['Fire', 'smoke']  # 화재와 연기에 집중
        
        print(f"=== 특정 클래스 검증: {target_classes} ===")
        
        if self.validation_results is None:
            print("검증을 먼저 실행해주세요.")
            return None
        
        try:
            results_dict = {}
            
            for class_name in target_classes:
                if class_name in Config.CLASS_NAMES:
                    class_idx = Config.CLASS_NAMES.index(class_name)
                    
                    # 해당 클래스의 성능 메트릭 추출
                    if hasattr(self.validation_results.box, 'ap_class_index'):
                        class_indices = self.validation_results.box.ap_class_index
                        if class_idx in class_indices:
                            idx_pos = np.where(class_indices == class_idx)[0][0]
                            
                            ap50 = self.validation_results.box.ap50[idx_pos]
                            ap75 = self.validation_results.box.ap[idx_pos] if idx_pos < len(self.validation_results.box.ap) else 0
                            
                            results_dict[class_name] = {
                                'ap50': ap50,
                                'ap75': ap75,
                                'class_idx': class_idx
                            }
                            
                            print(f"{class_name:10} - mAP@0.5: {ap50:.4f}, mAP@0.5:0.95: {ap75:.4f}")
            
            return results_dict
            
        except Exception as e:
            print(f"특정 클래스 검증 실패: {e}")
            return None
    
    def save_validation_report(self, output_path='validation_report.json'):
        """검증 보고서 저장"""
        if self.validation_results is None:
            print("❌ 검증 결과가 없습니다.")
            return False
        
        try:
            report = {
                'model_path': self.model_path,
                'timestamp': str(np.datetime64('now')),
                'dataset_info': {
                    'classes': Config.CLASS_NAMES,
                    'num_classes': Config.NUM_CLASSES
                },
                'metrics': {}
            }
            
            # 메트릭 추가
            if hasattr(self.validation_results, 'box') and self.validation_results.box:
                box = self.validation_results.box
                report['metrics'] = {
                    'map50': float(box.map50),
                    'map75': float(box.map),
                    'precision': float(box.p[0]) if len(box.p) > 0 else 0,
                    'recall': float(box.r[0]) if len(box.r) > 0 else 0
                }
            
            # 속도 정보
            if hasattr(self.validation_results, 'speed'):
                report['speed'] = dict(self.validation_results.speed)
            
            # JSON 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 검증 보고서 저장: {output_path}")
            return True
            
        except Exception as e:
            print(f"보고서 저장 실패: {e}")
            return False

def main():
    """모델 검증 테스트"""
    # 검증기 초기화
    validator = ModelValidator()
    
    # 모델 로드
    if validator.load_model():
        print("모델 로드 완료!")
        print("다음 함수들을 사용하여 검증하세요:")
        print("- validator.validate(): 기본 검증")
        print("- validator.confusion_matrix_analysis(): 혼동 행렬")
        print("- validator.precision_recall_analysis(): PR 분석")
        print("- validator.speed_benchmark(): 속도 벤치마크")

if __name__ == "__main__":
    main()
