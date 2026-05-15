# inference_engine.py - 추론 엔진
"""
YOLOv8 화재 및 연기 감지 추론 엔진 모듈
"""

import os
import cv2
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics import YOLO
from config import Config

logger = logging.getLogger(__name__)

class InferenceEngine:
    """추론 엔진 클래스"""
    
    def __init__(self, model_path=None, confidence=0.5, iou_threshold=0.45):
        self.model_path = model_path
        self.model = None
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.class_names = Config.CLASS_NAMES
        self.colors = self._generate_colors()
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
    def _generate_colors(self):
        """클래스별 색상 생성"""
        colors = {}
        color_palette = [
            (255, 0, 0),    # 빨간색 - Fire
            (128, 128, 128) # 회색 - smoke
        ]
        
        for i, class_name in enumerate(self.class_names):
            if i < len(color_palette):
                colors[class_name] = color_palette[i]
            else:
                # 랜덤 색상 생성
                colors[class_name] = tuple(np.random.randint(0, 255, 3).tolist())
        
        return colors
    
    def load_model(self, model_path=None):
        """모델 로드"""
        if model_path:
            self.model_path = model_path
        
        if not self.model_path or not os.path.exists(self.model_path):
            # 기본 경로에서 모델 찾기
            default_paths = [
                os.path.join(Config.HOME, 'runs', 'detect', 'fire_smoke_detection', 'weights', 'best.pt'),
                os.path.join(Config.HOME, 'runs', 'detect', 'train', 'weights', 'best.pt'),
                'yolov8n.pt'  # 사전 훈련된 모델 폴백
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    self.model_path = path
                    break
            else:
                logger.info("❌ 모델 파일을 찾을 수 없습니다.")
                return False
        
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"✅ 모델 로드 완료: {self.model_path}")

            if torch.cuda.is_available():
                logger.info(f"🚀 GPU 가속 사용 (device={self.device})")
            else:
                logger.info("ℹ️ CPU 추론 모드")

            return True
        except Exception as e:
            logger.info(f"❌ 모델 로드 실패: {e}")
            return False
    
    def predict_image(self, image_path, save_result=True, show_result=True):
        """단일 이미지 추론"""
        if self.model is None:
            logger.info("❌ 모델이 로드되지 않았습니다.")
            return None
        
        if not os.path.exists(image_path):
            logger.info(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
            return None
        
        logger.info(f"=== 이미지 추론: {os.path.basename(image_path)} ===")
        
        try:
            # 추론 실행
            results = self.model.predict(
                source=image_path,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.device,
                save=save_result,
                show_labels=True,
                show_conf=True,
                line_width=2
            )
            
            # 결과 분석
            result = results[0]
            detections = self._parse_detections(result)
            
            # 결과 출력
            self._print_detection_results(detections)
            
            # 시각화
            if show_result:
                self._visualize_result(result, image_path)
            
            return {
                'detections': detections,
                'raw_results': results,
                'image_path': image_path
            }
            
        except Exception as e:
            logger.info(f"❌ 추론 실패: {e}")
            return None
    
    def predict_batch(self, image_folder, output_folder=None):
        """배치 이미지 추론"""
        if self.model is None:
            logger.info("❌ 모델이 로드되지 않았습니다.")
            return None
        
        logger.info(f"=== 배치 추론: {image_folder} ===")
        
        # 이미지 파일 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([
                os.path.join(image_folder, f) for f in os.listdir(image_folder)
                if f.lower().endswith(ext.lower())
            ])
        
        if not image_files:
            logger.info("❌ 처리할 이미지가 없습니다.")
            return None
        
        logger.info(f"총 {len(image_files)}개 이미지 처리 예정")
        
        try:
            # 배치 추론 실행
            results = self.model.predict(
                source=image_folder,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.device,
                save=True,
                project=output_folder or 'batch_inference',
                exist_ok=True
            )
            
            # 결과 분석
            batch_results = []
            for i, result in enumerate(results):
                detections = self._parse_detections(result)
                batch_results.append({
                    'image_path': image_files[i] if i < len(image_files) else f"image_{i}",
                    'detections': detections
                })
            
            # 배치 결과 요약
            self._print_batch_summary(batch_results)
            
            return batch_results
            
        except Exception as e:
            logger.info(f"❌ 배치 추론 실패: {e}")
            return None
    
    def predict_video(self, video_path, output_path=None, show_live=False,
                      keep_detection_history=False):
        """비디오 추론 (ultralytics stream 모드)

        keep_detection_history=False(기본)면 프레임별 감지 결과는 카운트만 누적하고
        메모리에 보관하지 않습니다. 긴 비디오에서 OOM을 방지합니다.
        """
        if self.model is None:
            logger.info("❌ 모델이 로드되지 않았습니다.")
            return None

        logger.info(f"=== 비디오 추론: {os.path.basename(video_path)} ===")

        out = None
        try:
            # 메타데이터 확보를 위한 임시 캡처
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.info("❌ 비디오를 열 수 없습니다.")
                return None

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            logger.info(f"비디오 정보: {width}x{height}, {fps}FPS, {total_frames}프레임")

            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            detections_total = 0
            detection_frames = 0
            detection_history = [] if keep_detection_history else None

            # stream=True: 결과를 generator로 yield → 메모리 일정
            stream = self.model.predict(
                source=video_path,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.device,
                stream=True,
                verbose=False,
            )

            for result in stream:
                frame_count += 1
                annotated_frame = result.plot()
                detections = self._parse_detections(result)

                if detections:
                    detection_frames += 1
                    detections_total += len(detections)
                    if detection_history is not None:
                        detection_history.append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'detections': detections,
                        })

                if out is not None:
                    out.write(annotated_frame)

                if show_live:
                    cv2.imshow('Fire & Smoke Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if total_frames > 0 and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"진행률: {progress:.1f}% ({frame_count}/{total_frames})")

            logger.info(f"✅ 비디오 처리 완료! 감지 프레임 {detection_frames}개 / "
                  f"총 객체 {detections_total}개")

            return {
                'total_frames': total_frames,
                'processed_frames': frame_count,
                'detection_frames': detection_frames,
                'detections_total': detections_total,
                'detection_history': detection_history,
                'output_path': output_path,
            }

        except Exception as e:
            logger.info(f"❌ 비디오 추론 실패: {e}")
            return None
        finally:
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
    
    def real_time_detection(self, camera_index=0, save_video=False, output_path='real_time_detection.mp4'):
        """실시간 웹캠 감지"""
        if self.model is None:
            logger.info("❌ 모델이 로드되지 않았습니다.")
            return None
        
        logger.info("=== 실시간 화재/연기 감지 시작 ===")
        logger.info("종료하려면 'q' 키를 누르세요.")
        
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                logger.info("❌ 카메라를 열 수 없습니다.")
                return None
            
            # 카메라 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 비디오 저장 설정
            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
            
            detection_count = 0
            fps_counter = 0
            start_time = time.time()
            elapsed_time = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("프레임을 읽을 수 없습니다.")
                    break

                fps_counter += 1

                # 추론 실행
                results = self.model.predict(
                    source=frame,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False
                )
                
                # 결과 그리기
                annotated_frame = results[0].plot()
                
                # FPS 계산 및 표시
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = fps_counter / elapsed_time
                    cv2.putText(annotated_frame, f'FPS: {fps:.1f}', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 감지 수 표시
                detections = self._parse_detections(results[0])
                if detections:
                    detection_count += len(detections)
                    
                    # 알림 메시지
                    for detection in detections:
                        class_name = detection['class']
                        if class_name in ['Fire', 'smoke']:
                            cv2.putText(annotated_frame, f'ALERT: {class_name} DETECTED!', 
                                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # 화면 표시
                cv2.imshow('Fire & Smoke Detection (Press Q to quit)', annotated_frame)
                
                # 비디오 저장
                if save_video:
                    out.write(annotated_frame)
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 정리
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ 실시간 감지 종료. 총 {detection_count}개 객체 감지")

            return {
                'total_detections': detection_count,
                'duration': elapsed_time,
                'fps': fps_counter / elapsed_time if elapsed_time > 0 else 0
            }
            
        except Exception as e:
            logger.info(f"❌ 실시간 감지 실패: {e}")
            return None
    
    def _parse_detections(self, result):
        """감지 결과 파싱"""
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                class_id = int(classes[i])
                if class_id < len(self.class_names):
                    detection = {
                        'class': self.class_names[class_id],
                        'class_id': class_id,
                        'confidence': float(confidences[i]),
                        'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                        'center': [
                            (boxes[i][0] + boxes[i][2]) / 2,
                            (boxes[i][1] + boxes[i][3]) / 2
                        ]
                    }
                    detections.append(detection)
        
        return detections
    
    def _print_detection_results(self, detections):
        """감지 결과 출력"""
        if not detections:
            logger.info("❌ 감지된 객체 없음")
            return
        
        logger.info(f"✅ {len(detections)}개 객체 감지:")
        
        class_counts = {}
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            if class_name not in class_counts:
                class_counts[class_name] = []
            class_counts[class_name].append(confidence)
            
            logger.info(f"  - {class_name}: {confidence:.3f}")
        
        # 클래스별 요약
        logger.info("\n📊 클래스별 요약:")
        for class_name, confidences in class_counts.items():
            avg_conf = np.mean(confidences)
            max_conf = np.max(confidences)
            logger.info(f"  {class_name}: {len(confidences)}개 (평균: {avg_conf:.3f}, 최대: {max_conf:.3f})")
    
    def _print_batch_summary(self, batch_results):
        """배치 결과 요약"""
        total_images = len(batch_results)
        images_with_detections = sum(1 for r in batch_results if r['detections'])
        total_detections = sum(len(r['detections']) for r in batch_results)
        
        logger.info(f"\n=== 배치 처리 요약 ===")
        logger.info(f"총 이미지: {total_images}개")
        logger.info(f"감지된 이미지: {images_with_detections}개")
        logger.info(f"총 감지 객체: {total_detections}개")
        logger.info(f"감지율: {images_with_detections/total_images*100:.1f}%")
        
        # 클래스별 통계
        class_stats = {}
        for result in batch_results:
            for detection in result['detections']:
                class_name = detection['class']
                if class_name not in class_stats:
                    class_stats[class_name] = []
                class_stats[class_name].append(detection['confidence'])
        
        if class_stats:
            logger.info(f"\n📊 클래스별 통계:")
            for class_name, confidences in class_stats.items():
                count = len(confidences)
                avg_conf = np.mean(confidences)
                logger.info(f"  {class_name}: {count}개 (평균 신뢰도: {avg_conf:.3f})")
    
    def _visualize_result(self, result, image_path):
        """결과 시각화"""
        try:
            # 원본 이미지 로드
            image = Image.open(image_path)
            
            # 결과 이미지 생성
            result_image = result.plot()
            result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            
            # 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # 원본 이미지
            ax1.imshow(image)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # 결과 이미지
            ax2.imshow(result_image_pil)
            ax2.set_title('Detection Results')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.info(f"시각화 실패: {e}")
    
    def set_confidence_threshold(self, confidence):
        """신뢰도 임계값 설정"""
        self.confidence = confidence
        logger.info(f"신뢰도 임계값 변경: {confidence}")
    
    def set_iou_threshold(self, iou_threshold):
        """IoU 임계값 설정"""
        self.iou_threshold = iou_threshold
        logger.info(f"IoU 임계값 변경: {iou_threshold}")
    
    def get_model_info(self):
        """모델 정보 조회"""
        if self.model is None:
            logger.info("모델이 로드되지 않았습니다.")
            return None
        
        info = {
            'model_path': self.model_path,
            'confidence_threshold': self.confidence,
            'iou_threshold': self.iou_threshold,
            'class_names': self.class_names,
            'device': next(self.model.model.parameters()).device if self.model else 'unknown'
        }
        
        logger.info("=== 모델 정보 ===")
        for key, value in info.items():
            logger.info(f"{key}: {value}")
        
        return info

def main():
    """추론 엔진 테스트"""
    # 추론 엔진 초기화
    engine = InferenceEngine(confidence=0.5)
    
    # 모델 로드
    if engine.load_model():
        logger.info("추론 엔진 준비 완료!")
        logger.info("\n사용 가능한 기능:")
        logger.info("- engine.predict_image('이미지경로'): 단일 이미지 추론")
        logger.info("- engine.predict_batch('폴더경로'): 배치 추론")
        logger.info("- engine.predict_video('비디오경로'): 비디오 추론")
        logger.info("- engine.real_time_detection(): 실시간 감지")
        
        # 모델 정보 출력
        engine.get_model_info()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()