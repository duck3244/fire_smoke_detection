# visualization_utils.py - 시각화 유틸리티
"""
YOLOv8 화재 및 연기 감지 프로젝트 시각화 유틸리티
"""

import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import json
from pathlib import Path
from config import Config

class VisualizationUtils:
    """시각화 유틸리티 클래스"""
    
    def __init__(self):
        self.colors = {
            'Fire': '#FF4444',      # 빨간색
            'default': '#44FF44',   # 초록색  
            'smoke': '#888888'      # 회색
        }
        plt.style.use('seaborn-v0_8')
    
    def plot_training_results(self, results_path=None):
        """훈련 결과 시각화"""
        if results_path is None:
            results_path = os.path.join(Config.HOME, 'runs', 'detect', 'fire_smoke_detection')
        
        if not os.path.exists(results_path):
            print(f"❌ 결과 디렉토리를 찾을 수 없습니다: {results_path}")
            return
        
        print("=== 훈련 결과 시각화 ===")
        
        # 1. Results 이미지 표시
        results_img_path = os.path.join(results_path, 'results.png')
        if os.path.exists(results_img_path):
            img = Image.open(results_img_path)
            plt.figure(figsize=(15, 10))
            plt.imshow(img)
            plt.title('Training Results Overview', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # 2. Confusion Matrix
        confusion_matrix_path = os.path.join(results_path, 'confusion_matrix.png')
        if os.path.exists(confusion_matrix_path):
            img = Image.open(confusion_matrix_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # 3. PR Curve
        pr_curve_path = os.path.join(results_path, 'PR_curve.png')
        if os.path.exists(pr_curve_path):
            img = Image.open(pr_curve_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # 4. F1 Curve
        f1_curve_path = os.path.join(results_path, 'F1_curve.png')
        if os.path.exists(f1_curve_path):
            img = Image.open(f1_curve_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title('F1-Confidence Curve', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    def plot_validation_samples(self, results_path=None):
        """검증 샘플 시각화"""
        if results_path is None:
            results_path = os.path.join(Config.HOME, 'runs', 'detect', 'fire_smoke_detection')
        
        print("=== 검증 샘플 시각화 ===")
        
        # 검증 배치 이미지들 찾기
        val_batch_images = glob.glob(os.path.join(results_path, 'val_batch*_pred.jpg'))
        
        if not val_batch_images:
            print("검증 배치 이미지를 찾을 수 없습니다.")
            return
        
        # 처음 2개 배치 표시
        for i, batch_path in enumerate(val_batch_images[:2]):
            img = Image.open(batch_path)
            plt.figure(figsize=(15, 10))
            plt.imshow(img)
            plt.title(f'Validation Batch {i+1} Predictions', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    def plot_class_distribution(self, dataset_path=None):
        """데이터셋 클래스 분포 시각화"""
        if dataset_path is None:
            dataset_path = Config.DATASET_BASE_PATH
        
        print("=== 클래스 분포 분석 ===")
        
        class_counts = {name: 0 for name in Config.CLASS_NAMES}
        total_objects = 0
        
        # 모든 라벨 파일에서 클래스 카운트
        for split in ['train', 'valid', 'test']:
            labels_path = os.path.join(dataset_path, split, 'labels')
            if os.path.exists(labels_path):
                label_files = glob.glob(os.path.join(labels_path, '*.txt'))
                
                for label_file in label_files:
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(Config.CLASS_NAMES):
                                    class_counts[Config.CLASS_NAMES[class_id]] += 1
                                    total_objects += 1
                    except:
                        continue
        
        if total_objects == 0:
            print("클래스 데이터를 찾을 수 없습니다.")
            return
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 막대 그래프
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors_list = [self.colors.get(cls, '#888888') for cls in classes]
        
        bars = ax1.bar(classes, counts, color=colors_list, alpha=0.8, edgecolor='black')
        ax1.set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Objects')
        ax1.grid(True, alpha=0.3)
        
        # 막대 위에 수치 표시
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 파이 차트
        wedges, texts, autotexts = ax2.pie(counts, labels=classes, colors=colors_list, 
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        # 텍스트 스타일 조정
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.show()
        
        return class_counts
    
    def plot_detection_samples(self, dataset_path=None, num_samples=6):
        """라벨링된 샘플 이미지 시각화"""
        if dataset_path is None:
            dataset_path = Config.DATASET_BASE_PATH
        
        print(f"=== 라벨링 샘플 시각화 ({num_samples}개) ===")
        
        # 훈련 이미지에서 샘플 선택
        train_images_path = os.path.join(dataset_path, 'train', 'images')
        train_labels_path = os.path.join(dataset_path, 'train', 'labels')
        
        if not os.path.exists(train_images_path):
            print("훈련 이미지 디렉토리를 찾을 수 없습니다.")
            return
        
        # 이미지 파일 찾기
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(train_images_path, ext)))
        
        if len(image_files) == 0:
            print("이미지 파일을 찾을 수 없습니다.")
            return
        
        # 랜덤 샘플 선택
        import random
        sample_images = random.sample(image_files, min(num_samples, len(image_files)))
        
        # 그리드 설정
        rows = (num_samples + 2) // 3
        cols = min(3, num_samples)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, img_path in enumerate(sample_images):
            if i >= len(axes):
                break
            
            try:
                # 이미지 로드
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_height, img_width = image.shape[:2]
                
                # 라벨 파일 경로
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(train_labels_path, f"{base_name}.txt")
                
                # 라벨이 있으면 박스 그리기
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        labels = f.readlines()
                    
                    for label in labels:
                        parts = label.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # YOLO 형식을 픽셀 좌표로 변환
                            x1 = int((x_center - width/2) * img_width)
                            y1 = int((y_center - height/2) * img_height)
                            x2 = int((x_center + width/2) * img_width)
                            y2 = int((y_center + height/2) * img_height)
                            
                            # 박스 그리기
                            if class_id < len(Config.CLASS_NAMES):
                                class_name = Config.CLASS_NAMES[class_id]
                                color = self.colors.get(class_name, '#888888')
                                # BGR 값으로 변환
                                color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                
                                cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 2)
                                cv2.putText(image, class_name, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                
                axes[i].imshow(image)
                axes[i].set_title(f'Sample {i+1}: {os.path.basename(img_path)}')
                axes[i].axis('off')
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error loading\n{os.path.basename(img_path)}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Error: Sample {i+1}')
                axes[i].axis('off')
        
        # 빈 subplot 숨기기
        for i in range(len(sample_images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_inference_results(self, results_dict):
        """추론 결과 시각화"""
        print("=== 추론 결과 시각화 ===")
        
        if 'detections' not in results_dict or not results_dict['detections']:
            print("시각화할 감지 결과가 없습니다.")
            return
        
        detections = results_dict['detections']
        image_path = results_dict.get('image_path', '')
        
        # 클래스별 감지 수 계산
        class_counts = {}
        confidence_scores = []
        
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            confidence_scores.append(confidence)
        
        # 시각화
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 클래스별 감지 수 (상단 좌측)
        ax1 = plt.subplot(2, 3, 1)
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            colors_list = [self.colors.get(cls, '#888888') for cls in classes]
            
            bars = ax1.bar(classes, counts, color=colors_list, alpha=0.8, edgecolor='black')
            ax1.set_title('Detections by Class')
            ax1.set_ylabel('Count')
            
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. 신뢰도 분포 (상단 중앙)
        ax2 = plt.subplot(2, 3, 2)
        if confidence_scores:
            ax2.hist(confidence_scores, bins=10, alpha=0.7, edgecolor='black', color='skyblue')
            ax2.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                       label=f'평균: {np.mean(confidence_scores):.3f}')
            ax2.set_title('Confidence Score Distribution')
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Frequency')
            ax2.legend()
        
        # 3. 신뢰도 박스 플롯 (상단 우측)
        ax3 = plt.subplot(2, 3, 3)
        if confidence_scores:
            ax3.boxplot(confidence_scores, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue'))
            ax3.set_title('Confidence Score Box Plot')
            ax3.set_ylabel('Confidence')
            ax3.set_xticklabels(['All Detections'])
        
        # 4. 감지 통계 테이블 (하단)
        ax4 = plt.subplot(2, 1, 2)
        ax4.axis('off')
        
        # 통계 정보 생성
        stats_data = []
        total_detections = len(detections)
        
        stats_data.append(['총 감지 수', str(total_detections)])
        stats_data.append(['평균 신뢰도', f'{np.mean(confidence_scores):.3f}' if confidence_scores else 'N/A'])
        stats_data.append(['최대 신뢰도', f'{np.max(confidence_scores):.3f}' if confidence_scores else 'N/A'])
        stats_data.append(['최소 신뢰도', f'{np.min(confidence_scores):.3f}' if confidence_scores else 'N/A'])
        
        for class_name, count in class_counts.items():
            class_confidences = [d['confidence'] for d in detections if d['class'] == class_name]
            avg_conf = np.mean(class_confidences) if class_confidences else 0
            stats_data.append([f'{class_name} 감지 수', f'{count}개 (평균 신뢰도: {avg_conf:.3f})'])
        
        # 테이블 생성
        table = ax4.table(cellText=stats_data, 
                         colLabels=['항목', '값'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 헤더 스타일
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.suptitle(f'Inference Results Analysis\n{os.path.basename(image_path)}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_batch_analysis(self, batch_results):
        """배치 처리 결과 분석"""
        print("=== 배치 처리 결과 분석 ===")
        
        if not batch_results:
            print("분석할 배치 결과가 없습니다.")
            return
        
        # 데이터 수집
        detection_counts = []
        class_stats = {name: [] for name in Config.CLASS_NAMES}
        confidence_stats = []
        
        for result in batch_results:
            detections = result['detections']
            detection_counts.append(len(detections))
            
            for detection in detections:
                class_name = detection['class']
                confidence = detection['confidence']
                
                if class_name in class_stats:
                    class_stats[class_name].append(confidence)
                confidence_stats.append(confidence)
        
        # 시각화
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 이미지당 감지 수 히스토그램
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(detection_counts, bins=max(10, max(detection_counts)//2), 
                alpha=0.7, edgecolor='black', color='lightblue')
        ax1.set_title('Detections per Image')
        ax1.set_xlabel('Number of Detections')
        ax1.set_ylabel('Number of Images')
        ax1.grid(True, alpha=0.3)
        
        # 2. 전체 신뢰도 분포
        ax2 = plt.subplot(2, 3, 2)
        if confidence_stats:
            ax2.hist(confidence_stats, bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
            ax2.axvline(np.mean(confidence_stats), color='red', linestyle='--',
                       label=f'평균: {np.mean(confidence_stats):.3f}')
            ax2.set_title('Overall Confidence Distribution')
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 클래스별 감지 수
        ax3 = plt.subplot(2, 3, 3)
        class_counts = {name: len(confidences) for name, confidences in class_stats.items()}
        if any(class_counts.values()):
            classes = [name for name, count in class_counts.items() if count > 0]
            counts = [class_counts[name] for name in classes]
            colors_list = [self.colors.get(cls, '#888888') for cls in classes]
            
            bars = ax3.bar(classes, counts, color=colors_list, alpha=0.8, edgecolor='black')
            ax3.set_title('Total Detections by Class')
            ax3.set_ylabel('Total Count')
            ax3.grid(True, alpha=0.3)
            
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # 4. 클래스별 신뢰도 박스플롯
        ax4 = plt.subplot(2, 3, 4)
        box_data = []
        box_labels = []
        colors_for_box = []
        
        for class_name, confidences in class_stats.items():
            if confidences:
                box_data.append(confidences)
                box_labels.append(class_name)
                colors_for_box.append(self.colors.get(class_name, '#888888'))
        
        if box_data:
            bp = ax4.boxplot(box_data, patch_artist=True, labels=box_labels)
            for patch, color in zip(bp['boxes'], colors_for_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax4.set_title('Confidence by Class')
            ax4.set_ylabel('Confidence')
            ax4.grid(True, alpha=0.3)
        
        # 5. 감지율 파이차트
        ax5 = plt.subplot(2, 3, 5)
        images_with_detections = sum(1 for count in detection_counts if count > 0)
        images_without_detections = len(detection_counts) - images_with_detections
        
        if images_with_detections > 0 or images_without_detections > 0:
            sizes = [images_with_detections, images_without_detections]
            labels = ['With Detections', 'Without Detections']
            colors = ['#4CAF50', '#FFC107']
            
            wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors,
                                             autopct='%1.1f%%', startangle=90)
            ax5.set_title('Detection Rate')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 6. 통계 요약 테이블
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # 요약 통계 계산
        total_images = len(batch_results)
        total_detections = sum(detection_counts)
        avg_detections = np.mean(detection_counts) if detection_counts else 0
        detection_rate = (images_with_detections / total_images * 100) if total_images > 0 else 0
        
        summary_data = [
            ['총 이미지 수', f'{total_images:,}'],
            ['총 감지 수', f'{total_detections:,}'],
            ['감지된 이미지', f'{images_with_detections:,}'],
            ['감지율', f'{detection_rate:.1f}%'],
            ['이미지당 평균 감지', f'{avg_detections:.2f}'],
            ['평균 신뢰도', f'{np.mean(confidence_stats):.3f}' if confidence_stats else 'N/A']
        ]
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['지표', '값'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # 테이블 스타일
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#2196F3')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
        
        plt.suptitle('Batch Processing Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return {
            'total_images': total_images,
            'total_detections': total_detections,
            'detection_rate': detection_rate,
            'class_stats': class_stats
        }
    
    def plot_model_comparison(self, models_data):
        """여러 모델 성능 비교"""
        print("=== 모델 성능 비교 ===")
        
        if not models_data or len(models_data) < 2:
            print("비교할 모델이 충분하지 않습니다. (최소 2개 필요)")
            return
        
        # 데이터 정리
        model_names = list(models_data.keys())
        metrics = ['map50', 'map75', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 각 메트릭별 비교
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            values = []
            for model_name in model_names:
                value = models_data[model_name].get(metric, 0)
                values.append(value)
            
            # 막대 그래프
            bars = axes[i].bar(model_names, values, alpha=0.8, 
                              color=plt.cm.Set3(np.arange(len(model_names))))
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3)
            
            # 값 표시
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 종합 레이더 차트
        if len(axes) > len(metrics):
            ax_radar = axes[len(metrics)]
            ax_radar.remove()
            
            # 레이더 차트를 위한 새로운 subplot
            ax_radar = fig.add_subplot(2, 3, len(metrics)+1, projection='polar')
            
            # 각도 설정
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))
            
            # 각 모델별 레이더 차트
            for j, model_name in enumerate(model_names):
                values = []
                for metric in metrics:
                    values.append(models_data[model_name].get(metric, 0))
                values = np.concatenate((values, [values[0]]))
                
                ax_radar.plot(angles, values, 'o-', linewidth=2, 
                             label=model_name, color=plt.cm.Set3(j))
                ax_radar.fill(angles, values, alpha=0.25, color=plt.cm.Set3(j))
            
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels([m.upper() for m in metrics])
            ax_radar.set_ylim(0, 1)
            ax_radar.set_title('Overall Performance Comparison', pad=20)
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def save_visualization_report(self, output_path='visualization_report.html'):
        """시각화 보고서 HTML 생성"""
        print("=== 시각화 보고서 생성 ===")
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>YOLOv8 화재/연기 감지 - 시각화 보고서</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; 
                          border-radius: 5px; text-align: center; min-width: 150px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔥 YOLOv8 화재/연기 감지 시스템</h1>
                <h2>시각화 보고서</h2>
                <p>생성일: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h3>📊 주요 지표</h3>
                <div class="metric">
                    <div class="metric-value">3</div>
                    <div class="metric-label">감지 클래스</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{Config.CLASS_NAMES}</div>
                    <div class="metric-label">클래스 목록</div>
                </div>
            </div>
            
            <div class="section">
                <h3>🔍 모델 정보</h3>
                <p><strong>모델 아키텍처:</strong> YOLOv8</p>
                <p><strong>입력 크기:</strong> 640x640</p>
                <p><strong>클래스:</strong> {', '.join(Config.CLASS_NAMES)}</p>
            </div>
            
            <div class="section">
                <h3>📈 성능 분석</h3>
                <p>상세한 성능 분석 결과는 각 시각화 함수를 통해 확인할 수 있습니다.</p>
                <ul>
                    <li>plot_training_results() - 훈련 결과</li>
                    <li>plot_validation_samples() - 검증 샘플</li>
                    <li>plot_class_distribution() - 클래스 분포</li>
                    <li>plot_detection_samples() - 감지 샘플</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ 시각화 보고서 생성 완료: {output_path}")
        except Exception as e:
            print(f"❌ 보고서 생성 실패: {e}")

def main():
    """시각화 유틸리티 테스트"""
    viz = VisualizationUtils()
    
    print("시각화 유틸리티 준비 완료!")
    print("\n사용 가능한 기능:")
    print("- viz.plot_training_results(): 훈련 결과 시각화")
    print("- viz.plot_validation_samples(): 검증 샘플 시각화")
    print("- viz.plot_class_distribution(): 클래스 분포 분석")
    print("- viz.plot_detection_samples(): 라벨링 샘플 시각화")
    print("- viz.plot_inference_results(results): 추론 결과 분석")
    print("- viz.plot_batch_analysis(batch_results): 배치 결과 분석")

if __name__ == "__main__":
    main()