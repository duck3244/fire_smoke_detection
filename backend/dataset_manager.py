# dataset_manager.py - 데이터셋 관리
"""
YOLOv8 화재 및 연기 감지 프로젝트 데이터셋 관리 모듈
"""

import os
import glob
import shutil
import random
import logging
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from config import Config

logger = logging.getLogger(__name__)


class DatasetManager:
    """데이터셋 관리 클래스"""

    def __init__(self):
        self.base_path = Config.DATASET_BASE_PATH
        self.dataset_paths = Config.get_dataset_paths()
        self.label_paths = Config.get_label_paths()

    def create_directory_structure(self):
        """데이터셋 디렉토리 구조 생성"""
        logger.info("=== 데이터셋 디렉토리 구조 생성 ===")

        # 이미지 디렉토리 생성
        for split, path in self.dataset_paths.items():
            os.makedirs(path, exist_ok=True)
            logger.info(f"✅ 생성: {path}")

        # 라벨 디렉토리 생성
        for split, path in self.label_paths.items():
            os.makedirs(path, exist_ok=True)
            logger.info(f"✅ 생성: {path}")

        logger.info("디렉토리 구조 생성 완료!\n")

    def check_dataset_status(self):
        """데이터셋 상태 확인"""
        logger.info("=== 데이터셋 상태 확인 ===")

        total_images = 0
        total_labels = 0

        for split in ['train', 'val', 'test']:
            img_path = self.dataset_paths[split]
            label_path = self.label_paths[split]

            # 이미지 파일 확인
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            images = []
            for ext in image_extensions:
                images.extend(glob.glob(os.path.join(img_path, ext)))
                images.extend(glob.glob(os.path.join(img_path, ext.upper())))

            # 라벨 파일 확인
            labels = glob.glob(os.path.join(label_path, '*.txt'))

            num_images = len(images)
            num_labels = len(labels)

            total_images += num_images
            total_labels += num_labels

            logger.info(f"{split.upper():5} - 이미지: {num_images:4}개, 라벨: {num_labels:4}개")

            # 데이터 불균형 확인
            if num_images > 0 and num_labels > 0:
                if num_images != num_labels:
                    logger.info(f"  ⚠️  {split} 세트에서 이미지와 라벨 수가 일치하지 않습니다!")

        logger.info(f"\n총 이미지: {total_images}개, 총 라벨: {total_labels}개")

        if total_images == 0:
            logger.info("❌ 데이터셋이 비어있습니다. 데이터를 업로드해주세요.")
            self._print_upload_instructions()

        return {
            'total_images': total_images,
            'total_labels': total_labels,
            'has_data': total_images > 0
        }

    def _print_upload_instructions(self):
        """데이터 업로드 안내"""
        logger.info("\n=== 데이터 업로드 안내 ===")
        logger.info("1. Roboflow에서 데이터셋 다운로드:")
        logger.info("   URL: https://universe.roboflow.com/custom-thxhn/fire-wrpgm/dataset/8")
        logger.info("\n2. Google Drive에 다음 구조로 업로드:")
        logger.info(f"   {self.base_path}/")
        logger.info("   ├── train/")
        logger.info("   │   ├── images/  (훈련 이미지)")
        logger.info("   │   └── labels/  (훈련 라벨 .txt)")
        logger.info("   ├── valid/")
        logger.info("   │   ├── images/  (검증 이미지)")
        logger.info("   │   └── labels/  (검증 라벨 .txt)")
        logger.info("   └── test/")
        logger.info("       ├── images/  (테스트 이미지)")
        logger.info("       └── labels/  (테스트 라벨 .txt)")

    def visualize_dataset_samples(self, num_samples=4):
        """데이터셋 샘플 시각화"""
        logger.info("=== 데이터셋 샘플 시각화 ===")

        # 훈련 데이터에서 샘플 이미지 가져오기
        train_images = glob.glob(os.path.join(self.dataset_paths['train'], '*'))

        if len(train_images) == 0:
            logger.info("시각화할 이미지가 없습니다.")
            return

        # 랜덤 샘플 선택
        sample_images = random.sample(train_images, min(num_samples, len(train_images)))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, img_path in enumerate(sample_images):
            if i >= len(axes):
                break

            try:
                # 이미지 로드
                image = Image.open(img_path)
                axes[i].imshow(image)
                axes[i].set_title(f"Sample {i + 1}: {os.path.basename(img_path)}")
                axes[i].axis('off')

                # 대응하는 라벨 파일 확인
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(self.label_paths['train'], f"{base_name}.txt")

                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        labels = f.readlines()
                    axes[i].set_xlabel(f"Labels: {len(labels)} objects")
                else:
                    axes[i].set_xlabel("No label file found")

            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error loading image:\n{str(e)}',
                             ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"Error: {os.path.basename(img_path)}")

        plt.tight_layout()
        plt.show()

    def analyze_class_distribution(self):
        """클래스 분포 분석"""
        logger.info("=== 클래스 분포 분석 ===")

        class_counts = {name: 0 for name in Config.CLASS_NAMES}
        total_objects = 0

        # 모든 라벨 파일 확인
        for split in ['train', 'val', 'test']:
            label_files = glob.glob(os.path.join(self.label_paths[split], '*.txt'))

            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # YOLO 형식: class_id x y w h
                            class_id = int(parts[0])
                            if 0 <= class_id < len(Config.CLASS_NAMES):
                                class_counts[Config.CLASS_NAMES[class_id]] += 1
                                total_objects += 1

                except Exception as e:
                    logger.info(f"라벨 파일 읽기 오류 {label_file}: {e}")

        # 결과 출력
        logger.info(f"총 객체 수: {total_objects}")
        for class_name, count in class_counts.items():
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            logger.info(f"{class_name:10}: {count:5}개 ({percentage:5.1f}%)")

        # 시각화
        if total_objects > 0:
            plt.figure(figsize=(10, 6))
            classes = list(class_counts.keys())
            counts = list(class_counts.values())

            plt.bar(classes, counts)
            plt.title('Class Distribution in Dataset')
            plt.xlabel('Classes')
            plt.ylabel('Number of Objects')

            # 백분율 표시
            for i, count in enumerate(counts):
                percentage = count / total_objects * 100
                plt.text(i, count + max(counts) * 0.01, f'{percentage:.1f}%',
                         ha='center', va='bottom')

            plt.tight_layout()
            plt.show()

        return class_counts

    def validate_dataset_format(self):
        """데이터셋 형식 검증"""
        logger.info("=== 데이터셋 형식 검증 ===")

        issues = []

        for split in ['train', 'val', 'test']:
            logger.info(f"\n{split.upper()} 데이터 검증 중...")

            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(self.dataset_paths[split], ext)))

            for img_file in image_files:
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                label_file = os.path.join(self.label_paths[split], f"{base_name}.txt")

                # 라벨 파일 존재 확인
                if not os.path.exists(label_file):
                    issues.append(f"Missing label for {img_file}")
                    continue

                # 라벨 형식 검증
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()

                    for line_num, line in enumerate(lines, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            issues.append(f"Invalid format in {label_file}:{line_num}")
                            continue

                        # YOLO 형식 검증
                        try:
                            class_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:])

                            if not (0 <= class_id < Config.NUM_CLASSES):
                                issues.append(f"Invalid class_id {class_id} in {label_file}:{line_num}")

                            if not all(0 <= val <= 1 for val in [x, y, w, h]):
                                issues.append(f"Invalid bbox coordinates in {label_file}:{line_num}")

                        except ValueError:
                            issues.append(f"Invalid number format in {label_file}:{line_num}")

                except Exception as e:
                    issues.append(f"Error reading {label_file}: {e}")

        # 검증 결과 출력
        if issues:
            logger.info(f"\n❌ {len(issues)}개의 문제 발견:")
            for issue in issues[:10]:  # 처음 10개만 출력
                logger.info(f"  - {issue}")
            if len(issues) > 10:
                logger.info(f"  ... 그리고 {len(issues) - 10}개 더")
        else:
            logger.info("\n✅ 데이터셋 형식 검증 완료! 문제 없음")

        return len(issues) == 0

    def split_dataset(self, source_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """데이터셋 분할 (이미 분할된 데이터가 없을 때 사용)"""
        if train_ratio + val_ratio + test_ratio != 1.0:
            logger.info("❌ 비율의 합이 1.0이 되어야 합니다.")
            return

        logger.info(f"=== 데이터셋 분할 (Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}) ===")

        # 소스 디렉토리의 모든 이미지 파일 가져오기
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(source_dir, 'images', ext)))

        # 랜덤 셔플
        random.shuffle(image_files)

        total_files = len(image_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }

        # 파일 복사
        for split, files in splits.items():
            logger.info(f"{split}: {len(files)}개 파일")

            for img_file in files:
                # 이미지 복사
                dst_img = os.path.join(self.dataset_paths[split], os.path.basename(img_file))
                shutil.copy2(img_file, dst_img)

                # 대응하는 라벨 파일 복사
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                src_label = os.path.join(source_dir, 'labels', f"{base_name}.txt")

                if os.path.exists(src_label):
                    dst_label = os.path.join(self.label_paths[split], f"{base_name}.txt")
                    shutil.copy2(src_label, dst_label)

        logger.info("데이터셋 분할 완료!")


def main():
    """데이터셋 매니저 테스트"""
    manager = DatasetManager()

    # 디렉토리 구조 생성
    manager.create_directory_structure()

    # 데이터셋 상태 확인
    status = manager.check_dataset_status()

    if not status['has_data']:
        logger.info("\n⚠️ 데이터셋이 비어 있습니다.")
        logger.info("Roboflow 데이터셋을 다운로드하려면 다음 스크립트를 사용하세요:")
        logger.info("  python download_dataset.py")
        logger.info("  python instant_download.py")
        logger.info("  python roboflow_dataset_downloader.py")
        return

    # 데이터가 있으면 추가 분석
    logger.info("\n📊 데이터셋 분석 시작...")
    manager.validate_dataset_format()
    manager.analyze_class_distribution()
    manager.visualize_dataset_samples()
    logger.info("✅ 데이터셋 준비 완료!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()