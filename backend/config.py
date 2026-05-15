# config.py - 간단하고 안전한 설정 파일
"""
모든 오류를 해결한 간단한 설정 파일
"""

import os
import logging
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)

# config.py는 backend/ 내부에 위치. datasets/, runs/ 는 backend의 상위(레포 루트)에 둠.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
CURRENT_DIR = str(PROJECT_ROOT)


class Config:
    """간단한 프로젝트 설정 클래스"""

    # 기본 경로 (현재 디렉토리 기준)
    HOME = CURRENT_DIR
    DATASET_BASE_PATH = os.path.join(CURRENT_DIR, 'datasets', 'fire')
    RESULTS_PATH = os.path.join(CURRENT_DIR, 'runs', 'detect')

    # 모델 설정
    MODEL_SIZE = 'yolov8n.pt'
    EPOCHS = 100
    BATCH_SIZE = 32  # RTX 4060 최적화
    IMAGE_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.5

    # 클래스 설정
    CLASS_NAMES = ['Fire', 'smoke']
    NUM_CLASSES = 2

    # Roboflow 설정
    ROBOFLOW_CONFIG = {
        'license': 'CC BY 4.0',
        'project': 'fire-wrpgm',
        'url': 'https://universe.roboflow.com/custom-thxhn/fire-wrpgm/dataset/8',
        'version': 8,
        'workspace': 'custom-thxhn'
    }

    @classmethod
    def get_dataset_paths(cls):
        """데이터셋 경로 반환"""
        return {
            'train': os.path.join(cls.DATASET_BASE_PATH, 'train', 'images'),
            'val': os.path.join(cls.DATASET_BASE_PATH, 'valid', 'images'),
            'test': os.path.join(cls.DATASET_BASE_PATH, 'test', 'images')
        }

    @classmethod
    def get_label_paths(cls):
        """라벨 경로 반환"""
        return {
            'train': os.path.join(cls.DATASET_BASE_PATH, 'train', 'labels'),
            'val': os.path.join(cls.DATASET_BASE_PATH, 'valid', 'labels'),
            'test': os.path.join(cls.DATASET_BASE_PATH, 'test', 'labels')
        }


def setup_environment():
    """환경 설정 및 확인"""
    logger.info("=== 환경 설정 확인 ===")

    # GPU 확인
    gpu_available = torch.cuda.is_available()
    logger.info(f"GPU 사용 가능: {gpu_available}")
    if gpu_available:
        logger.info(f"GPU 모델: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")

    # 디렉토리 정보 출력
    logger.info(f"현재 작업 디렉토리: {os.getcwd()}")
    logger.info(f"HOME: {Config.HOME}")
    logger.info(f"데이터셋 경로: {Config.DATASET_BASE_PATH}")

    return gpu_available


def mount_google_drive():
    """Google Drive 마운트 (로컬에서는 불필요)"""
    logger.info("로컬 환경에서는 Google Drive 마운트가 필요하지 않습니다.")
    return True


def check_requirements():
    """필수 패키지 import 가능 여부 확인 (설치는 외부 환경에 위임)"""
    required = {
        'ultralytics': 'ultralytics',
        'roboflow': 'roboflow',
        'opencv-python': 'cv2',
        'matplotlib': 'matplotlib',
        'pillow': 'PIL',
        'pyyaml': 'yaml',
    }

    missing = []
    for pkg, mod in required.items():
        try:
            __import__(mod)
            logger.info(f"✅ {pkg}")
        except ImportError:
            logger.info(f"❌ {pkg} 미설치")
            missing.append(pkg)

    if missing:
        logger.info("\n다음 명령으로 설치하세요:")
        logger.info(f"  pip install {' '.join(missing)}")
        logger.info("또는: pip install -r requirements.txt")

    return not missing


def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        Config.DATASET_BASE_PATH,
        Config.RESULTS_PATH,
        os.path.join(Config.DATASET_BASE_PATH, 'train', 'images'),
        os.path.join(Config.DATASET_BASE_PATH, 'train', 'labels'),
        os.path.join(Config.DATASET_BASE_PATH, 'valid', 'images'),
        os.path.join(Config.DATASET_BASE_PATH, 'valid', 'labels'),
        os.path.join(Config.DATASET_BASE_PATH, 'test', 'images'),
        os.path.join(Config.DATASET_BASE_PATH, 'test', 'labels')
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def create_data_yaml():
    """data.yaml 파일 생성 (path 기반 상대 경로)"""
    # 디렉토리 먼저 생성
    create_directories()

    target_yaml = os.path.join(Config.HOME, 'data.yaml')

    # Ultralytics는 path + 상대경로 형식을 권장
    data_config = {
        'path': Config.DATASET_BASE_PATH,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': Config.NUM_CLASSES,
        'names': Config.CLASS_NAMES
    }

    with open(target_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"✅ data.yaml 생성: {target_yaml}")
    return target_yaml


def verify_ultralytics():
    """Ultralytics 확인"""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        logger.info("✅ YOLOv8 로드 성공")
        return True
    except Exception as e:
        logger.info(f"⚠️ YOLOv8 확인 필요: {e}")
        return False


def initialize_project():
    """프로젝트 초기화"""
    logger.info("=== 간단한 프로젝트 초기화 ===")

    # 1. 환경 설정
    gpu_available = setup_environment()

    # 2. 드라이브 마운트 (로컬에서는 스킵)
    mount_google_drive()

    # 3. 패키지 설치 확인
    logger.info("패키지 확인 중...")
    check_requirements()

    # 4. YOLOv8 확인
    ultralytics_ok = verify_ultralytics()

    # 5. data.yaml 생성
    yaml_path = create_data_yaml()

    logger.info("\n✅ 초기화 완료!")

    return {
        'gpu_available': gpu_available,
        'ultralytics_ok': ultralytics_ok,
        'yaml_path': yaml_path
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    initialize_project()