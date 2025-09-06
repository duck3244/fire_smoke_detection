# config.py - 간단하고 안전한 설정 파일
"""
모든 오류를 해결한 간단한 설정 파일
"""

import os
import torch

# YAML 처리
try:
    import yaml
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyYAML'])
    import yaml

# 현재 디렉토리 기준으로 모든 경로 설정
CURRENT_DIR = os.getcwd()


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
    CLASS_NAMES = ['Fire', 'default', 'smoke']
    NUM_CLASSES = 3

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
    print("=== 환경 설정 확인 ===")

    # GPU 확인
    gpu_available = torch.cuda.is_available()
    print(f"GPU 사용 가능: {gpu_available}")
    if gpu_available:
        print(f"GPU 모델: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")

    # 디렉토리 정보 출력
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"HOME: {Config.HOME}")
    print(f"데이터셋 경로: {Config.DATASET_BASE_PATH}")

    return gpu_available


def mount_google_drive():
    """Google Drive 마운트 (로컬에서는 불필요)"""
    print("로컬 환경에서는 Google Drive 마운트가 필요하지 않습니다.")
    return True


def install_requirements():
    """필수 패키지 설치"""
    packages = ['ultralytics', 'roboflow', 'opencv-python', 'matplotlib', 'pillow', 'pyyaml']

    import subprocess
    import sys

    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package}")
        except:
            print(f"⚠️ {package} 설치 확인 필요")


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
    """data.yaml 파일 생성"""
    # 디렉토리 먼저 생성
    create_directories()

    target_yaml = os.path.join(Config.HOME, 'data.yaml')

    data_config = {
        'train': Config.get_dataset_paths()['train'],
        'val': Config.get_dataset_paths()['val'],
        'test': Config.get_dataset_paths()['test'],
        'nc': Config.NUM_CLASSES,
        'names': Config.CLASS_NAMES
    }

    with open(target_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ data.yaml 생성: {target_yaml}")
    return target_yaml


def verify_ultralytics():
    """Ultralytics 확인"""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 로드 성공")
        return True
    except Exception as e:
        print(f"⚠️ YOLOv8 확인 필요: {e}")
        return False


def initialize_project():
    """프로젝트 초기화"""
    print("=== 간단한 프로젝트 초기화 ===")

    # 1. 환경 설정
    gpu_available = setup_environment()

    # 2. 드라이브 마운트 (로컬에서는 스킵)
    mount_google_drive()

    # 3. 패키지 설치 확인
    print("패키지 확인 중...")
    install_requirements()

    # 4. YOLOv8 확인
    ultralytics_ok = verify_ultralytics()

    # 5. data.yaml 생성
    yaml_path = create_data_yaml()

    print("\n✅ 초기화 완료!")

    return {
        'gpu_available': gpu_available,
        'ultralytics_ok': ultralytics_ok,
        'yaml_path': yaml_path
    }


if __name__ == "__main__":
    initialize_project()