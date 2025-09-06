# local_config.py - 로컬 환경 전용 설정
"""
로컬 환경에서 실행하기 위한 안전한 설정 파일
"""

import os
import torch
from pathlib import Path

# YAML 처리를 위한 안전한 import
try:
    import yaml
except ImportError:
    print("PyYAML 패키지를 설치합니다...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyYAML'])
    import yaml

class LocalConfig:
    """로컬 환경 전용 설정 클래스"""
    
    # 현재 디렉토리 기준 경로 설정
    HOME = os.getcwd()
    DATASET_BASE_PATH = os.path.join(HOME, 'datasets', 'fire')
    RESULTS_PATH = os.path.join(HOME, 'runs', 'detect')
    
    # 모델 설정
    MODEL_SIZE = 'yolov8n.pt'
    EPOCHS = 100
    BATCH_SIZE = 32  # RTX 4060에 최적화
    IMAGE_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.5
    
    # 클래스 설정
    CLASS_NAMES = ['Fire', 'default', 'smoke']
    NUM_CLASSES = len(CLASS_NAMES)
    
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
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        dirs_to_create = [
            cls.DATASET_BASE_PATH,
            cls.RESULTS_PATH,
            os.path.join(cls.DATASET_BASE_PATH, 'train', 'images'),
            os.path.join(cls.DATASET_BASE_PATH, 'train', 'labels'),
            os.path.join(cls.DATASET_BASE_PATH, 'valid', 'images'),
            os.path.join(cls.DATASET_BASE_PATH, 'valid', 'labels'),
            os.path.join(cls.DATASET_BASE_PATH, 'test', 'images'),
            os.path.join(cls.DATASET_BASE_PATH, 'test', 'labels')
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ 디렉토리 생성/확인: {directory}")

def setup_local_environment():
    """로컬 환경 설정"""
    print("=== 로컬 환경 설정 ===")
    
    # GPU 확인
    gpu_available = torch.cuda.is_available()
    print(f"GPU 사용 가능: {gpu_available}")
    if gpu_available:
        print(f"GPU 모델: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    
    # 현재 작업 디렉토리 확인
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    
    # 필요한 디렉토리 생성
    LocalConfig.create_directories()
    
    return {
        'gpu_available': gpu_available,
        'home_dir': LocalConfig.HOME,
        'dataset_path': LocalConfig.DATASET_BASE_PATH
    }

def install_requirements():
    """필수 패키지 설치"""
    import subprocess
    import sys
    
    packages = [
        'ultralytics',
        'roboflow',
        'opencv-python',
        'matplotlib',
        'pillow',
        'pyyaml',
        'seaborn',
        'pandas',
        'scipy'
    ]
    
    print("=== 필수 패키지 설치 확인 ===")
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} 이미 설치됨")
        except ImportError:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} 설치 완료")
            except subprocess.CalledProcessError:
                print(f"❌ {package} 설치 실패")

def create_local_data_yaml():
    """로컬 환경용 data.yaml 생성"""
    print("=== 로컬 data.yaml 생성 ===")
    
    # 기존 파일 확인
    existing_yaml = os.path.join(LocalConfig.DATASET_BASE_PATH, 'data.yaml')
    target_yaml = os.path.join(LocalConfig.HOME, 'data.yaml')
    
    if os.path.exists(existing_yaml):
        print(f"📋 기존 data.yaml 발견: {existing_yaml}")
        # 기존 파일 복사 및 경로 수정
        try:
            with open(existing_yaml, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            # 로컬 경로로 수정
            data_config['train'] = LocalConfig.get_dataset_paths()['train']
            data_config['val'] = LocalConfig.get_dataset_paths()['val']
            data_config['test'] = LocalConfig.get_dataset_paths()['test']
            
        except Exception as e:
            print(f"⚠️ 기존 yaml 처리 중 오류: {e}, 새로 생성합니다.")
            data_config = {
                'train': LocalConfig.get_dataset_paths()['train'],
                'val': LocalConfig.get_dataset_paths()['val'],
                'test': LocalConfig.get_dataset_paths()['test'],
                'nc': LocalConfig.NUM_CLASSES,
                'names': LocalConfig.CLASS_NAMES,
                'roboflow': LocalConfig.ROBOFLOW_CONFIG
            }
    else:
        print("📋 새 data.yaml 파일 생성")
        data_config = {
            'train': LocalConfig.get_dataset_paths()['train'],
            'val': LocalConfig.get_dataset_paths()['val'],
            'test': LocalConfig.get_dataset_paths()['test'],
            'nc': LocalConfig.NUM_CLASSES,
            'names': LocalConfig.CLASS_NAMES,
            'roboflow': LocalConfig.ROBOFLOW_CONFIG
        }
    
    # 파일 저장
    with open(target_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ data.yaml 파일 생성 완료: {target_yaml}")
    
    # 내용 출력
    with open(target_yaml, 'r', encoding='utf-8') as f:
        print("\n=== data.yaml 내용 ===")
        print(f.read())
    
    return target_yaml

def verify_local_setup():
    """로컬 환경 설정 검증"""
    print("=== 로컬 환경 검증 ===")
    
    checks = []
    
    # 1. GPU 확인
    gpu_ok = torch.cuda.is_available()
    checks.append(("GPU 사용 가능", gpu_ok))
    
    # 2. 디렉토리 확인
    dirs_ok = all(os.path.exists(path) for path in [
        LocalConfig.HOME,
        LocalConfig.DATASET_BASE_PATH,
        LocalConfig.RESULTS_PATH
    ])
    checks.append(("디렉토리 구조", dirs_ok))
    
    # 3. 패키지 확인
    try:
        from ultralytics import YOLO
        import roboflow
        packages_ok = True
    except ImportError:
        packages_ok = False
    checks.append(("필수 패키지", packages_ok))
    
    # 4. data.yaml 확인
    yaml_ok = os.path.exists(os.path.join(LocalConfig.HOME, 'data.yaml'))
    checks.append(("data.yaml 파일", yaml_ok))
    
    # 결과 출력
    print("\n검증 결과:")
    all_ok = True
    for check_name, status in checks:
        icon = "✅" if status else "❌"
        print(f"  {icon} {check_name}: {'OK' if status else 'FAIL'}")
        if not status:
            all_ok = False
    
    if all_ok:
        print("\n🎉 로컬 환경 설정 완료!")
    else:
        print("\n⚠️ 일부 설정에 문제가 있습니다.")
    
    return all_ok

def initialize_local_project():
    """로컬 프로젝트 초기화"""
    print("🔥 로컬 환경 YOLOv8 화재 감지 프로젝트 초기화")
    print("=" * 60)
    
    # 1. 패키지 설치
    install_requirements()
    
    # 2. 환경 설정
    env_result = setup_local_environment()
    
    # 3. data.yaml 생성
    yaml_path = create_local_data_yaml()
    
    # 4. 설정 검증
    setup_ok = verify_local_setup()
    
    return {
        'setup_ok': setup_ok,
        'env_result': env_result,
        'yaml_path': yaml_path
    }

if __name__ == "__main__":
    initialize_local_project()
