#!/usr/bin/env python3
# instant_download.py - 즉시 데이터셋 다운로드
"""
Roboflow에서 화재 데이터셋을 즉시 다운로드하고 올바른 위치에 배치
"""

import os
import shutil
import subprocess
import sys

def install_roboflow():
    """roboflow 패키지 설치"""
    try:
        import roboflow
        print("✅ roboflow 패키지 이미 설치됨")
        return True
    except ImportError:
        print("📦 roboflow 패키지 설치 중...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'roboflow'])
            print("✅ roboflow 패키지 설치 완료")
            return True
        except Exception as e:
            print(f"❌ roboflow 설치 실패: {e}")
            return False

def download_dataset(api_key):
    """데이터셋 다운로드"""
    try:
        import roboflow
        
        print("🔥 화재 데이터셋 다운로드 시작...")
        print("데이터셋: custom-thxhn/fire-wrpgm (979개 이미지)")
        
        rf = roboflow.Roboflow(api_key=api_key)
        project = rf.workspace("custom-thxhn").project("fire-wrpgm")
        dataset = project.version(8).download("yolov8")
        
        print(f"✅ 다운로드 완료: {dataset.location}")
        return dataset.location
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        print("\n가능한 원인:")
        print("1. API 키가 올바르지 않음")
        print("2. 인터넷 연결 문제")
        print("3. Roboflow 서비스 일시적 문제")
        return None

def organize_dataset(download_path):
    """다운로드된 데이터셋을 프로젝트 구조에 맞게 정리"""
    target_base = os.path.join(os.getcwd(), 'datasets', 'fire')
    
    print(f"📁 데이터셋 정리 중...")
    print(f"소스: {download_path}")
    print(f"타겟: {target_base}")
    
    try:
        # 타겟 디렉토리 생성
        for split in ['train', 'valid', 'test']:
            for data_type in ['images', 'labels']:
                target_dir = os.path.join(target_base, split, data_type)
                os.makedirs(target_dir, exist_ok=True)
        
        # 파일 복사
        copied_files = 0
        for split in ['train', 'valid', 'test']:
            # 소스 경로
            src_images = os.path.join(download_path, split, 'images')
            src_labels = os.path.join(download_path, split, 'labels')
            
            # 타겟 경로
            dst_images = os.path.join(target_base, split, 'images')
            dst_labels = os.path.join(target_base, split, 'labels')
            
            # 이미지 복사
            if os.path.exists(src_images):
                for file_name in os.listdir(src_images):
                    src_file = os.path.join(src_images, file_name)
                    dst_file = os.path.join(dst_images, file_name)
                    shutil.copy2(src_file, dst_file)
                    copied_files += 1
                print(f"✅ {split} 이미지 복사 완료")
            
            # 라벨 복사
            if os.path.exists(src_labels):
                for file_name in os.listdir(src_labels):
                    src_file = os.path.join(src_labels, file_name)
                    dst_file = os.path.join(dst_labels, file_name)
                    shutil.copy2(src_file, dst_file)
                    copied_files += 1
                print(f"✅ {split} 라벨 복사 완료")
        
        print(f"✅ 총 {copied_files}개 파일 복사 완료")
        
        # data.yaml 복사
        src_yaml = os.path.join(download_path, 'data.yaml')
        dst_yaml = os.path.join(os.getcwd(), 'data.yaml')
        
        if os.path.exists(src_yaml):
            shutil.copy2(src_yaml, dst_yaml)
            print("✅ data.yaml 복사 완료")
            
            # data.yaml 경로 수정
            fix_data_yaml_paths(dst_yaml, target_base)
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터셋 정리 실패: {e}")
        return False

def fix_data_yaml_paths(yaml_path, base_path):
    """data.yaml의 경로를 로컬 경로로 수정"""
    try:
        import yaml
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 경로 수정
        data['train'] = os.path.join(base_path, 'train', 'images')
        data['val'] = os.path.join(base_path, 'valid', 'images')
        data['test'] = os.path.join(base_path, 'test', 'images')
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print("✅ data.yaml 경로 수정 완료")
        
    except Exception as e:
        print(f"⚠️ data.yaml 수정 실패: {e}")

def verify_dataset():
    """데이터셋 설치 확인"""
    base_path = os.path.join(os.getcwd(), 'datasets', 'fire')
    
    print("\n=== 데이터셋 설치 확인 ===")
    
    total_images = 0
    total_labels = 0
    
    for split in ['train', 'valid', 'test']:
        images_path = os.path.join(base_path, split, 'images')
        labels_path = os.path.join(base_path, split, 'labels')
        
        if os.path.exists(images_path):
            images = len([f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_images += images
        else:
            images = 0
        
        if os.path.exists(labels_path):
            labels = len([f for f in os.listdir(labels_path) if f.endswith('.txt')])
            total_labels += labels
        else:
            labels = 0
        
        print(f"{split.upper():5} - 이미지: {images:4}개, 라벨: {labels:4}개")
    
    print(f"\n총 이미지: {total_images}개, 총 라벨: {total_labels}개")
    
    if total_images > 0:
        print("🎉 데이터셋 설치 완료!")
        return True
    else:
        print("❌ 데이터셋 설치 실패")
        return False

def cleanup_download_folder(download_path):
    """다운로드 임시 폴더 정리"""
    try:
        if os.path.exists(download_path) and download_path != os.getcwd():
            shutil.rmtree(download_path)
            print("🗑️ 임시 다운로드 폴더 정리 완료")
    except Exception as e:
        print(f"⚠️ 임시 폴더 정리 실패: {e}")

def main():
    """메인 다운로드 함수"""
    print("🔥 YOLOv8 화재 데이터셋 즉시 다운로드")
    print("=" * 50)
    
    # API 키 입력
    api_key = input("Roboflow API 키를 입력하세요: ").strip()
    
    if not api_key:
        print("❌ API 키가 필요합니다.")
        print("\nAPI 키 발급 방법:")
        print("1. https://roboflow.com 회원가입")
        print("2. Settings → API Keys → Generate New Key")
        return False
    
    # 1. roboflow 패키지 설치
    if not install_roboflow():
        return False
    
    # 2. 데이터셋 다운로드
    download_path = download_dataset(api_key)
    if not download_path:
        return False
    
    # 3. 데이터셋 정리
    if not organize_dataset(download_path):
        return False
    
    # 4. 설치 확인
    if verify_dataset():
        # 5. 임시 폴더 정리
        cleanup_download_folder(download_path)
        
        print("\n" + "=" * 50)
        print("🎉 데이터셋 다운로드 및 설치 완료!")
        print("\n다음 명령어로 훈련을 시작하세요:")
        print("python main.py --mode full --epochs 50 --batch-size 32")
        return True
    else:
        return False

if __name__ == "__main__":
    main()
