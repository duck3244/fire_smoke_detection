# roboflow_dataset_downloader.py - Roboflow 데이터셋 다운로더
"""
Roboflow에서 화재/연기 감지 데이터셋을 다운로드하는 전용 모듈
"""

import os
import shutil
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm


class RoboflowDatasetDownloader:
    """Roboflow 데이터셋 다운로드 클래스"""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://universe.roboflow.com"

        # 추천 화재/연기 데이터셋 목록
        self.recommended_datasets = {
            "fire-wrpgm": {
                "workspace": "custom-thxhn",
                "project": "fire-wrpgm",
                "version": 8,
                "description": "979개 화재 이미지, 3클래스 (Fire, default, smoke)",
                "url": "https://universe.roboflow.com/custom-thxhn/fire-wrpgm/dataset/8",
                "classes": ["Fire", "default", "smoke"]
            },
            "fire-smoke-detection": {
                "workspace": "middle-east-tech-university",
                "project": "fire-and-smoke-detection-hiwia",
                "version": 3,
                "description": "6391개 화재/연기 이미지, 2클래스 (fire, smoke)",
                "url": "https://universe.roboflow.com/middle-east-tech-university/fire-and-smoke-detection-hiwia",
                "classes": ["fire", "smoke"]
            },
            "fire-smoke-yolov11": {
                "workspace": "sayed-gamall",
                "project": "fire-smoke-detection-yolov11",
                "version": 2,
                "description": "4359개 화재/연기 이미지, YOLOv11 최적화",
                "url": "https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11",
                "classes": ["fire", "smoke"]
            }
        }

    def get_api_key(self):
        """API 키 가져오기 및 설정 안내"""
        if self.api_key:
            return self.api_key

        print("🔑 Roboflow API 키가 필요합니다!")
        print("\nAPI 키 발급 방법:")
        print("1. https://roboflow.com 에 회원가입/로그인")
        print("2. 좌측 메뉴에서 'Settings' > 'API Keys' 클릭")
        print("3. 'Generate New Key' 버튼 클릭하여 새 키 생성")
        print("4. Private API Key를 복사")

        api_key = input("\nAPI 키를 입력하세요: ").strip()

        if not api_key:
            raise ValueError("API 키가 필요합니다!")

        self.api_key = api_key
        return api_key

    def list_recommended_datasets(self):
        """추천 데이터셋 목록 출력"""
        print("🔥 추천 화재/연기 감지 데이터셋:")
        print("=" * 80)

        for i, (key, dataset) in enumerate(self.recommended_datasets.items(), 1):
            print(f"{i}. {key}")
            print(f"   📊 {dataset['description']}")
            print(f"   🏷️  클래스: {', '.join(dataset['classes'])}")
            print(f"   🔗 URL: {dataset['url']}")
            print()

    def install_roboflow_package(self):
        """Roboflow 패키지 설치"""
        try:
            import roboflow
            print("✅ roboflow 패키지가 이미 설치되어 있습니다.")
            return True
        except ImportError:
            print("📦 roboflow 패키지를 설치합니다...")
            import subprocess
            import sys

            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'roboflow'])
                print("✅ roboflow 패키지 설치 완료!")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ 패키지 설치 실패: {e}")
                return False

    def download_dataset_by_key(self, dataset_key, output_dir=None, format="yolov8"):
        """추천 데이터셋 키로 다운로드"""
        if dataset_key not in self.recommended_datasets:
            print(f"❌ 알 수 없는 데이터셋 키: {dataset_key}")
            print("사용 가능한 키:", list(self.recommended_datasets.keys()))
            return None

        dataset_info = self.recommended_datasets[dataset_key]

        return self.download_dataset(
            workspace=dataset_info["workspace"],
            project=dataset_info["project"],
            version=dataset_info["version"],
            output_dir=output_dir,
            format=format
        )

    def download_dataset(self, workspace, project, version, output_dir=None, format="yolov8"):
        """Roboflow에서 데이터셋 다운로드"""
        # 패키지 설치 확인
        if not self.install_roboflow_package():
            return None

        # API 키 확인
        api_key = self.get_api_key()

        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = "/content/drive/MyDrive/datasets/fire"

        try:
            from roboflow import Roboflow

            print(f"🚀 데이터셋 다운로드 시작...")
            print(f"   Workspace: {workspace}")
            print(f"   Project: {project}")
            print(f"   Version: {version}")
            print(f"   Format: {format}")
            print(f"   Output: {output_dir}")

            # Roboflow 인스턴스 생성
            rf = Roboflow(api_key=api_key)
            project_obj = rf.workspace(workspace).project(project)
            dataset = project_obj.version(version).download(format, location=output_dir)

            print(f"✅ 데이터셋 다운로드 완료!")
            print(f"📁 저장 위치: {dataset.location}")

            # 다운로드된 데이터셋 정보 출력
            self._analyze_downloaded_dataset(dataset.location)

            return dataset

        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            print("\n해결 방법:")
            print("1. API 키가 올바른지 확인")
            print("2. 인터넷 연결 상태 확인")
            print("3. workspace/project/version 정보가 정확한지 확인")
            return None

    def _analyze_downloaded_dataset(self, dataset_path):
        """다운로드된 데이터셋 분석"""
        print(f"\n📊 데이터셋 분석:")

        try:
            import glob

            # 각 분할별 파일 수 확인
            splits = ['train', 'valid', 'test']
            total_images = 0
            total_labels = 0

            for split in splits:
                images_path = os.path.join(dataset_path, split, 'images')
                labels_path = os.path.join(dataset_path, split, 'labels')

                if os.path.exists(images_path):
                    images = len(glob.glob(os.path.join(images_path, '*')))
                    labels = len(glob.glob(os.path.join(labels_path, '*'))) if os.path.exists(labels_path) else 0

                    total_images += images
                    total_labels += labels

                    print(f"   {split:5}: {images:4}개 이미지, {labels:4}개 라벨")

            print(f"   총합: {total_images:4}개 이미지, {total_labels:4}개 라벨")

            # data.yaml 파일 확인
            yaml_path = os.path.join(dataset_path, 'data.yaml')
            if os.path.exists(yaml_path):
                print(f"   ✅ data.yaml 파일 존재")

                # yaml 내용 출력
                with open(yaml_path, 'r') as f:
                    yaml_content = f.read()
                print(f"\n📋 data.yaml 내용:")
                print(yaml_content)
            else:
                print(f"   ❌ data.yaml 파일 없음")

        except Exception as e:
            print(f"   분석 중 오류: {e}")

    def download_custom_dataset(self, dataset_url, output_dir=None, format="yolov8"):
        """커스텀 데이터셋 URL로 다운로드"""
        # URL에서 workspace, project, version 추출
        try:
            # URL 형식: https://universe.roboflow.com/workspace/project/dataset/version
            parts = dataset_url.strip('/').split('/')
            workspace = parts[-4]
            project = parts[-3]
            version = int(parts[-1])

            print(f"🔍 URL에서 추출된 정보:")
            print(f"   Workspace: {workspace}")
            print(f"   Project: {project}")
            print(f"   Version: {version}")

            return self.download_dataset(workspace, project, version, output_dir, format)

        except Exception as e:
            print(f"❌ URL 파싱 실패: {e}")
            print("올바른 URL 형식: https://universe.roboflow.com/workspace/project/dataset/version")
            return None

    def setup_dataset_for_yolov8(self, dataset_path):
        """YOLOv8 훈련을 위한 데이터셋 설정"""
        print("🔧 YOLOv8용 데이터셋 설정...")

        # 환경별 타겟 경로 설정
        try:
            import google.colab
            target_base = "/content/drive/MyDrive/datasets/fire"
        except ImportError:
            target_base = os.path.join(os.getcwd(), "datasets", "fire")

        try:
            # 기존 데이터 백업
            if os.path.exists(target_base):
                backup_path = f"{target_base}_backup"
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                shutil.move(target_base, backup_path)
                print(f"   기존 데이터를 {backup_path}로 백업")

            # 새 구조로 복사
            os.makedirs(target_base, exist_ok=True)

            splits = ['train', 'valid', 'test']
            for split in splits:
                src_images = os.path.join(dataset_path, split, 'images')
                src_labels = os.path.join(dataset_path, split, 'labels')

                dst_images = os.path.join(target_base, split, 'images')
                dst_labels = os.path.join(target_base, split, 'labels')

                if os.path.exists(src_images):
                    os.makedirs(os.path.dirname(dst_images), exist_ok=True)
                    shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
                    print(f"   ✅ {split} 이미지 복사 완료")

                if os.path.exists(src_labels):
                    os.makedirs(os.path.dirname(dst_labels), exist_ok=True)
                    shutil.copytree(src_labels, dst_labels, dirs_exist_ok=True)
                    print(f"   ✅ {split} 라벨 복사 완료")

            # data.yaml 복사
            src_yaml = os.path.join(dataset_path, 'data.yaml')
            dst_yaml = os.path.join(target_base, 'data.yaml')

            if os.path.exists(src_yaml):
                shutil.copy2(src_yaml, dst_yaml)
                print(f"   ✅ data.yaml 복사 완료")

                # data.yaml 경로 수정
                self._update_yaml_paths(dst_yaml, target_base)

            print(f"✅ 데이터셋 설정 완료: {target_base}")
            return target_base

        except Exception as e:
            print(f"❌ 데이터셋 설정 실패: {e}")
            return None

    def _update_yaml_paths(self, yaml_path, base_path):
        """data.yaml의 경로를 프로젝트 구조에 맞게 수정"""
        try:
            import yaml

            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            # 경로 업데이트
            data['train'] = os.path.join(base_path, 'train', 'images')
            data['val'] = os.path.join(base_path, 'valid', 'images')
            data['test'] = os.path.join(base_path, 'test', 'images')

            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)

            print(f"   ✅ data.yaml 경로 업데이트 완료")

        except Exception as e:
            print(f"   ⚠️ data.yaml 업데이트 실패: {e}")


def interactive_downloader():
    """대화형 다운로더"""
    downloader = RoboflowDatasetDownloader()

    print("🔥 Roboflow 화재/연기 데이터셋 다운로더")
    print("=" * 50)

    # 추천 데이터셋 목록 출력
    downloader.list_recommended_datasets()

    # 사용자 선택
    print("다운로드 옵션:")
    print("1. 추천 데이터셋 사용 (간편)")
    print("2. 커스텀 URL 입력")
    print("3. 직접 설정")

    choice = input("\n선택하세요 (1-3): ").strip()

    if choice == '1':
        # 추천 데이터셋
        dataset_keys = list(downloader.recommended_datasets.keys())
        print("\n추천 데이터셋:")
        for i, key in enumerate(dataset_keys, 1):
            print(f"{i}. {key}")

        try:
            selection = int(input(f"\n선택하세요 (1-{len(dataset_keys)}): ")) - 1
            if 0 <= selection < len(dataset_keys):
                dataset_key = dataset_keys[selection]
                dataset = downloader.download_dataset_by_key(dataset_key)

                if dataset:
                    # YOLOv8용 설정
                    setup_choice = input("\nYOLOv8 프로젝트 구조로 설정하시겠습니까? (y/n): ").lower()
                    if setup_choice == 'y':
                        downloader.setup_dataset_for_yolov8(dataset.location)

            else:
                print("잘못된 선택입니다.")
        except ValueError:
            print("숫자를 입력해주세요.")

    elif choice == '2':
        # 커스텀 URL
        url = input("\nRoboflow 데이터셋 URL을 입력하세요: ").strip()
        dataset = downloader.download_custom_dataset(url)

        if dataset:
            setup_choice = input("\nYOLOv8 프로젝트 구조로 설정하시겠습니까? (y/n): ").lower()
            if setup_choice == 'y':
                downloader.setup_dataset_for_yolov8(dataset.location)

    elif choice == '3':
        # 직접 설정
        workspace = input("\nWorkspace ID: ").strip()
        project = input("Project ID: ").strip()
        version = int(input("Version: ").strip())

        dataset = downloader.download_dataset(workspace, project, version)

        if dataset:
            setup_choice = input("\nYOLOv8 프로젝트 구조로 설정하시겠습니까? (y/n): ").lower()
            if setup_choice == 'y':
                downloader.setup_dataset_for_yolov8(dataset.location)

    else:
        print("잘못된 선택입니다.")


# 빠른 사용을 위한 함수들
def download_fire_dataset(api_key=None, dataset_key="fire-wrpgm"):
    """빠른 화재 데이터셋 다운로드"""
    downloader = RoboflowDatasetDownloader(api_key)
    dataset = downloader.download_dataset_by_key(dataset_key)

    if dataset:
        return downloader.setup_dataset_for_yolov8(dataset.location)
    return None


def download_large_fire_dataset(api_key=None):
    """큰 화재/연기 데이터셋 다운로드 (6391개 이미지)"""
    return download_fire_dataset(api_key, "fire-smoke-detection")


if __name__ == "__main__":
    interactive_downloader()