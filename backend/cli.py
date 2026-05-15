# main.py - 메인 실행 파일
"""
YOLOv8 화재 및 연기 감지 프로젝트 메인 실행 파일
모든 기능을 통합하여 실행할 수 있는 인터페이스 제공
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 프로젝트 모듈 import
from config import Config, initialize_project
from dataset_manager import DatasetManager
from model_trainer import ModelTrainer
from model_validator import ModelValidator
from inference_engine import InferenceEngine
from visualization_utils import VisualizationUtils

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FireSmokeDetectionPipeline:
    """화재/연기 감지 파이프라인 클래스"""

    def __init__(self):
        self.config = Config()
        self.dataset_manager = DatasetManager()
        self.trainer = None
        self.validator = None
        self.inference_engine = None
        self.visualizer = VisualizationUtils()

        logger.info("🔥 YOLOv8 화재 및 연기 감지 시스템 초기화 완료")

    def setup_environment(self):
        """환경 설정"""
        logger.info("\n=== 1. 환경 설정 ===")
        return initialize_project()

    def prepare_dataset(self):
        """데이터셋 준비 및 검증"""
        logger.info("\n=== 2. 데이터셋 준비 ===")

        # 디렉토리 구조 생성
        self.dataset_manager.create_directory_structure()

        # 데이터셋 상태 확인
        status = self.dataset_manager.check_dataset_status()

        if status['has_data']:
            logger.info("✅ 데이터셋 발견!")

            # 데이터셋 검증
            is_valid = self.dataset_manager.validate_dataset_format()

            if is_valid:
                # 클래스 분포 분석
                class_dist = self.dataset_manager.analyze_class_distribution()

                # 샘플 시각화
                self.dataset_manager.visualize_dataset_samples()

                return True
            else:
                logger.info("❌ 데이터셋 형식에 문제가 있습니다.")
                return False
        else:
            logger.info("❌ 데이터셋이 없습니다. 데이터를 업로드해주세요.")
            return False

    def train_model(self, epochs=100, batch_size=16, model_size='yolov8n.pt'):
        """모델 훈련"""
        logger.info(f"\n=== 3. 모델 훈련 ({epochs} epochs) ===")

        # 훈련기 초기화
        self.trainer = ModelTrainer(model_size)

        if not self.trainer.load_model():
            logger.info("❌ 모델 로드 실패")
            return False

        # 모델 정보 출력
        self.trainer.get_model_info()

        # 훈련 실행
        results = self.trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            project_name='fire_smoke_detection'
        )

        if results:
            logger.info("✅ 모델 훈련 완료!")

            # 훈련 결과 시각화
            self.visualizer.plot_training_results()

            return True
        else:
            logger.info("❌ 모델 훈련 실패")
            return False

    def validate_model(self):
        """모델 검증"""
        logger.info("\n=== 4. 모델 검증 ===")

        # 검증기 초기화
        self.validator = ModelValidator()

        if not self.validator.load_model():
            logger.info("❌ 모델 로드 실패")
            return False

        # 검증 실행
        results = self.validator.validate()

        if results:
            logger.info("✅ 모델 검증 완료!")

            # 혼동 행렬 분석
            self.validator.confusion_matrix_analysis()

            # PR 분석
            self.validator.precision_recall_analysis()

            # 검증 샘플 시각화
            self.visualizer.plot_validation_samples()

            # 검증 보고서 저장
            self.validator.save_validation_report()

            return True
        else:
            logger.info("❌ 모델 검증 실패")
            return False

    def run_inference(self, source, inference_type='image'):
        """추론 실행"""
        logger.info(f"\n=== 5. 추론 실행 ({inference_type}) ===")

        # 추론 엔진 초기화
        self.inference_engine = InferenceEngine()

        if not self.inference_engine.load_model():
            logger.info("❌ 모델 로드 실패")
            return False

        try:
            if inference_type == 'image':
                results = self.inference_engine.predict_image(source)
                if results:
                    self.visualizer.plot_inference_results(results)

            elif inference_type == 'batch':
                results = self.inference_engine.predict_batch(source)
                if results:
                    self.visualizer.plot_batch_analysis(results)

            elif inference_type == 'video':
                results = self.inference_engine.predict_video(source,
                                                              output_path='output_video.mp4')

            elif inference_type == 'realtime':
                results = self.inference_engine.real_time_detection()

            else:
                logger.info(f"❌ 지원하지 않는 추론 타입: {inference_type}")
                return False

            if results:
                logger.info("✅ 추론 완료!")
                return True
            else:
                logger.info("❌ 추론 실패")
                return False

        except Exception as e:
            logger.info(f"❌ 추론 중 오류 발생: {e}")
            return False

    def run_full_pipeline(self, epochs=50, batch_size=16):
        """전체 파이프라인 실행"""
        logger.info("🚀 전체 파이프라인 실행 시작!")

        # 1. 환경 설정
        setup_result = self.setup_environment()
        if not setup_result['ultralytics_ok']:
            logger.info("❌ 환경 설정 실패")
            return False

        # 2. 데이터셋 준비 (Roboflow 다운로드 포함)
        if not self.prepare_dataset():
            logger.info("❌ 데이터셋 준비 실패")
            return False

        # 3. 모델 훈련
        if not self.train_model(epochs=epochs, batch_size=batch_size):
            logger.info("❌ 모델 훈련 실패")
            return False

        # 4. 모델 검증
        if not self.validate_model():
            logger.info("❌ 모델 검증 실패")
            return False

        logger.info("🎉 전체 파이프라인 실행 완료!")
        return True

    def interactive_mode(self):
        """대화형 모드"""
        logger.info("\n🤖 대화형 모드 시작")
        logger.info("사용 가능한 명령어:")
        logger.info("1. setup - 환경 설정")
        logger.info("2. dataset - 데이터셋 준비")
        logger.info("3. train - 모델 훈련")
        logger.info("4. validate - 모델 검증")
        logger.info("5. infer - 추론 실행")
        logger.info("6. viz - 시각화")
        logger.info("7. full - 전체 파이프라인")
        logger.info("8. quit - 종료")

        while True:
            try:
                command = input("\n명령어를 입력하세요: ").strip().lower()

                if command == '1' or command == 'setup':
                    self.setup_environment()

                elif command == '2' or command == 'dataset':
                    self.prepare_dataset()

                elif command == '3' or command == 'train':
                    epochs = input("에포크 수 (기본값: 50): ").strip()
                    epochs = int(epochs) if epochs else 50

                    batch_size = input("배치 크기 (기본값: 16): ").strip()
                    batch_size = int(batch_size) if batch_size else 16

                    self.train_model(epochs=epochs, batch_size=batch_size)

                elif command == '4' or command == 'validate':
                    self.validate_model()

                elif command == '5' or command == 'infer':
                    logger.info("추론 타입을 선택하세요:")
                    logger.info("1. image - 단일 이미지")
                    logger.info("2. batch - 배치 이미지")
                    logger.info("3. video - 비디오")
                    logger.info("4. realtime - 실시간")

                    infer_type = input("선택 (1-4): ").strip()

                    if infer_type == '1':
                        image_path = input("이미지 경로: ").strip()
                        self.run_inference(image_path, 'image')
                    elif infer_type == '2':
                        folder_path = input("폴더 경로: ").strip()
                        self.run_inference(folder_path, 'batch')
                    elif infer_type == '3':
                        video_path = input("비디오 경로: ").strip()
                        self.run_inference(video_path, 'video')
                    elif infer_type == '4':
                        self.run_inference(None, 'realtime')

                elif command == '6' or command == 'viz':
                    logger.info("시각화 옵션:")
                    logger.info("1. 훈련 결과")
                    logger.info("2. 검증 샘플")
                    logger.info("3. 클래스 분포")
                    logger.info("4. 감지 샘플")

                    viz_option = input("선택 (1-4): ").strip()

                    if viz_option == '1':
                        self.visualizer.plot_training_results()
                    elif viz_option == '2':
                        self.visualizer.plot_validation_samples()
                    elif viz_option == '3':
                        self.visualizer.plot_class_distribution()
                    elif viz_option == '4':
                        self.visualizer.plot_detection_samples()

                elif command == '7' or command == 'full':
                    epochs = input("에포크 수 (기본값: 50): ").strip()
                    epochs = int(epochs) if epochs else 50

                    self.run_full_pipeline(epochs=epochs)

                elif command == '8' or command == 'quit':
                    logger.info("👋 프로그램을 종료합니다.")
                    break

                else:
                    logger.info("❌ 올바르지 않은 명령어입니다.")

            except KeyboardInterrupt:
                logger.info("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                logger.info(f"❌ 오류 발생: {e}")

    def export_model(self, format='onnx'):
        """모델 내보내기"""
        logger.info(f"\n=== 모델 내보내기 ({format}) ===")

        if self.trainer is None:
            self.trainer = ModelTrainer()

        export_path = self.trainer.export_model(format)

        if export_path:
            logger.info(f"✅ 모델 내보내기 완료: {export_path}")
            return export_path
        else:
            logger.info("❌ 모델 내보내기 실패")
            return None


def create_argument_parser():
    """명령행 인수 파서 생성"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 화재 및 연기 감지 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py --mode setup                     # 환경 설정만
  python main.py --mode train --epochs 100       # 모델 훈련
  python main.py --mode infer --source image.jpg # 단일 이미지 추론
  python main.py --mode full --epochs 50         # 전체 파이프라인
  python main.py --interactive                   # 대화형 모드
        """
    )

    parser.add_argument('--mode', choices=['setup', 'dataset', 'train', 'validate', 'infer', 'full'],
                        help='실행 모드 선택')

    parser.add_argument('--interactive', action='store_true',
                        help='대화형 모드 실행')

    parser.add_argument('--epochs', type=int, default=50,
                        help='훈련 에포크 수 (기본값: 50)')

    parser.add_argument('--batch-size', type=int, default=16,
                        help='배치 크기 (기본값: 16)')

    parser.add_argument('--model-size', default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help='YOLO 모델 크기')

    parser.add_argument('--download', choices=['quick', 'large', 'interactive'],
                        help='Roboflow 데이터셋 다운로드 옵션')

    parser.add_argument('--api-key', type=str,
                        help='Roboflow API 키')

    parser.add_argument('--source', type=str,
                        help='추론용 소스 (이미지/폴더/비디오 경로)')

    parser.add_argument('--inference-type', choices=['image', 'batch', 'video', 'realtime'],
                        default='image', help='추론 타입')

    parser.add_argument('--confidence', type=float, default=0.5,
                        help='신뢰도 임계값 (기본값: 0.5)')

    parser.add_argument('--export', choices=['onnx', 'tensorrt', 'coreml', 'openvino'],
                        help='모델 내보내기 형식')

    return parser


def main():
    """메인 함수"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # 파이프라인 초기화
    pipeline = FireSmokeDetectionPipeline()

    # 대화형 모드
    if args.interactive:
        pipeline.interactive_mode()
        return

    # 모드별 실행
    if args.mode == 'setup':
        pipeline.setup_environment()

    elif args.mode == 'dataset':
        pipeline.setup_environment()
        pipeline.prepare_dataset()

    elif args.mode == 'train':
        pipeline.setup_environment()
        if pipeline.prepare_dataset():
            pipeline.train_model(
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_size=args.model_size
            )

    elif args.mode == 'validate':
        pipeline.validate_model()

    elif args.mode == 'infer':
        if not args.source:
            logger.info("❌ --source 인수가 필요합니다.")
            sys.exit(1)

        pipeline.run_inference(args.source, args.inference_type)

    elif args.mode == 'full':
        pipeline.run_full_pipeline(
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    # 모델 내보내기
    if args.export:
        pipeline.export_model(args.export)

    # 모드가 지정되지 않은 경우 대화형 모드
    if not args.mode:
        logger.info("모드가 지정되지 않았습니다. 대화형 모드를 시작합니다.")
        pipeline.interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 프로그램이 중단되었습니다.")
    except Exception as e:
        logger.info(f"❌ 예상치 못한 오류: {e}")
        sys.exit(1)