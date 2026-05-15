#!/usr/bin/env python3
# simple_validation.py - model_validator의 간편 진입점
"""
훈련된 모델을 빠르게 검증합니다.
상세 분석/혼동행렬/PR 분석이 필요하면 model_validator.ModelValidator를 직접 사용하세요.
"""

import logging

from model_validator import ModelValidator

logger = logging.getLogger(__name__)


def validate_model():
    validator = ModelValidator()
    if not validator.load_model():
        logger.info("먼저 모델을 훈련하세요: python main.py --mode train --epochs 50")
        return False

    results = validator.validate()
    return results is not None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    validate_model()
