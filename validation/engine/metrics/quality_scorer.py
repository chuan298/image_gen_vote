import gc

import numpy as np
import sklearn
import torch
from huggingface_hub import hf_hub_download
from joblib import load
import time
from validation.engine.models.aethtetic_model import AestheticsPredictorModel
from validation.engine.models.quality_model import QualityClassifierModel
from validation.engine.utils.statistics_computation_utils import compute_mean, filter_outliers

from loguru import logger

class ImageQualityMetric:
    """Metric that measures the quality of the rendered images"""

    def __init__(self, verbose: bool = False) -> None:
        self._quality_classifier_model: QualityClassifierModel = QualityClassifierModel()
        self._aesthetics_predictor_model: AestheticsPredictorModel = AestheticsPredictorModel()
        self._polynomial_pipeline_model: sklearn.pipeline.Pipeline | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._verbose = verbose

        self._expected_pipeline_views = 16
        self._expected_pipeline_input_features = 32

    def load_models(
        self,
        repo_id: str = "404-Gen/validation",
        quality_classifier_model: str = "score_based_classifier_params.pth",
        aesthetics_predictor_model: str = "aesthetic_predictor.pth",
        polynomial_pipeline_model: str = "poly_fit.joblib",
    ) -> None:
        if self._polynomial_pipeline_model is not None:
             logger.info("Quality models already loaded.")
             return
        logger.info("Loading quality scoring models.")
        self._quality_classifier_model.load_model(repo_id, quality_classifier_model)
        self._aesthetics_predictor_model.load_model(repo_id, aesthetics_predictor_model)
        self._polynomial_pipeline_model = load(hf_hub_download(repo_id, polynomial_pipeline_model))
        # Cố gắng kiểm tra input features mong đợi (optional but good practice)
        try:
            poly_features_step = self._polynomial_pipeline_model.steps[0][1]
            if hasattr(poly_features_step, 'n_features_in_'):
                 self._expected_pipeline_input_features = poly_features_step.n_features_in_
                 if self._expected_pipeline_input_features != self._expected_pipeline_views * 2:
                      logger.warning(f"Pipeline expects {self._expected_pipeline_input_features} input features, "
                                     f"but expected {self._expected_pipeline_views * 2} based on view count. Check pipeline definition.")
                 else:
                      logger.info(f"Polynomial pipeline expects {self._expected_pipeline_input_features} input features.")
            else:
                 logger.warning("Could not determine expected features from pipeline step 0. Assuming 32.")
                 self._expected_pipeline_input_features = self._expected_pipeline_views * 2
        except Exception as e:
             logger.warning(f"Error inspecting pipeline for expected features: {e}. Assuming 32.")
             self._expected_pipeline_input_features = self._expected_pipeline_views * 2
        logger.info("Quality scoring models loaded.")

    def unload_models(self) -> None:
        """Function for unloading all models"""

        self._quality_classifier_model.unload_model()
        self._aesthetics_predictor_model.unload_model()
        self._polynomial_pipeline_model = None

        torch.cuda.empty_cache()
        gc.collect()

    def score_images_quality(
        self, images: list[torch.Tensor], mean_op: str = "mean", use_filter_outliers: bool = False
    ) -> float:
        """Function for computing quality score of the input data"""

        if self._quality_classifier_model is None:
            raise RuntimeError("Quality Classifier model has not been loaded!")
        elif self._aesthetics_predictor_model is None:
            raise RuntimeError("Aesthetic Predictor model has not been loaded!")
        elif self._polynomial_pipeline_model is None:
            raise RuntimeError("Polynomial pipeline model has not been loaded!")

        classifier_validator_predictions = self._quality_classifier_model.score(list(images)).squeeze()
        aesthetic_validator_predictions = self._aesthetics_predictor_model.score(list(images)).squeeze()
        X = np.column_stack((classifier_validator_predictions, aesthetic_validator_predictions))
        combined_score_v1 = self._polynomial_pipeline_model.predict(X.reshape(1, -1))
        final_scores = torch.tensor(combined_score_v1).squeeze().clip(max=1.0)

        if use_filter_outliers:
            final_scores = filter_outliers(torch.tensor(final_scores))
        final_score = compute_mean(final_scores, mean_op)

        return float(final_score)

    def score_images_quality_2d(
        self, images: list[torch.Tensor]
    ) -> list[float]:
        """Function for computing quality score of the input data"""

        if self._quality_classifier_model is None:
            raise RuntimeError("Quality Classifier model has not been loaded!")
        elif self._aesthetics_predictor_model is None:
            raise RuntimeError("Aesthetic Predictor model has not been loaded!")
        elif self._polynomial_pipeline_model is None:
            raise RuntimeError("Polynomial pipeline model has not been loaded!")

        cls_scores = self._quality_classifier_model.score(list(images)).squeeze()
        aes_scores = self._aesthetics_predictor_model.score(list(images)).squeeze()


        cls_scores = cls_scores.detach().cpu().numpy() if torch.is_tensor(cls_scores) else np.array(cls_scores)
        aes_scores = aes_scores.detach().cpu().numpy() if torch.is_tensor(aes_scores) else np.array(aes_scores)

        # Tính trung bình từng cặp
        final_scores = ((cls_scores + aes_scores) / 2).tolist()

        return final_scores

   