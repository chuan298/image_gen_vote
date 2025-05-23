import torch
import time
from typing import List
from loguru import logger

from validation.engine.metrics.alignment_scorer import ImageVSImageMetric, TextVSImageMetric
from validation.engine.metrics.quality_scorer import ImageQualityMetric


class ValidationEngine:
    """Class that handles all validation metrics"""

    def __init__(self, verbose: bool = False, device: str = "cuda") -> None:
        self.device = torch.device(device)
        torch.set_default_device(self.device)

        self._verbose = verbose
        self._image_quality_metric = ImageQualityMetric()
        self._text_vs_image_metric = TextVSImageMetric()

    def load_pipelines(self) -> None:
        """Function for loading all pipelines (metrics) that are used within the engine"""

        self._image_quality_metric.load_models()
        self._text_vs_image_metric.load_model("convnext_large_d", "laion2b_s26b_b102k_augreg")

    def unload_pipelines(self) -> None:
        """Function for unloading all pipelines (metrics) from the memory"""

        self._image_quality_metric.unload_models()
        self._text_vs_image_metric.unload_model()

    
