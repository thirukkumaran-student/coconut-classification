from pathlib import Path
from typing import List

import torch
from ultralytics import YOLO

from core.faster_rcnn import load_fasterrcnn


class ModelRegistry:

    def __init__(self, models_dir):

        self.models_dir = Path(models_dir)

        self.task_dirs = {

            "Aerial Tree Detection":
                self.models_dir / "aerial tree detection",

            "Coconut Counting":
                self.models_dir / "coconut counting",

            "Maturity & Variety Classification":
                self.models_dir / "maturity variety classification",

            "Disease Detection":
                self.models_dir / "disease classification",
        }

    # =========================================================
    # TASKS
    # =========================================================

    @property
    def tasks(self) -> List[str]:

        return list(
            self.task_dirs.keys()
        )

    # =========================================================
    # MODEL NAMES
    # =========================================================

    def model_names(
        self,
        task: str
    ) -> List[str]:

        if task not in self.task_dirs:

            raise ValueError(
                f"Unknown task: {task}"
            )

        model_dir = self.task_dirs[task]

        if not model_dir.exists():

            return []

        models = []

        for file in model_dir.iterdir():

            if (
                file.is_file()
                and file.suffix.lower()
                in [".pt", ".pth"]
            ):

                models.append(
                    file.name
                )

        return sorted(models)

    # =========================================================
    # MODEL PATH
    # =========================================================

    def model_path(
        self,
        task: str,
        model_name: str
    ) -> Path:

        if task not in self.task_dirs:

            raise ValueError(
                f"Unknown task: {task}"
            )

        path = (
            self.task_dirs[task]
            / model_name
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Model not found:\n{path}"
            )

        return path

    # =========================================================
    # DEVICE
    # =========================================================

    @staticmethod
    def resolve_device(
        device: str
    ) -> str:

        if device == "auto":

            if torch.cuda.is_available():

                return "cuda"

            return "cpu"

        if (
            device == "cuda"
            and not torch.cuda.is_available()
        ):

            return "cpu"

        return device

    # =========================================================
    # LOAD MODEL
    # =========================================================

    def load(
        self,
        task: str,
        model_name: str,
        device: str = "auto"
    ):

        path = self.model_path(
            task,
            model_name
        )

        runtime_device = (
            self.resolve_device(device)
        )

        # =====================================================
        # FASTER R-CNN
        # =====================================================

        if path.suffix.lower() == ".pth":

            return load_fasterrcnn(
                str(path),
                runtime_device
            )

        # =====================================================
        # YOLO
        # =====================================================

        if path.suffix.lower() == ".pt":

            model = YOLO(
                str(path)
            )

            model.to(
                runtime_device
            )

            return model

        # =====================================================
        # UNSUPPORTED
        # =====================================================

        raise ValueError(
            f"Unsupported model format: {path}"
        )