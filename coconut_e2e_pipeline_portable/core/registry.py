from pathlib import Path
import torch
from ultralytics import YOLO

TASK_DIRS = {
    "Aerial Tree Detection": "aerial tree detection",
    "Coconut Counting": "coconut counting",
    "Maturity & Variety Classification": "maturity variety classification",
    "Disease Detection": "disease classification",
}

class ModelRegistry:
    def __init__(self, models_root: Path):
        self.root = Path(models_root)
        self.tasks = list(TASK_DIRS.keys())

    def _files(self, task):
        folder = self.root / TASK_DIRS[task]
        if not folder.exists():
            raise FileNotFoundError(f"Missing model folder: {folder}")
        return sorted([p for p in folder.iterdir() if p.suffix.lower() in (".pt", ".pth")])

    def model_names(self, task):
        return [p.stem for p in self._files(task)]

    def _path(self, task, name):
        for p in self._files(task):
            if p.stem == name:
                return p
        raise FileNotFoundError(f"Model '{name}' not found for {task}")

    def load(self, task, name, device="auto"):
        path = self._path(task, name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if path.suffix.lower() == ".pt":
            model = YOLO(str(path))
            model.to(device)
            return {"backend": "ultralytics", "model": model, "device": device}

        if task == "Coconut Counting" and path.suffix.lower() == ".pth":
            from .faster_rcnn import load_fasterrcnn
            return {
                "backend": "fasterrcnn",
                "model": load_fasterrcnn(path, device),
                "device": device,
            }

        raise ValueError(f"Unsupported checkpoint: {path}")
