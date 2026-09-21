import os
import platform
import psutil
import torch

def get_system_info():
    gpu = "None / CPU only"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": psutil.virtual_memory().total / (1024**3),
        "gpu": gpu,
    }
