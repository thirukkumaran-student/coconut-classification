import platform

import torch

try:
    import psutil
except ImportError:  # keep the dashboard usable without psutil
    psutil = None


def get_system_info():

    gpu = "None / CPU only"
    cuda = None

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cuda = getattr(torch.version, "cuda", None)

    ram_gb = None

    if psutil is not None:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    return {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": ram_gb,
        "gpu": gpu,
        "cuda": cuda,
        "torch": getattr(torch, "__version__", None),
        "python": platform.python_version(),
        "os": platform.platform(),
    }
