from .src.spectrum_node import SpectrumSDXL
from .src.calibrated_spectrum_node import SpectrumSDXLCalibrated


NODE_CLASS_MAPPINGS = {"SpectrumSDXL": SpectrumSDXL, "CalibratedSpectrumSDXL": SpectrumSDXLCalibrated}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumSDXL": "Spectrum Adaptive Forecaster (SDXL)", "CalibratedSpectrumSDXL": "Calibrated Spectrum Adaptive Forecaster (SDXL)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]