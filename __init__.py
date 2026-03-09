from .spectrum_node import SpectrumSDXL, SpectrumSDXLBatch

NODE_CLASS_MAPPINGS = {"SpectrumSDXL": SpectrumSDXL, "SpectrumSDXLBatch": SpectrumSDXLBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumSDXL": "Spectrum Adaptive Forecaster (SDXL)", "SpectrumSDXLBatch": "Spectrum Adaptive Forecaster Batch (SDXL)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]