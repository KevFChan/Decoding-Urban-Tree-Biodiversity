# Decoding Urban Tree Biodiversity Using Transformer-Based Time Series Models

Researching the use of **transformer-based time-series models** (e.g., **PatchTST**) to classify **urban tree species** from high-resolution satellite imagery and model **multi-year urban ecological dynamics**.  
Work conducted in collaboration with **Dr. Juwon Kong (Yale School of the Environment)**.

---

## 🌳 Project Overview
This project aims to leverage high-resolution remote sensing and advanced deep learning architectures to classify tree species within urban environments and track biodiversity change over time. By integrating multi-sensor satellite imagery (e.g., Landsat ARD, Sentinel-2, Planet Fusion) with temporal modeling approaches, we seek to enable scalable and accurate ecological monitoring for urban planning and climate-resilience strategy development.

---

## ⚙️ Code Usage & Configuration

At the beginning of the main script for the data processing pipeline, the following parameters must be defined:

```python
script_working_dir = "/path/to/your/python/scripts"
data_input_dir     = "/path/to/SDC500_and_Landsat_inputs"
data_output_dir    = "/path/to/output_directory"
