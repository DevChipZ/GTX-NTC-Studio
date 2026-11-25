# GTX-NTC Studio

**Standalone, Engine-Agnostic Neural Texture Compression & 4GB VRAM Optimizer for NVIDIA GTX GPUs.**

GTX-NTC Studio is a self-contained Windows application designed to reduce texture VRAM footprint and PCIe memory thrashing on 4GB VRAM hardware (such as the NVIDIA GTX 1650) without relying on upscalers or frame generation. It operates strictly on standard texture formats (`.png`, `.jpg`, `.tga`, `.bmp`, `.dds`), making it compatible with any DirectX 11/12 game pipeline (e.g., *Crimson Desert*, *God of War Ragnarök*, and modern AAA releases).

---

## Key Features

1. **Tab 1 — Neural Compress (`.ntc`)**
   * Compresses raw textures into an **INT8 Latent Grid + FP16 MLP** neural representation (`.ntc`).
   * Displays real-time **Original vs. Neural Decoded** side-by-side previews, compression ratios, and **PSNR (dB)** quality metrics.
2. **Tab 2 — Game Optimizer (`.dds`)**
   * Batch-converts input textures (`.png`, `.jpg`, `.tga`, `.dds`) into 4GB-VRAM-safe **BC1_UNORM (1024px cap)** `.dds` packs via Microsoft DirectXTex (`texconv.exe`).
   * Cuts texture memory bandwidth in half for native 1080p testing.
3. **Tab 3 — Live GTX Proof**
   * Launches the NVIDIA RTXNTC Renderer with `--dx12 --no-coopVec` to demonstrate real-time neural texture decompression on non-Tensor-Core GTX GPUs.

---

## Verified Local Benchmarks (GTX 1650 4GB)

| Input Batch | Raw Size | Compressed (`.ntc`) | Reduction | Quality (PSNR) |
| :--- | :---: | :---: | :---: | :---: |
| Test Texture 1 | 6,075.0 KB | 611.4 KB | **9.9x smaller** | **26.54 dB** |
| Test Texture 2 | 6,075.0 KB | 601.3 KB | **10.1x smaller** | **27.34 dB** |
| **Total Batch** | **11.87 MB** | **1.18 MB** | **10.0x reduction** | **~26.94 dB avg** |

---

## Quick Start (Standalone Windows Build)

1. Download **`GTX-NTC-Studio.exe`** from the [Releases](../../releases) page.
2. *(Optional, for Tab 2 `.dds` export)* Place Microsoft DirectXTex **`texconv.exe`** in the same folder as `GTX-NTC-Studio.exe`.
3. Launch **`GTX-NTC-Studio.exe`**, select any folder containing textures in **Tab 1**, and run either **Neural Compress (`.ntc`)** or **Game Optimizer (`.dds`)**.

---

## Tech Stack
* **GUI & Threading:** Python, CustomTkinter, Pillow
* **Compression Engine:** Custom INT8 Latent Grid + FP16 MLP (`ntc_engine.py`)
* **DirectX Texture Pipeline:** Microsoft DirectXTex (`texconv.exe`)
* **CI/CD:** Automated Windows `.exe` builds via GitHub Actions (PyInstaller)
