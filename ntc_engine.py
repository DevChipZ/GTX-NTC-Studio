import io
import math
import numpy as np
from PIL import Image


class GTXNeuralCompressor:
    """
    Lightweight GTX-compatible Neural Texture Compression (NTC):
    - 2D Latent Feature Grid (quantized to INT8)
    - Tiny 2-layer MLP decoder (quantized to FP16)
    Runs without RTX Tensor Cores.
    """

    def __init__(self, latent_dim=6, hidden_dim=16, grid_div=4):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.grid_div = grid_div

    def _sample_grid(self, grid, ys, xs, H, W):
        gh, gw, _ = grid.shape
        gy = ys * (gh - 1) / max(1, H - 1)
        gx = xs * (gw - 1) / max(1, W - 1)

        y0 = np.floor(gy).astype(np.int32)
        x0 = np.floor(gx).astype(np.int32)
        y1 = np.minimum(y0 + 1, gh - 1)
        x1 = np.minimum(x0 + 1, gw - 1)

        wy = (gy - y0)[:, None]
        wx = (gx - x0)[:, None]

        w00 = (1.0 - wy) * (1.0 - wx)
        w01 = (1.0 - wy) * wx
        w10 = wy * (1.0 - wx)
        w11 = wy * wx

        f = (
            w00 * grid[y0, x0]
            + w01 * grid[y0, x1]
            + w10 * grid[y1, x0]
            + w11 * grid[y1, x1]
        )
        return f, (y0, x0, y1, x1, w00, w01, w10, w11)

    def compress_image(self, image_path, output_ntc_path, steps=180, batch_size=16384, progress_cb=None):
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        H, W, _ = arr.shape
        raw_bytes = H * W * 3

        gh = max(4, H // self.grid_div)
        gw = max(4, W // self.grid_div)

        rng = np.random.default_rng(42)
        grid = (rng.standard_normal((gh, gw, self.latent_dim)) * 0.1).astype(np.float32)
        W1 = (rng.standard_normal((self.latent_dim, self.hidden_dim)) * np.sqrt(2.0 / self.latent_dim)).astype(np.float32)
        b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        W2 = (rng.standard_normal((self.hidden_dim, 3)) * np.sqrt(2.0 / self.hidden_dim)).astype(np.float32)
        b2 = np.full(3, 0.5, dtype=np.float32)

        # Adam optimizer state
        params = [grid, W1, b1, W2, b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        lr = 0.03
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        total_pixels = H * W
        flat_target = arr.reshape(-1, 3)

        for step in range(1, steps + 1):
            idx = rng.integers(0, total_pixels, size=min(batch_size, total_pixels))
            ys = idx // W
            xs = idx % W
            target = flat_target[idx]

            # Forward pass
            f, cache = self._sample_grid(grid, ys, xs, H, W)
            z1 = f @ W1 + b1
            h1 = np.maximum(0.0, z1)
            pred = h1 @ W2 + b2

            diff = pred - target
            d_out = (2.0 / len(idx)) * diff

            # Backward pass
            dW2 = h1.T @ d_out
            db2 = d_out.sum(axis=0)
            dh1 = d_out @ W2.T
            dh1[z1 <= 0.0] = 0.0
            dW1 = f.T @ dh1
            db1 = dh1.sum(axis=0)
            df = dh1 @ W1.T

            y0, x0, y1, x1, w00, w01, w10, w11 = cache
            d_grid = np.zeros_like(grid)
            np.add.at(d_grid, (y0, x0), df * w00)
            np.add.at(d_grid, (y0, x1), df * w01)
            np.add.at(d_grid, (y1, x0), df * w10)
            np.add.at(d_grid, (y1, x1), df * w11)

            grads = [d_grid, dW1, db1, dW2, db2]
            for i in range(len(params)):
                m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i]
                v[i] = beta2 * v[i] + (1.0 - beta2) * (grads[i] ** 2)
                m_hat = m[i] / (1.0 - beta1 ** step)
                v_hat = v[i] / (1.0 - beta2 ** step)
                params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if progress_cb and step % 45 == 0:
                mse = float(np.mean(diff ** 2))
                psnr = -10.0 * math.log10(max(mse, 1e-10))
                progress_cb(step, steps, psnr)

        # Quantize Latent Grid -> INT8, MLP -> FP16
        g_min = float(grid.min())
        g_max = float(grid.max())
        scale = float((g_max - g_min) / 255.0 + 1e-8)
        q_grid = np.clip(np.round((grid - g_min) / scale), 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            meta=np.array([H, W], dtype=np.int32),
            quant=np.array([g_min, scale], dtype=np.float32),
            q_grid=q_grid,
            W1=W1.astype(np.float16),
            b1=b1.astype(np.float16),
            W2=W2.astype(np.float16),
            b2=b2.astype(np.float16),
        )
        ntc_data = buf.getvalue()
        with open(output_ntc_path, "wb") as f_out:
            f_out.write(ntc_data)

        # Decode back to verify exact PSNR
        recon_img = self.decode_ntc(output_ntc_path)
        recon_arr = np.array(recon_img, dtype=np.float32) / 255.0
        final_mse = float(np.mean((recon_arr - arr) ** 2))
        final_psnr = -10.0 * math.log10(max(final_mse, 1e-10))

        return {
            "raw_bytes": raw_bytes,
            "ntc_bytes": len(ntc_data),
            "ratio": raw_bytes / max(1, len(ntc_data)),
            "psnr": final_psnr,
            "recon_image": recon_img,
        }

    def decode_ntc(self, ntc_path):
        with open(ntc_path, "rb") as f_in:
            data = np.load(io.BytesIO(f_in.read()))

        H, W = int(data["meta"][0]), int(data["meta"][1])
        g_min, scale = float(data["quant"][0]), float(data["quant"][1])
        grid = data["q_grid"].astype(np.float32) * scale + g_min
        W1 = data["W1"].astype(np.float32)
        b1 = data["b1"].astype(np.float32)
        W2 = data["W2"].astype(np.float32)
        b2 = data["b2"].astype(np.float32)

        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        ys = ys.ravel()
        xs = xs.ravel()

        out = np.empty((H * W, 3), dtype=np.float32)
        chunk = 65536
        for start in range(0, H * W, chunk):
            end = min(start + chunk, H * W)
            f, _ = self._sample_grid(grid, ys[start:end], xs[start:end], H, W)
            h1 = np.maximum(0.0, f @ W1 + b1)
            out[start:end] = np.clip(h1 @ W2 + b2, 0.0, 1.0)

        rgb = (out.reshape(H, W, 3) * 255.0).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
