import os
import sys
import glob
import webbrowser
import subprocess
import threading
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox
from ntc_engine import GTXNeuralCompressor

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
TEXCONV_PATH = os.path.join(APP_DIR, "texconv.exe")


class GTXNTCStudio(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GTX-NTC Studio | 4GB VRAM Neural Texture Pipeline")
        self.geometry("920x660")
        self.compressor = GTXNeuralCompressor()
        ctk.CTkLabel(
            self, text="GTX-NTC Studio", font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=(12, 2))
        ctk.CTkLabel(
            self,
            text="Neural Texture Compression (INT8 Latent Grid + FP16 MLP) & 4GB VRAM Optimizer",
            text_color="gray",
        ).pack(pady=(0, 8))

        self.tabs = ctk.CTkTabview(self, width=880, height=560)
        self.tabs.pack(padx=16, pady=8, fill="both", expand=True)

        self.tab_compress = self.tabs.add("1. Neural Compress (.ntc)")
        self.tab_game = self.tabs.add("2. Game Optimizer (.dds)")
        self.tab_live = self.tabs.add("3. Live GTX Proof")

        self.build_compress_tab()
        self.build_game_tab()
        self.build_live_tab()

    # ---------------- TAB 1: NEURAL COMPRESS + PREVIEW ----------------
    def build_compress_tab(self):
        top = ctk.CTkFrame(self.tab_compress)
        top.pack(fill="x", padx=10, pady=8)

        self.in_dir = ctk.StringVar(value=APP_DIR)
        self.out_dir = ctk.StringVar(value=os.path.join(APP_DIR, "ntc_output"))

        ctk.CTkLabel(top, text="Input Textures:").grid(row=0, column=0, padx=10, pady=6, sticky="w")
        ctk.CTkEntry(top, textvariable=self.in_dir, width=540).grid(row=0, column=1, padx=5)
        ctk.CTkButton(top, text="Browse", width=85, command=lambda: self.pick_dir(self.in_dir)).grid(row=0, column=2, padx=8)

        ctk.CTkLabel(top, text="Output Folder:").grid(row=1, column=0, padx=10, pady=6, sticky="w")
        ctk.CTkEntry(top, textvariable=self.out_dir, width=540).grid(row=1, column=1, padx=5)
        ctk.CTkButton(top, text="Browse", width=85, command=lambda: self.pick_dir(self.out_dir)).grid(row=1, column=2, padx=8)

        self.btn_compress = ctk.CTkButton(
            self.tab_compress,
            text="Train & Compress to .ntc (INT8 Grid + FP16 MLP)",
            height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.start_compression,
        )
        self.btn_compress.pack(pady=8)

        # Side-by-side preview
        preview_frame = ctk.CTkFrame(self.tab_compress)
        preview_frame.pack(fill="x", padx=10, pady=4)
        preview_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(preview_frame, text="Original Texture", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=4)
        ctk.CTkLabel(preview_frame, text="Neural Decoded (.ntc)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, pady=4)

        self.lbl_orig = ctk.CTkLabel(preview_frame, text="[Preview appears here]", width=220, height=160, fg_color="gray20")
        self.lbl_orig.grid(row=1, column=0, padx=10, pady=6)
        self.lbl_recon = ctk.CTkLabel(preview_frame, text="[Decoded appears here]", width=220, height=160, fg_color="gray20")
        self.lbl_recon.grid(row=1, column=1, padx=10, pady=6)

        self.log_box = ctk.CTkTextbox(self.tab_compress, height=130, font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=6)

    def log(self, msg):
        self.after(0, lambda: (self.log_box.insert("end", msg + "\n"), self.log_box.see("end")))

    def update_previews(self, orig_img: Image.Image, recon_img: Image.Image):
        ctk_orig = ctk.CTkImage(light_image=orig_img, dark_image=orig_img, size=(220, 160))
        ctk_recon = ctk.CTkImage(light_image=recon_img, dark_image=recon_img, size=(220, 160))
        self.lbl_orig.configure(image=ctk_orig, text="")
        self.lbl_recon.configure(image=ctk_recon, text="")
        self._img_refs = (ctk_orig, ctk_recon)

    def pick_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def start_compression(self):
        self.btn_compress.configure(state="disabled")
        self.log_box.delete("1.0", "end")
        threading.Thread(target=self.run_compression, daemon=True).start()

    def run_compression(self):
        src, dst = self.in_dir.get(), self.out_dir.get()
        os.makedirs(dst, exist_ok=True)
        files = [
            f for f in glob.glob(os.path.join(src, "*.*"))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tga", ".bmp"))
        ]
        if not files:
            self.log("[!] No images found. Browse to a folder containing .png/.jpg textures.")
            self.after(0, lambda: self.btn_compress.configure(state="normal"))
            return

        total_raw, total_ntc = 0, 0
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            self.log(f"[...] Training neural weights for {name}...")
            out_path = os.path.join(dst, f"{name}.ntc")
            res = self.compressor.compress_image(f, out_path)

            total_raw += res["raw_bytes"]
            total_ntc += res["ntc_bytes"]
            self.log(
                f"[OK] {name} | Raw: {res['raw_bytes']/1024:.1f} KB -> .ntc: {res['ntc_bytes']/1024:.1f} KB "
                f"({res['ratio']:.1f}x smaller) | PSNR: {res['psnr']:.2f} dB"
            )
            orig_img = Image.open(f).convert("RGB")
            self.after(0, lambda o=orig_img, r=res["recon_image"]: self.update_previews(o, r))

        overall = total_raw / max(1, total_ntc)
        self.log("-" * 68)
        self.log(f"DONE | Total Raw: {total_raw/1048576:.2f} MB -> .ntc: {total_ntc/1048576:.2f} MB ({overall:.1f}x reduction)")
        self.after(0, lambda: self.btn_compress.configure(state="normal"))

    # ---------------- TAB 2: GAME OPTIMIZER ----------------
    def build_game_tab(self):
        ctk.CTkLabel(
            self.tab_game,
            text="Generate 4GB-VRAM-Safe .DDS Pack (BC1 + 1024px Cap)",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(30, 6))
        ctk.CTkLabel(
            self.tab_game,
            text="Cuts VRAM bandwidth in half for Native testing (No Upscaler / No Frame Gen).",
            text_color="gray",
        ).pack(pady=(0, 18))
        ctk.CTkButton(
            self.tab_game,
            text="Convert Input Folder to 4GB-Safe .DDS",
            height=44,
            width=360,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.run_texconv,
        ).pack(pady=10)
        self.dds_status = ctk.CTkLabel(self.tab_game, text="Status: Ready", text_color="lightgreen")
        self.dds_status.pack(pady=12)

    def run_texconv(self):
        if not os.path.exists(TEXCONV_PATH):
            messagebox.showwarning(
                "texconv.exe Missing",
                "Place Microsoft DirectXTex texconv.exe in the same folder as GTX-NTC-Studio.exe.",
            )
            return
        dds_out = os.path.join(self.out_dir.get(), "game_dds_4gb")
        os.makedirs(dds_out, exist_ok=True)
        files = [
            f for f in glob.glob(os.path.join(self.in_dir.get(), "*.*"))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tga"))
        ]
        if not files:
            messagebox.showwarning("No Textures", "Select an Input Textures folder in Tab 1 first.")
            return
        self.dds_status.configure(text="Converting to BC1 .DDS (1024px cap)...", text_color="yellow")
        self.update()
        for f in files:
            subprocess.run(
                [TEXCONV_PATH, "-f", "BC1_UNORM", "-w", "1024", "-h", "1024", "-y", "-o", dds_out, f],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self.dds_status.configure(text=f"Done! Saved {len(files)} 4GB-safe .dds files to {dds_out}", text_color="lightgreen")

    # ---------------- TAB 3: LIVE GTX PROOF ----------------
    def build_live_tab(self):
        ctk.CTkLabel(
            self.tab_live,
            text="Real-Time GTX Neural Texture Renderer (Native 1080p)",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(35, 8))
        ctk.CTkLabel(
            self.tab_live,
            text="Launches NVIDIA RTXNTC Renderer with --dx12 --no-coopVec\nto prove live neural texture decompression on non-Tensor-Core GTX GPUs.",
            text_color="gray",
        ).pack(pady=(0, 22))
        ctk.CTkButton(
            self.tab_live,
            text="Launch Local RTXNTC Renderer",
            height=46,
            width=330,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.launch_renderer,
        ).pack(pady=8)
        ctk.CTkButton(
            self.tab_live,
            text="Download NVIDIA RTXNTC SDK (v0.10.1b)",
            height=36,
            width=330,
            fg_color="gray30",
            command=lambda: webbrowser.open("https://github.com/NVIDIA-RTX/RTXNTC/releases"),
        ).pack(pady=8)

    def launch_renderer(self):
        matches = glob.glob(os.path.join(APP_DIR, "**", "*Renderer*.exe"), recursive=True)
        if not matches:
            messagebox.showinfo(
                "Renderer Not Found",
                "Unzip RTXNTC_windows_x64_v0.10.1b.zip into this app's folder to launch live.",
            )
            return
        subprocess.Popen([matches[0], "--dx12", "--no-coopVec"], cwd=os.path.dirname(matches[0]))


if __name__ == "__main__":
    GTXNTCStudio().mainloop()
