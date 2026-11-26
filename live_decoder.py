import io
import os
import sys
import time
import numpy as np
import glfw
import moderngl

VERTEX_SHADER = """
#version 330 core
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_grid0;
uniform sampler2D u_grid1;
uniform vec2 u_grid_size;
uniform float u_gmin;
uniform float u_scale255;

uniform float u_W1[96];  // 6 x 16
uniform float u_b1[16];
uniform float u_W2[48];  // 16 x 3
uniform float u_b2[3];

uniform float u_zoom;
uniform vec2 u_pan;

void main() {
    vec2 uv = (v_uv - 0.5) / u_zoom + 0.5 + u_pan;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        fragColor = vec4(0.08, 0.08, 0.10, 1.0);
        return;
    }

    // Flip Y to match PIL image top-left origin and align texel centers
    vec2 img_uv = vec2(uv.x, 1.0 - uv.y);
    vec2 sample_uv = (img_uv * (u_grid_size - 1.0) + 0.5) / u_grid_size;

    vec3 c0 = texture(u_grid0, sample_uv).rgb * u_scale255 + u_gmin;
    vec3 c1 = texture(u_grid1, sample_uv).rgb * u_scale255 + u_gmin;

    float f[6];
    f[0] = c0.r; f[1] = c0.g; f[2] = c0.b;
    f[3] = c1.r; f[4] = c1.g; f[5] = c1.b;

    // Layer 1: 6 -> 16 + ReLU
    float h[16];
    for (int j = 0; j < 16; j++) {
        float acc = u_b1[j];
        for (int i = 0; i < 6; i++) {
            acc += f[i] * u_W1[i * 16 + j];
        }
        h[j] = max(0.0, acc);
    }

    // Layer 2: 16 -> 3
    vec3 rgb;
    for (int k = 0; k < 3; k++) {
        float acc = u_b2[k];
        for (int j = 0; j < 16; j++) {
            acc += h[j] * u_W2[j * 3 + k];
        }
        rgb[k] = clamp(acc, 0.0, 1.0);
    }

    fragColor = vec4(rgb, 1.0);
}
"""


def run_live_decoder(ntc_path: str):
    if not os.path.exists(ntc_path):
        raise FileNotFoundError(f"NTC file not found: {ntc_path}")

    with open(ntc_path, "rb") as f_in:
        data = np.load(io.BytesIO(f_in.read()))

    H, W = int(data["meta"][0]), int(data["meta"][1])
    g_min, scale = float(data["quant"][0]), float(data["quant"][1])
    q_grid = data["q_grid"]  # (gh, gw, 6) uint8
    gh, gw, _ = q_grid.shape

    W1 = data["W1"].astype(np.float32).ravel()
    b1 = data["b1"].astype(np.float32).ravel()
    W2 = data["W2"].astype(np.float32).ravel()
    b2 = data["b2"].astype(np.float32).ravel()

    # Exact GPU VRAM footprint (INT8 latent grid + FP16 MLP weights)
    vram_kb = (q_grid.nbytes + data["W1"].nbytes + data["b1"].nbytes + data["W2"].nbytes + data["b2"].nbytes) / 1024.0
    raw_kb = (H * W * 3) / 1024.0

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW.")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    win_w, win_h = min(1280, max(640, W)), min(720, max(480, H))
    window = glfw.create_window(win_w, win_h, "GTX-NTC Live GPU Decoder", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create OpenGL 3.3 window.")

    glfw.make_context_current(window)
    glfw.swap_interval(0)  # Uncapped FPS for true GPU decode benchmark

    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

    quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
    vbo = ctx.buffer(quad.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, "in_pos")

    # Upload INT8 latent grid as two hardware-filtered RGB8 textures
    tex0 = ctx.texture((gw, gh), 3, np.ascontiguousarray(q_grid[:, :, 0:3]).tobytes())
    tex1 = ctx.texture((gw, gh), 3, np.ascontiguousarray(q_grid[:, :, 3:6]).tobytes())
    for t in (tex0, tex1):
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = False
        t.repeat_y = False

    tex0.use(location=0)
    tex1.use(location=1)

    prog["u_grid0"].value = 0
    prog["u_grid1"].value = 1
    prog["u_grid_size"].value = (float(gw), float(gh))
    prog["u_gmin"].value = g_min
    prog["u_scale255"].value = scale * 255.0
    prog["u_W1"].value = tuple(W1.tolist())
    prog["u_b1"].value = tuple(b1.tolist())
    prog["u_W2"].value = tuple(W2.tolist())
    prog["u_b2"].value = tuple(b2.tolist())

    state = {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "drag": False, "last": (0.0, 0.0)}

    def on_scroll(_win, _xoff, yoff):
        state["zoom"] = max(0.5, min(12.0, state["zoom"] * (1.15 if yoff > 0 else 0.87)))

    def on_mouse_btn(win, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            state["drag"] = action == glfw.PRESS
            state["last"] = glfw.get_cursor_pos(win)

    def on_cursor(win, xpos, ypos):
        if state["drag"]:
            lx, ly = state["last"]
            w, h = glfw.get_window_size(win)
            state["pan_x"] -= (xpos - lx) / max(1, w) / state["zoom"]
            state["pan_y"] += (ypos - ly) / max(1, h) / state["zoom"]
            state["last"] = (xpos, ypos)

    def on_key(win, key, _sc, action, _mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_R:
                state["zoom"], state["pan_x"], state["pan_y"] = 1.0, 0.0, 0.0

    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_mouse_button_callback(window, on_mouse_btn)
    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_key_callback(window, on_key)

    fname = os.path.basename(ntc_path)
    frames, t0 = 0, time.perf_counter()

    while not glfw.window_should_close(window):
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        ctx.viewport = (0, 0, fb_w, fb_h)
        ctx.clear(0.08, 0.08, 0.10)

        prog["u_zoom"].value = state["zoom"]
        prog["u_pan"].value = (state["pan_x"], state["pan_y"])
        vao.render(moderngl.TRIANGLE_STRIP)

        glfw.swap_buffers(window)
        glfw.poll_events()

        frames += 1
        dt = time.perf_counter() - t0
        if dt >= 0.25:
            fps = frames / dt
            ms = (dt / frames) * 1000.0
            glfw.set_window_title(
                window,
                f"GTX-NTC Live GPU | {fname} ({W}x{H}) | {fps:.0f} FPS ({ms:.2f} ms) | "
                f"VRAM: {vram_kb:.1f} KB vs {raw_kb:.0f} KB Raw | Scroll:Zoom Drag:Pan R:Reset",
            )
            frames, t0 = 0, time.perf_counter()

    glfw.terminate()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run_live_decoder(sys.argv[1])
