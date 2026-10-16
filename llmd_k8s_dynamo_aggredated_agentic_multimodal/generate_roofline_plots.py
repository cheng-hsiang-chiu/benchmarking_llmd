#!/usr/bin/env python3
"""Render publication-quality Hardware Roofline and Serving Roofline Pareto PNGs using PyCairo + PangoCairo."""
import json
import math
from pathlib import Path
import cairo
import gi
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo

DATA_PATH = Path("/usr/local/google/home/cchenghsiang/.gemini/jetski/brain/ee3b9add-1d9b-4811-9e20-74e27f6f95ca/scratch/mmshopbench_plot_data.json")
OUT_DIR = Path("/usr/local/google/home/cchenghsiang/.gemini/jetski/brain/ee3b9add-1d9b-4811-9e20-74e27f6f95ca")

# Hardware constants for 1 Replica = TP=2 NVIDIA H200 SXM (141GB HBM3e each)
NUM_REPLICAS = 8
GPUS_PER_REPLICA = 2
TOTAL_GPUS = NUM_REPLICAS * GPUS_PER_REPLICA  # 16 H200 GPUs

PEAK_TFLOPS_PER_GPU = 989.0
PEAK_BW_TBPS_PER_GPU = 4.8

REPLICA_PEAK_TFLOPS = GPUS_PER_REPLICA * PEAK_TFLOPS_PER_GPU  # 1,978 TFLOPS
REPLICA_PEAK_BW_TBPS = GPUS_PER_REPLICA * PEAK_BW_TBPS_PER_GPU  # 9.6 TB/s
RIDGE_AI = REPLICA_PEAK_TFLOPS / REPLICA_PEAK_BW_TBPS          # ~206.04 FLOPs/Byte

# Qwen3-VL-32B-Instruct model constants (BF16)
N_PARAMS_LLM = 32.8e9
WEIGHT_BYTES_REPLICA = N_PARAMS_LLM * 2.0     # 65.6 GB
GFLOPS_PER_DECODE_TOK = 2.0 * 32.8 + 1.0      # 66.6 GFLOPs/tok
GFLOPS_PER_PREFILL_TOK = 2.0 * 32.8 + 2.9     # 68.5 GFLOPs/tok
KV_BYTES_PER_TOK = 64 * 2 * 8 * 128 * 2       # 262,144 Bytes/token
AVG_CTX_LEN = 768.7


def hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def compute_roofline_metrics(entry: dict) -> dict:
    out_tps_cluster = entry["output_tok_per_sec_mean"]
    tot_tps_cluster = entry["total_tok_per_sec_mean"]
    in_tps_cluster = tot_tps_cluster - out_tps_cluster
    itl_ms = entry["itl_p50_ms_mean"]
    hit_rate = entry["prefix_cache_hit_pct_mean"] / 100.0
    miss_rate = 1.0 - hit_rate

    r_out = out_tps_cluster / NUM_REPLICAS
    r_in = in_tps_cluster / NUM_REPLICAS
    r_in_uncached = miss_rate * r_in

    step_freq = 1000.0 / itl_ms
    b_dec = r_out / step_freq

    phys_tflops = (GFLOPS_PER_DECODE_TOK * r_out + GFLOPS_PER_PREFILL_TOK * r_in_uncached) / 1000.0
    eff_tflops = (GFLOPS_PER_DECODE_TOK * r_out + GFLOPS_PER_PREFILL_TOK * r_in) / 1000.0

    kv_read_per_step_gb = (b_dec * AVG_CTX_LEN * KV_BYTES_PER_TOK) / 1e9
    weight_and_decode_kv_gbps = step_freq * (WEIGHT_BYTES_REPLICA / 1e9 + kv_read_per_step_gb)
    prefill_kv_read_gbps = (r_in * KV_BYTES_PER_TOK) / 1e9
    hbm_gbps = weight_and_decode_kv_gbps + prefill_kv_read_gbps
    hbm_tbps = hbm_gbps / 1000.0

    phys_ai = phys_tflops / hbm_tbps
    eff_ai = eff_tflops / hbm_tbps

    interactivity = 1000.0 / itl_ms
    out_tps_per_gpu = out_tps_cluster / TOTAL_GPUS
    tot_tps_per_gpu = tot_tps_cluster / TOTAL_GPUS

    return {
        "c": entry["concurrency"],
        "b_dec": b_dec,
        "hit_pct": entry["prefix_cache_hit_pct_mean"],
        "miss_pct": miss_rate * 100.0,
        "phys_ai": phys_ai,
        "phys_tflops": phys_tflops,
        "eff_ai": eff_ai,
        "eff_tflops": eff_tflops,
        "hbm_tbps": hbm_tbps,
        "interactivity": interactivity,
        "out_tps_per_gpu": out_tps_per_gpu,
        "tot_tps_per_gpu": tot_tps_per_gpu,
        "out_tps_per_gpu_std": entry["output_tok_per_sec_std"] / TOTAL_GPUS,
        "tot_tps_per_gpu_std": entry["total_tok_per_sec_std"] / TOTAL_GPUS,
    }


def draw_text(ctx, text: str, x: float, y: float, font_str: str, color=(0, 0, 0),
              ha="center", va="center", angle_deg=0.0):
    ctx.save()
    ctx.translate(x, y)
    if angle_deg != 0.0:
        ctx.rotate(math.radians(angle_deg))
    layout = PangoCairo.create_layout(ctx)
    desc = Pango.FontDescription(font_str)
    layout.set_font_description(desc)
    layout.set_text(text, -1)
    ink_rect, logical_rect = layout.get_pixel_extents()
    w = logical_rect.width
    h = logical_rect.height
    dx = 0.0 if ha == "left" else (-w / 2.0 if ha == "center" else -w)
    dy = 0.0 if va == "top" else (-h / 2.0 if va == "center" else -h)
    ctx.move_to(dx, dy)
    ctx.set_source_rgb(*color)
    PangoCairo.show_layout(ctx, layout)
    ctx.restore()
    return w, h


def draw_leader(ctx, x0: float, y0: float, x1: float, y1: float, color: tuple):
    ctx.save()
    ctx.set_source_rgba(*color, 0.75)
    ctx.set_line_width(2.0)
    ctx.move_to(x0, y0)
    ctx.line_to(x1, y1)
    ctx.stroke()
    ctx.restore()


def draw_marker(ctx, shape: str, x: float, y: float, size: float, color: tuple):
    ctx.save()
    ctx.set_source_rgb(*color)
    if shape == "o":
        ctx.arc(x, y, size, 0, 2 * math.pi)
        ctx.fill()
    elif shape == "s":
        ctx.rectangle(x - size, y - size, 2 * size, 2 * size)
        ctx.fill()
    elif shape == "^":
        ctx.move_to(x, y - size * 1.18)
        ctx.line_to(x - size * 1.12, y + size * 0.95)
        ctx.line_to(x + size * 1.12, y + size * 0.95)
        ctx.close_path()
        ctx.fill()
    elif shape == "d":
        ctx.move_to(x, y - size * 1.25)
        ctx.line_to(x - size * 1.0, y)
        ctx.line_to(x, y + size * 1.25)
        ctx.line_to(x + size * 1.0, y)
        ctx.close_path()
        ctx.fill()
    ctx.restore()


def apply_dash(ctx, ls: str):
    if ls == "-":
        ctx.set_dash([])
    elif ls == "--":
        ctx.set_dash([18.0, 9.0])
    elif ls == "-.":
        ctx.set_dash([20.0, 7.0, 4.5, 7.0])
    elif ls == ":":
        ctx.set_dash([4.0, 6.0])


def render_hardware_roofline(metrics: dict, out_path: Path):
    W, H = 2640, 1560
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surface)

    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()

    left, right, top, bottom = 275, W - 75, 130, H - 180
    pw, ph = right - left, bottom - top

    # Log10 bounds: X in [1.0, 5000.0], Y in [6.0, 3000.0]
    xmin, xmax = 1.0, 5000.0
    ymin, ymax = 6.0, 3000.0
    lxmin, lxmax = math.log10(xmin), math.log10(xmax)
    lymin, lymax = math.log10(ymin), math.log10(ymax)

    def to_px(x_val: float, y_val: float) -> tuple[float, float]:
        lx = math.log10(max(x_val, 1e-6))
        ly = math.log10(max(y_val, 1e-6))
        px = left + (lx - lxmin) / (lxmax - lxmin) * pw
        py = bottom - (ly - lymin) / (lymax - lymin) * ph
        return px, py

    x_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    y_ticks = [10, 20, 50, 100, 200, 500, 1000, 2000]

    ctx.save()
    ctx.set_line_width(1.6)
    ctx.set_dash([3.0, 5.0])
    ctx.set_source_rgba(0.68, 0.68, 0.68, 0.75)
    for xt in x_ticks:
        px, _ = to_px(xt, ymin)
        ctx.move_to(px, top)
        ctx.line_to(px, bottom)
        ctx.stroke()
    for yt in y_ticks:
        _, py = to_px(xmin, yt)
        ctx.move_to(left, py)
        ctx.line_to(right, py)
        ctx.stroke()
    ctx.restore()

    ctx.save()
    ctx.rectangle(left, top, pw, ph)
    ctx.clip()

    # 1. Theoretical Hardware Roofline (9.6 TB/s slope -> 1,978 TFLOPS ceiling)
    p1 = to_px(xmin, REPLICA_PEAK_BW_TBPS * xmin)
    p_ridge = to_px(RIDGE_AI, REPLICA_PEAK_TFLOPS)
    p2 = to_px(xmax, REPLICA_PEAK_TFLOPS)
    ctx.save()
    ctx.set_source_rgb(*hex_rgb("#202124"))
    ctx.set_line_width(5.2)
    ctx.move_to(*p1)
    ctx.line_to(*p_ridge)
    ctx.line_to(*p2)
    ctx.stroke()
    ctx.restore()

    ctx.save()
    ctx.set_source_rgba(0.12, 0.13, 0.14, 0.05)
    ctx.move_to(*p1)
    ctx.line_to(*p_ridge)
    ctx.line_to(*p2)
    ctx.line_to(right, top)
    ctx.line_to(left, top)
    ctx.close_path()
    ctx.fill()
    ctx.restore()

    # 2. Vertical Ridge Point line at AI = 206 FLOPs/Byte
    rx, ry = p_ridge
    ctx.save()
    ctx.set_source_rgba(0.35, 0.38, 0.41, 0.75)
    ctx.set_line_width(2.8)
    ctx.set_dash([12.0, 8.0])
    ctx.move_to(rx, ry)
    ctx.line_to(rx, bottom)
    ctx.stroke()
    ctx.restore()

    # 3. Sustained Physical GPU Compute Ceiling (~361-388 TFLOPS/replica across all 3 arms at c=4096..8192)
    _, py_phys_ceil = to_px(xmin, 375.0)
    ctx.save()
    ctx.set_source_rgba(0.45, 0.25, 0.65, 0.75)
    ctx.set_line_width(3.2)
    ctx.set_dash([8.0, 6.0])
    ctx.move_to(left, py_phys_ceil)
    ctx.line_to(right, py_phys_ceil)
    ctx.stroke()
    ctx.restore()

    color_llmd = hex_rgb("#1a73e8")
    color_k8s = hex_rgb("#d93025")
    color_dyn = hex_rgb("#1e8e3e")
    color_phys = hex_rgb("#6a1b9a")

    # Physical GPU Compute curve for Arm A (llm-d)
    phys_pts = [to_px(m["phys_ai"], m["phys_tflops"]) for m in metrics["llmd"]]
    ctx.save()
    ctx.set_source_rgba(*color_phys, 0.85)
    ctx.set_line_width(4.2)
    apply_dash(ctx, ":")
    for i, (px, py) in enumerate(phys_pts):
        if i == 0:
            ctx.move_to(px, py)
        else:
            ctx.line_to(px, py)
    ctx.stroke()
    ctx.restore()
    for px, py in phys_pts:
        draw_marker(ctx, "d", px, py, 8.5, color_phys)

    # Effective Logical Compute Trajectories for all 3 arms
    arm_specs = [
        ("k8s_rr", color_k8s, "s", "--"),
        ("dynamo", color_dyn, "^", "-."),
        ("llmd", color_llmd, "o", "-"),
    ]
    for arm, col, mk, ls in arm_specs:
        pts = [to_px(m["eff_ai"], m["eff_tflops"]) for m in metrics[arm]]
        ctx.save()
        ctx.set_source_rgb(*col)
        ctx.set_line_width(5.8)
        apply_dash(ctx, ls)
        for i, (px, py) in enumerate(pts):
            if i == 0:
                ctx.move_to(px, py)
            else:
                ctx.line_to(px, py)
        ctx.stroke()
        ctx.restore()
        for px, py in pts:
            draw_marker(ctx, mk, px, py, 10.5, col)

    ctx.restore()  # remove clip

    # Spines
    ctx.save()
    ctx.set_source_rgb(0.1, 0.1, 0.1)
    ctx.set_line_width(2.6)
    ctx.move_to(left, top)
    ctx.line_to(left, bottom)
    ctx.line_to(right, bottom)
    ctx.stroke()

    for xt in x_ticks:
        px, _ = to_px(xt, ymin)
        ctx.move_to(px, bottom)
        ctx.line_to(px, bottom + 12)
        ctx.stroke()
        draw_text(ctx, f"{xt:,}", px, bottom + 22, "DejaVu Sans 21", (0.1, 0.1, 0.1), ha="center", va="top")

    for yt in y_ticks:
        _, py = to_px(xmin, yt)
        ctx.move_to(left - 12, py)
        ctx.line_to(left, py)
        ctx.stroke()
        draw_text(ctx, f"{yt:,}", left - 20, py, "DejaVu Sans 21", (0.1, 0.1, 0.1), ha="right", va="center")
    ctx.restore()

    draw_text(ctx, "MMShopBench Roofline Analysis: Qwen3-VL-32B (TP=2 H200 Replica, c=8..8192)",
              left + pw / 2, 58, "DejaVu Sans Bold 26", (0.05, 0.05, 0.05), ha="center", va="center")
    draw_text(ctx, "Operational Intensity (FLOPs / HBM Byte Transferred) [Log Scale]",
              left + pw / 2, H - 55, "DejaVu Sans Bold 23", (0.05, 0.05, 0.05), ha="center", va="center")
    draw_text(ctx, "Compute Throughput per Replica (TFLOPS/sec, TP=2 H200) [Log Scale]",
              65, top + ph / 2, "DejaVu Sans Bold 23", (0.05, 0.05, 0.05), ha="center", va="center", angle_deg=-90)

    # Ridge point annotation
    draw_text(ctx, "Hardware Ridge Point\nAI = 206 FLOPs/Byte",
              rx + 18, bottom - 85, "DejaVu Sans Bold 17.5", (0.25, 0.27, 0.30), ha="left", va="center")

    # Theoretical ceiling label
    _, py_1978 = to_px(900, REPLICA_PEAK_TFLOPS)
    draw_text(ctx, "Peak BF16 Tensor Core Roof: 1,978 TFLOPS/replica (2 × 989 TFLOPS)",
              to_px(950, REPLICA_PEAK_TFLOPS)[0], py_1978 - 20,
              "DejaVu Sans Bold 17.5", hex_rgb("#202124"), ha="center", va="bottom")

    # Sustained physical compute ceiling label (placed cleanly on the left side above the purple dashed line)
    draw_text(ctx, "Sustained Physical GPU Ceiling (~361–388 Physical TFLOPS/replica across all 3 arms)",
              left + 28, py_phys_ceil - 18, "DejaVu Sans Bold 16.5", color_phys, ha="left", va="bottom")

    # Key concurrency annotations on Arm A (llm-d) Effective curve
    m_c8 = metrics["llmd"][0]
    px8, py8 = to_px(m_c8["eff_ai"], m_c8["eff_tflops"])
    draw_text(ctx, "c=8 (ITL=11.0 ms, B_dec=1.0)\nHBM3e Bandwidth Floor",
              px8 + 18, py8 + 10, "DejaVu Sans Bold 17", color_llmd, ha="left", va="top")

    for idx_c in [2, 4, 6, 7, 8]:  # c=32, 128, 512, 1024, 2048
        mc = metrics["llmd"][idx_c]
        pxc, pyc = to_px(mc["eff_ai"], mc["eff_tflops"])
        draw_text(ctx, f"c={mc['c']}", pxc - 12, pyc - 16, "DejaVu Sans Bold 16.5", color_llmd, ha="right", va="bottom")

    # Clean non-overlapping annotations on the right side (AI = 650..5000) with leader lines
    m_a4096 = metrics["llmd"][9]
    px_a4, py_a4 = to_px(m_a4096["eff_ai"], m_a4096["eff_tflops"])
    draw_leader(ctx, px_a4, py_a4, px_a4 + 35, py_a4 - 55, color_llmd)
    draw_text(ctx, "llm-d Peak (c=4096):\n1,030 Eff TFLOPS (2.85× Phys)",
              px_a4 + 42, py_a4 - 55, "DejaVu Sans Bold 16.5", color_llmd, ha="left", va="center")

    m_d4096 = metrics["dynamo"][9]
    px_d4, py_d4 = to_px(m_d4096["eff_ai"], m_d4096["eff_tflops"])
    draw_leader(ctx, px_d4, py_d4, px_d4 + 80, py_d4 - 22, color_dyn)
    draw_text(ctx, "Dynamo Peak (c=4096): 859 Eff TFLOPS",
              px_d4 + 88, py_d4 - 22, "DejaVu Sans Bold 16.5", color_dyn, ha="left", va="center")

    m_a8192 = metrics["llmd"][10]
    px_a8, py_a8 = to_px(m_a8192["eff_ai"], m_a8192["eff_tflops"])
    draw_leader(ctx, px_a8, py_a8, px_a8 + 40, py_a8 + 4, color_llmd)
    draw_text(ctx, "llm-d c=8192 KV Wall: 768 Eff TFLOPS\n(Cache Hit: 89.7% → 68.4%)",
              px_a8 + 48, py_a8 + 6, "DejaVu Sans Bold 16", color_llmd, ha="left", va="center")

    m_k4096 = metrics["k8s_rr"][9]
    px_k4, py_k4 = to_px(m_k4096["eff_ai"], m_k4096["eff_tflops"])
    draw_leader(ctx, px_k4, py_k4, px_k4 + 80, py_k4 + 36, color_k8s)
    draw_text(ctx, "K8s RR Peak (c=4096): 669 Eff TFLOPS",
              px_k4 + 88, py_k4 + 38, "DejaVu Sans Bold 16.5", color_k8s, ha="left", va="center")

    # Physical curve c=4096 & c=8192 labels below the purple curve
    px_p4, py_p4 = to_px(m_a4096["phys_ai"], m_a4096["phys_tflops"])
    draw_leader(ctx, px_p4, py_p4, px_p4 + 10, py_p4 + 45, color_phys)
    draw_text(ctx, "c=4096 Physical (362 TFLOPS)", px_p4 + 16, py_p4 + 48,
              "DejaVu Sans Bold 16", color_phys, ha="left", va="top")

    px_p8, py_p8 = to_px(m_a8192["phys_ai"], m_a8192["phys_tflops"])
    draw_leader(ctx, px_p8, py_p8, px_p8 + 30, py_p8 + 6, color_phys)
    draw_text(ctx, "c=8192 Physical (388 TFLOPS)", px_p8 + 36, py_p8 + 6,
              "DejaVu Sans Bold 16", color_phys, ha="left", va="center")

    # Legend box (top-left)
    leg_x, leg_y, leg_w, leg_h = left + 24, top + 22, 1180, 260
    ctx.save()
    ctx.set_source_rgba(1, 1, 1, 0.93)
    ctx.rectangle(leg_x, leg_y, leg_w, leg_h)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.75, 0.75, 0.75, 0.9)
    ctx.set_line_width(2.0)
    ctx.stroke()
    ctx.restore()

    legend_items = [
        ("#202124", None, "-", "Theoretical TP=2 H200 Roofline (9.6 TB/s HBM3e, 1,978 TFLOPS BF16)"),
        ("#1a73e8", "o", "-", "Arm A: llm-d Effective Logical Compute (89.7% Prefix Hit at Peak)"),
        ("#1e8e3e", "^", "-.", "Arm C: NVIDIA Dynamo Effective Logical Compute (78.3% Hit at Peak)"),
        ("#d93025", "s", "--", "Arm B: K8s Round-Robin Effective Logical Compute (63.8% Hit at Peak)"),
        ("#6a1b9a", "d", ":", "Actual Physical GPU Compute Executed (Saturates at ~361–388 TFLOPS)"),
    ]
    for i, (col_hex, mk, ls, lbl) in enumerate(legend_items):
        ly = leg_y + 30 + i * 48
        lx0, lx1 = leg_x + 24, leg_x + 104
        col = hex_rgb(col_hex)
        ctx.save()
        ctx.set_source_rgb(*col)
        ctx.set_line_width(5.2)
        apply_dash(ctx, ls)
        ctx.move_to(lx0, ly)
        ctx.line_to(lx1, ly)
        ctx.stroke()
        ctx.restore()
        if mk:
            draw_marker(ctx, mk, (lx0 + lx1) / 2.0, ly, 9.5, col)
        draw_text(ctx, lbl, lx1 + 22, ly, "DejaVu Sans 19.5", (0.1, 0.1, 0.1), ha="left", va="center")

    surface.write_to_png(str(out_path))
    print("Saved:", out_path)


def render_serving_pareto_roofline(metrics: dict, out_path: Path):
    W, H = 2640, 1560
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surface)

    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()

    left, right, top, bottom = 275, W - 75, 130, H - 180
    pw, ph = right - left, bottom - top

    xmin, xmax = 0.0, 100.0
    ymin, ymax = 0.0, 8800.0

    def to_px(x_val: float, y_val: float) -> tuple[float, float]:
        px = left + (x_val - xmin) / (xmax - xmin) * pw
        py = bottom - (y_val - ymin) / (ymax - ymin) * ph
        return px, py

    x_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    y_ticks = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

    ctx.save()
    ctx.set_line_width(1.6)
    ctx.set_dash([3.0, 5.0])
    ctx.set_source_rgba(0.68, 0.68, 0.68, 0.75)
    for xt in x_ticks:
        px, _ = to_px(xt, ymin)
        ctx.move_to(px, top)
        ctx.line_to(px, bottom)
        ctx.stroke()
    for yt in y_ticks:
        _, py = to_px(xmin, yt)
        ctx.move_to(left, py)
        ctx.line_to(right, py)
        ctx.stroke()
    ctx.restore()

    color_llmd = hex_rgb("#1a73e8")
    color_k8s = hex_rgb("#d93025")
    color_dyn = hex_rgb("#1e8e3e")

    ctx.save()
    ctx.rectangle(left, top, pw, ph)
    ctx.clip()

    # Vertical HBM3e Single-Stream Interactivity Roof (~91-95 tok/s/user = ~10.5-11.0 ms ITL)
    px_hbm, _ = to_px(92.25, 0)
    ctx.save()
    ctx.set_source_rgba(0.25, 0.27, 0.30, 0.7)
    ctx.set_line_width(3.0)
    ctx.set_dash([12.0, 8.0])
    ctx.move_to(px_hbm, top)
    ctx.line_to(px_hbm, bottom)
    ctx.stroke()
    ctx.restore()

    arm_specs = [
        ("k8s_rr", color_k8s, "s", "--"),
        ("dynamo", color_dyn, "^", "-."),
        ("llmd", color_llmd, "o", "-"),
    ]

    for arm, col, mk, ls in arm_specs:
        pts_hi = [to_px(m["interactivity"], m["tot_tps_per_gpu"] + m["tot_tps_per_gpu_std"]) for m in metrics[arm]]
        pts_lo = [to_px(m["interactivity"], m["tot_tps_per_gpu"] - m["tot_tps_per_gpu_std"]) for m in reversed(metrics[arm])]
        ctx.save()
        ctx.set_source_rgba(*col, 0.15)
        for i, (px, py) in enumerate(pts_hi):
            if i == 0:
                ctx.move_to(px, py)
            else:
                ctx.line_to(px, py)
        for px, py in pts_lo:
            ctx.line_to(px, py)
        ctx.close_path()
        ctx.fill()
        ctx.restore()

        pts = [to_px(m["interactivity"], m["tot_tps_per_gpu"]) for m in metrics[arm]]
        ctx.save()
        ctx.set_source_rgb(*col)
        ctx.set_line_width(6.0)
        apply_dash(ctx, ls)
        for i, (px, py) in enumerate(pts):
            if i == 0:
                ctx.move_to(px, py)
            else:
                ctx.line_to(px, py)
        ctx.stroke()
        ctx.restore()
        for px, py in pts:
            draw_marker(ctx, mk, px, py, 10.5, col)

    ctx.restore()

    # Spines
    ctx.save()
    ctx.set_source_rgb(0.1, 0.1, 0.1)
    ctx.set_line_width(2.6)
    ctx.move_to(left, top)
    ctx.line_to(left, bottom)
    ctx.line_to(right, bottom)
    ctx.stroke()

    for xt in x_ticks:
        px, _ = to_px(xt, ymin)
        ctx.move_to(px, bottom)
        ctx.line_to(px, bottom + 12)
        ctx.stroke()
        draw_text(ctx, f"{xt}", px, bottom + 22, "DejaVu Sans 21", (0.1, 0.1, 0.1), ha="center", va="top")

    for yt in y_ticks:
        _, py = to_px(xmin, yt)
        ctx.move_to(left - 12, py)
        ctx.line_to(left, py)
        ctx.stroke()
        draw_text(ctx, f"{yt:,}", left - 20, py, "DejaVu Sans 21", (0.1, 0.1, 0.1), ha="right", va="center")
    ctx.restore()

    draw_text(ctx, "MMShopBench Serving Roofline: Throughput vs. Interactivity Pareto Frontier (16×H200)",
              left + pw / 2, 58, "DejaVu Sans Bold 25.5", (0.05, 0.05, 0.05), ha="center", va="center")
    draw_text(ctx, "Per-User Decode Interactivity (tokens/sec/user = 1000 / ITL p50) — Right is Faster Decode",
              left + pw / 2, H - 55, "DejaVu Sans Bold 22.5", (0.05, 0.05, 0.05), ha="center", va="center")
    draw_text(ctx, "Total Token Throughput per GPU (tokens/sec/H200) — Higher is Better",
              65, top + ph / 2, "DejaVu Sans Bold 22.5", (0.05, 0.05, 0.05), ha="center", va="center", angle_deg=-90)

    # HBM3e wall annotation
    draw_text(ctx, "HBM3e Single-Stream Bandwidth Wall\n(~91–95 tok/s/user = 10.5–11.0 ms ITL)",
              px_hbm - 16, top + 400, "DejaVu Sans Bold 17.5", (0.25, 0.27, 0.30), ha="right", va="center")

    # Concurrency annotations along Arm A (llm-d)
    for idx in [0, 3, 4, 5, 6, 7, 8]:  # c=8, 64, 128, 256, 512, 1024, 2048
        m = metrics["llmd"][idx]
        px, py = to_px(m["interactivity"], m["tot_tps_per_gpu"])
        if m["c"] == 8:
            draw_text(ctx, "c=8 (91.0 tok/s/user)", px - 14, py - 24,
                      "DejaVu Sans Bold 17", color_llmd, ha="right", va="bottom")
        else:
            draw_text(ctx, f"c={m['c']} ({m['tot_tps_per_gpu']:,.0f})", px + 14, py - 14,
                      "DejaVu Sans Bold 16.5", color_llmd, ha="left", va="bottom")

    # Clean non-overlapping Peak (c=4096) and Rollover (c=8192) annotations with leader lines
    m_a4 = metrics["llmd"][9]
    px_a4, py_a4 = to_px(m_a4["interactivity"], m_a4["tot_tps_per_gpu"])
    draw_leader(ctx, px_a4, py_a4, px_a4 + 125, py_a4 - 38, color_llmd)
    draw_text(ctx, "llm-d Peak (c=4096): 7,580 tok/s/GPU\n(+20.0% vs Dynamo, +54.0% vs K8s RR)",
              px_a4 + 132, py_a4 - 40, "DejaVu Sans Bold 17.5", color_llmd, ha="left", va="center")

    m_a8 = metrics["llmd"][10]
    px_a8, py_a8 = to_px(m_a8["interactivity"], m_a8["tot_tps_per_gpu"])
    draw_leader(ctx, px_a8, py_a8, px_a8 + 2, py_a8 - 255, color_llmd)
    draw_text(ctx, "c=8192 KV Rollover\n(5,649 tok/s/GPU)",
              px_a8 + 2, py_a8 - 262, "DejaVu Sans Bold 16.5", color_llmd, ha="left", va="bottom")

    m_d4 = metrics["dynamo"][9]
    px_d4, py_d4 = to_px(m_d4["interactivity"], m_d4["tot_tps_per_gpu"])
    draw_leader(ctx, px_d4, py_d4, px_d4 + 365, py_d4 + 18, color_dyn)
    draw_text(ctx, "Dynamo Peak (c=4096): 6,316 tok/s/GPU",
              px_d4 + 372, py_d4 + 18, "DejaVu Sans Bold 16.5", color_dyn, ha="left", va="center")

    m_k4 = metrics["k8s_rr"][9]
    px_k4, py_k4 = to_px(m_k4["interactivity"], m_k4["tot_tps_per_gpu"])
    draw_leader(ctx, px_k4, py_k4, px_k4 + 10, py_k4 + 215, color_k8s)
    draw_text(ctx, "K8s RR Peak (c=4096): 4,922 tok/s/GPU",
              px_k4 + 16, py_k4 + 218, "DejaVu Sans Bold 16.5", color_k8s, ha="left", va="center")

    # Legend (top-right)
    leg_w, leg_h = 900, 175
    leg_x, leg_y = right - leg_w - 28, top + 24
    ctx.save()
    ctx.set_source_rgba(1, 1, 1, 0.93)
    ctx.rectangle(leg_x, leg_y, leg_w, leg_h)
    ctx.fill_preserve()
    ctx.set_source_rgba(0.75, 0.75, 0.75, 0.9)
    ctx.set_line_width(2.0)
    ctx.stroke()
    ctx.restore()

    legend_items = [
        ("#1a73e8", "o", "-", "Arm A: llm-d Multimodal Router (EPP)"),
        ("#d93025", "s", "--", "Arm B: K8s Service (Round-Robin Baseline)"),
        ("#1e8e3e", "^", "-.", "Arm C: NVIDIA Dynamo KV Router"),
    ]
    for i, (col_hex, mk, ls, lbl) in enumerate(legend_items):
        ly = leg_y + 34 + i * 52
        lx0, lx1 = leg_x + 26, leg_x + 106
        col = hex_rgb(col_hex)
        ctx.save()
        ctx.set_source_rgb(*col)
        ctx.set_line_width(5.5)
        apply_dash(ctx, ls)
        ctx.move_to(lx0, ly)
        ctx.line_to(lx1, ly)
        ctx.stroke()
        ctx.restore()
        draw_marker(ctx, mk, (lx0 + lx1) / 2.0, ly, 10.0, col)
        draw_text(ctx, lbl, lx1 + 22, ly, "DejaVu Sans 20.5", (0.1, 0.1, 0.1), ha="left", va="center")

    surface.write_to_png(str(out_path))
    print("Saved:", out_path)


def main() -> None:
    data = json.loads(DATA_PATH.read_text())
    concs = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    metrics = {arm: [compute_roofline_metrics(data[arm][str(c)]) for c in concs] for arm in ["llmd", "dynamo", "k8s_rr"]}

    render_hardware_roofline(metrics, OUT_DIR / "mmshopbench_hardware_roofline.png")
    render_serving_pareto_roofline(metrics, OUT_DIR / "mmshopbench_serving_roofline_pareto.png")


if __name__ == "__main__":
    main()
