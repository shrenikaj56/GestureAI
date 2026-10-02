from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 24
DURATION = 9.2
FRAME_COUNT = round(FPS * DURATION)

CYAN = (235, 210, 90)
BLUE = (220, 140, 55)
VIOLET = (215, 125, 190)
WHITE = (245, 250, 255)


def smoothstep(start: float, end: float, value: float) -> float:
    ratio = np.clip((value - start) / (end - start), 0.0, 1.0)
    return float(ratio * ratio * (3.0 - 2.0 * ratio))


def make_background() -> np.ndarray:
    x = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)[None, :, None]
    y = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32)[:, None, None]
    base = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)
    base[:] = (13.0, 8.0, 5.0)
    vignette = np.clip(1.0 - 0.38 * (x * x + y * y), 0.42, 1.0)
    blue_haze = np.exp(-((x + 0.55) ** 2 / 0.52 + (y + 0.15) ** 2 / 1.6))
    cyan_haze = np.exp(-((x - 0.62) ** 2 / 0.45 + (y - 0.16) ** 2 / 1.4))
    base *= vignette
    base[:, :, 0] += (12.0 * blue_haze[:, :, 0] + 5.0 * cyan_haze[:, :, 0])
    base[:, :, 1] += (9.0 * blue_haze[:, :, 0] + 8.0 * cyan_haze[:, :, 0])
    base[:, :, 2] += (4.0 * blue_haze[:, :, 0] + 12.0 * cyan_haze[:, :, 0])
    return np.clip(base, 0, 255).astype(np.uint8)


def add_glow(frame: np.ndarray, layer: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    bloom = cv2.GaussianBlur(layer, (0, 0), sigmaX=sigma, sigmaY=sigma)
    frame = cv2.addWeighted(frame, 1.0, bloom, strength, 0)
    return cv2.addWeighted(frame, 1.0, layer, 0.62, 0)


def draw_grid(layer: np.ndarray, time_seconds: float) -> None:
    horizon = int(HEIGHT * 0.56)
    center_x = WIDTH // 2
    for offset in range(-12, 13):
        end_x = int(center_x + offset * WIDTH * 0.105)
        cv2.line(layer, (center_x, horizon), (end_x, HEIGHT), (30, 56, 78), 1, cv2.LINE_AA)
    for index in range(1, 14):
        distance = (index / 13.0) ** 1.9
        y = int(horizon + distance * (HEIGHT - horizon))
        shift = int(math.sin(time_seconds * 0.25) * 5)
        cv2.line(layer, (0, y + shift), (WIDTH, y + shift), (23, 44, 65), 1, cv2.LINE_AA)
    for x in (120, WIDTH - 120):
        cv2.line(layer, (x, 0), (x, HEIGHT), (22, 39, 60), 1, cv2.LINE_AA)


def draw_trails(layer: np.ndarray, time_seconds: float, opacity: float) -> None:
    colors = (CYAN, BLUE, VIOLET)
    phases = (0.2, 2.3, 4.4)
    vertical_centers = (0.20, 0.76, 0.37)
    for index, (color, phase, center) in enumerate(zip(colors, phases, vertical_centers)):
        u = np.linspace(0.0, 1.0, 180, dtype=np.float32)
        shift = (time_seconds * (0.075 + index * 0.012) + index * 0.23) % 1.4 - 0.2
        x = (u + shift) * WIDTH
        y = HEIGHT * center + 56.0 * np.sin(u * math.tau + time_seconds * 0.62 + phase)
        y += 15.0 * np.sin(u * math.tau * 2.0 - time_seconds * 0.36 + phase)
        points = np.column_stack((x, y)).astype(np.int32)
        cv2.polylines(layer, [points], False, color, 2, cv2.LINE_AA)
    if opacity < 0.999:
        layer[:] = (layer.astype(np.float32) * opacity).astype(np.uint8)


HAND_POINTS = {
    0: (0, 220),
    1: (-72, 128),
    2: (-122, 87),
    3: (-165, 46),
    4: (-196, 8),
    5: (-91, 73),
    6: (-96, -7),
    7: (-101, -87),
    8: (-105, -158),
    9: (-35, 54),
    10: (-37, -43),
    11: (-38, -137),
    12: (-39, -211),
    13: (22, 60),
    14: (29, -36),
    15: (37, -124),
    16: (45, -188),
    17: (76, 84),
    18: (98, 10),
    19: (113, -61),
    20: (124, -123),
}

HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (5, 9), (9, 13), (13, 17),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

HAND_OUTLINE = (
    (0, 220), (-34, 185), (-72, 140), (-119, 99), (-160, 58), (-191, 26),
    (-205, 4), (-199, -12), (-184, -20), (-168, -14), (-148, 5), (-117, 39),
    (-105, 49), (-108, -129), (-115, -157), (-108, -177), (-90, -185),
    (-73, -179), (-65, -161), (-61, 44), (-53, 58), (-50, -176),
    (-50, -211), (-37, -232), (-17, -239), (2, -231), (9, -211), (7, 53),
    (17, 55), (20, -159), (20, -190), (34, -211), (53, -218), (69, -205),
    (72, -184), (64, 62), (77, 76), (92, -99), (93, -124), (107, -143),
    (126, -148), (143, -133), (146, -112), (132, -33), (115, 43),
    (99, 111), (69, 155), (30, 188), (24, 220),
)


def draw_hand(layer: np.ndarray, time_seconds: float, center: tuple[int, int], scale: float) -> None:
    points = {
        index: (int(center[0] + x * scale), int(center[1] + y * scale))
        for index, (x, y) in HAND_POINTS.items()
    }
    outline = np.array(
        [[int(center[0] + x * scale), int(center[1] + y * scale)] for x, y in HAND_OUTLINE],
        dtype=np.int32,
    )
    cv2.fillPoly(layer, [outline], (70, 30, 12), lineType=cv2.LINE_AA)
    cv2.polylines(layer, [outline], True, CYAN, 2, cv2.LINE_AA)
    for start, end in HAND_BONES:
        cv2.line(layer, points[start], points[end], CYAN, 3, cv2.LINE_AA)
    for index, point in points.items():
        radius = 7 if index in {0, 4, 8, 12, 16, 20} else 5
        pulse = 0.7 + 0.3 * math.sin(time_seconds * 2.0 + index * 0.55)
        color = tuple(int(channel * pulse) for channel in WHITE)
        cv2.circle(layer, point, radius + 5, BLUE, 1, cv2.LINE_AA)
        cv2.circle(layer, point, radius, color, -1, cv2.LINE_AA)


def draw_interface(layer: np.ndarray, center: tuple[int, int], radius: int, opacity: float) -> None:
    color = tuple(int(channel * opacity) for channel in BLUE)
    for ring_radius in (radius, int(radius * 1.12)):
        cv2.circle(layer, center, ring_radius, color, 1, cv2.LINE_AA)
    left, right = center[0] - radius, center[0] + radius
    top, bottom = center[1] - radius, center[1] + radius
    corner = 42
    for x, y, dx, dy in (
        (left, top, 1, 1), (right, top, -1, 1),
        (left, bottom, 1, -1), (right, bottom, -1, -1),
    ):
        cv2.line(layer, (x, y), (x + dx * corner, y), color, 2, cv2.LINE_AA)
        cv2.line(layer, (x, y), (x, y + dy * corner), color, 2, cv2.LINE_AA)
    for angle in range(0, 360, 30):
        radians = math.radians(angle)
        inner = radius * 1.05
        outer = radius * 1.09
        p1 = (int(center[0] + math.cos(radians) * inner), int(center[1] + math.sin(radians) * inner))
        p2 = (int(center[0] + math.cos(radians) * outer), int(center[1] + math.sin(radians) * outer))
        cv2.line(layer, p1, p2, color, 2, cv2.LINE_AA)


def draw_control_icons(layer: np.ndarray, opacity: float) -> None:
    color = tuple(int(channel * opacity) for channel in CYAN)
    icon_centers = ((365, 466), (1537, 406), (1515, 700))
    for center in icon_centers:
        cv2.circle(layer, center, 65, color, 1, cv2.LINE_AA)

    # Presentation display.
    cx, cy = icon_centers[0]
    cv2.rectangle(layer, (cx - 34, cy - 24), (cx + 34, cy + 20), color, 2, cv2.LINE_AA)
    cv2.line(layer, (cx, cy + 20), (cx, cy + 37), color, 2, cv2.LINE_AA)
    cv2.line(layer, (cx - 18, cy + 37), (cx + 18, cy + 37), color, 2, cv2.LINE_AA)
    cv2.line(layer, (cx - 24, cy + 9), (cx - 4, cy - 8), color, 2, cv2.LINE_AA)
    cv2.line(layer, (cx - 4, cy - 8), (cx + 10, cy + 2), color, 2, cv2.LINE_AA)
    cv2.line(layer, (cx + 10, cy + 2), (cx + 26, cy - 14), color, 2, cv2.LINE_AA)

    # Browser globe.
    cx, cy = icon_centers[1]
    cv2.circle(layer, (cx, cy), 34, color, 2, cv2.LINE_AA)
    cv2.ellipse(layer, (cx, cy), (15, 34), 0, 0, 360, color, 1, cv2.LINE_AA)
    cv2.line(layer, (cx - 31, cy), (cx + 31, cy), color, 1, cv2.LINE_AA)
    cv2.ellipse(layer, (cx, cy), (33, 14), 0, 0, 360, color, 1, cv2.LINE_AA)

    # Document.
    cx, cy = icon_centers[2]
    cv2.rectangle(layer, (cx - 27, cy - 34), (cx + 27, cy + 34), color, 2, cv2.LINE_AA)
    cv2.line(layer, (cx - 15, cy - 11), (cx + 15, cy - 11), color, 2, cv2.LINE_AA)
    cv2.line(layer, (cx - 15, cy + 1), (cx + 15, cy + 1), color, 2, cv2.LINE_AA)
    cv2.line(layer, (cx - 15, cy + 13), (cx + 8, cy + 13), color, 2, cv2.LINE_AA)

    for start, end in ((icon_centers[0], (690, 520)), ((1230, 500), icon_centers[1]), ((1200, 620), icon_centers[2])):
        cv2.line(layer, start, end, tuple(int(channel * opacity * 0.38) for channel in BLUE), 1, cv2.LINE_AA)


def draw_particles(layer: np.ndarray, time_seconds: float, particles: np.ndarray) -> None:
    for x, y, speed, radius, phase in particles:
        px = int((x + time_seconds * speed * 16.0) % WIDTH)
        py = int(y + 4.0 * math.sin(time_seconds * 0.55 + phase))
        cv2.circle(layer, (px, py), int(radius), (90, 151, 187), -1, cv2.LINE_AA)


def blend_text_gradient(
    frame: np.ndarray,
    mask: Image.Image,
    box: tuple[int, int, int, int],
    opacity: float,
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
) -> np.ndarray:
    left, top, right, bottom = box
    cropped_mask = np.asarray(mask.crop(box), dtype=np.float32) / 255.0
    x = np.linspace(0.0, 1.0, max(1, right - left), dtype=np.float32)[None, :, None]
    start = np.array(start_rgb, dtype=np.float32)
    end = np.array(end_rgb, dtype=np.float32)
    rgb = start * (1.0 - x) + end * x
    patch = frame[top:bottom, left:right, ::-1].astype(np.float32)
    alpha = (cropped_mask * opacity)[:, :, None]
    frame[top:bottom, left:right, :] = np.clip(
        (patch * (1.0 - alpha) + rgb * alpha)[:, :, ::-1], 0, 255
    ).astype(np.uint8)
    return frame


def draw_brand(frame: np.ndarray, time_seconds: float) -> np.ndarray:
    opacity = smoothstep(5.15, 6.45, time_seconds)
    opacity *= 1.0 - smoothstep(8.96, DURATION, time_seconds) * 0.55
    if opacity <= 0:
        return frame

    font_path = "C:/Windows/Fonts/segoeuil.ttf"
    title_font = ImageFont.truetype(font_path, 112)
    tagline_font = ImageFont.truetype(font_path, 42)
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    title = "Gesture"
    ai = "AI"
    title_width = int(draw.textlength(title + ai, font=title_font))
    title_left = 1210 - title_width // 2
    title_top = 432
    draw.text((title_left, title_top), title, font=title_font, fill=255)
    ai_left = title_left + int(draw.textlength(title, font=title_font))
    ai_mask = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(ai_mask).text((ai_left, title_top), ai, font=title_font, fill=255)
    title_box = (title_left, title_top, ai_left + int(draw.textlength(ai, font=title_font)) + 4, title_top + 135)
    draw.bitmap((0, 0), ai_mask, fill=255)

    title_glow = cv2.GaussianBlur(np.asarray(mask), (0, 0), 17)
    glow_layer = np.zeros_like(frame)
    glow_layer[:, :, 0] = title_glow * 0.14
    glow_layer[:, :, 1] = title_glow * 0.10
    glow_layer[:, :, 2] = title_glow * 0.04
    frame = cv2.addWeighted(frame, 1.0, glow_layer, opacity, 0)
    frame = blend_text_gradient(frame, mask, title_box, opacity, (248, 252, 255), (195, 232, 251))
    frame = blend_text_gradient(frame, ai_mask, title_box, opacity, (95, 225, 255), (192, 151, 255))

    tagline = "Move naturally, control digitally."
    tagline_box = draw.textbbox((0, 0), tagline, font=tagline_font)
    tagline_width = tagline_box[2] - tagline_box[0]
    tagline_x = 1210 - tagline_width // 2
    draw.text((tagline_x, 582), tagline, font=tagline_font, fill=235)
    tagline_crop = (tagline_x, 580, tagline_x + tagline_width + 2, 640)
    frame = blend_text_gradient(frame, mask, tagline_crop, opacity * 0.96, (223, 239, 250), (185, 218, 241))
    return frame


def render_frame(time_seconds: float, background: np.ndarray, particles: np.ndarray) -> np.ndarray:
    zoom = 1.0 + 0.012 * smoothstep(0.0, 5.2, time_seconds)
    transform = np.array(
        [[zoom, 0.0, WIDTH * (1.0 - zoom) * 0.5], [0.0, zoom, HEIGHT * (1.0 - zoom) * 0.5]],
        dtype=np.float32,
    )
    frame = cv2.warpAffine(background, transform, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)

    grid = np.zeros_like(frame)
    draw_grid(grid, time_seconds)
    frame = cv2.addWeighted(frame, 1.0, grid, 0.22, 0)

    particles_layer = np.zeros_like(frame)
    draw_particles(particles_layer, time_seconds, particles)
    particle_opacity = 1.0 - smoothstep(7.4, 8.8, time_seconds) * 0.58
    frame = cv2.addWeighted(frame, 1.0, particles_layer, 0.46 * particle_opacity, 0)

    trail_alpha = smoothstep(0.55, 1.6, time_seconds) * (1.0 - smoothstep(7.2, 8.4, time_seconds))
    trails = np.zeros_like(frame)
    draw_trails(trails, time_seconds, trail_alpha)
    frame = add_glow(frame, trails, sigma=16, strength=0.40)

    interface_alpha = smoothstep(2.0, 3.1, time_seconds) * (1.0 - smoothstep(7.1, 8.2, time_seconds))
    if interface_alpha > 0:
        interface = np.zeros_like(frame)
        hand_center_x = int(960 - 490 * smoothstep(4.75, 6.35, time_seconds))
        hand_center_y = 550
        radius = int(286 - 58 * smoothstep(4.75, 6.35, time_seconds))
        draw_interface(interface, (hand_center_x, hand_center_y), radius, interface_alpha)
        frame = add_glow(frame, interface, sigma=11, strength=0.32)

    hand_alpha = smoothstep(2.0, 3.15, time_seconds) * (1.0 - smoothstep(7.0, 8.15, time_seconds) * 0.72)
    if hand_alpha > 0:
        hand_layer = np.zeros_like(frame)
        hand_center_x = int(960 - 490 * smoothstep(4.75, 6.35, time_seconds))
        hand_center_y = 550
        hand_scale = 1.28 - 0.16 * smoothstep(4.75, 6.35, time_seconds)
        draw_hand(hand_layer, time_seconds, (hand_center_x, hand_center_y), hand_scale)
        bloom = cv2.GaussianBlur(hand_layer, (0, 0), sigmaX=12, sigmaY=12)
        frame = cv2.addWeighted(frame, 1.0, bloom, 0.48 * hand_alpha, 0)
        frame = cv2.addWeighted(frame, 1.0, hand_layer, 0.72 * hand_alpha, 0)

    icon_alpha = smoothstep(4.0, 5.2, time_seconds) * (1.0 - smoothstep(7.2, 8.0, time_seconds))
    if icon_alpha > 0:
        icons = np.zeros_like(frame)
        draw_control_icons(icons, icon_alpha)
        frame = add_glow(frame, icons, sigma=9, strength=0.26)

    frame = draw_brand(frame, time_seconds)
    final_fade = smoothstep(8.94, DURATION, time_seconds)
    if final_fade > 0:
        frame = (frame.astype(np.float32) * (1.0 - final_fade)).astype(np.uint8)
    return frame


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate the local GestureAI startup intro video.")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "assets" / "gestureai_intro.mp4",
    )
    parser.add_argument("--preview", type=Path, help="Optional path for a 4-frame visual contact sheet.")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    background = make_background()
    rng = np.random.default_rng(23)
    particles = np.column_stack(
        (
            rng.uniform(0, WIDTH, 115),
            rng.uniform(0, HEIGHT, 115),
            rng.uniform(0.15, 1.0, 115),
            rng.integers(1, 3, 115),
            rng.uniform(0, math.tau, 115),
        )
    )

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.CAP_MSMF,
        cv2.VideoWriter_fourcc(*"H264"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 video writer.")

    preview_times = (1.1, 3.45, 5.8, 8.45)
    previews: list[np.ndarray] = []
    try:
        for index in range(FRAME_COUNT):
            time_seconds = index / FPS
            frame = render_frame(time_seconds, background, particles)
            writer.write(frame)
            if args.preview and any(abs(time_seconds - sample) < 0.5 / FPS for sample in preview_times):
                previews.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(args.output))
    if not capture.isOpened():
        raise RuntimeError("Generated MP4 could not be reopened for validation.")
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if (actual_width, actual_height) != (WIDTH, HEIGHT) or actual_frames < FRAME_COUNT - 1:
        raise RuntimeError("Generated MP4 did not pass its dimensions/frame-count check.")

    if args.preview and previews:
        sheet = np.vstack((np.hstack(previews[:2]), np.hstack(previews[2:])))
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview), sheet):
            raise RuntimeError("Could not write the requested preview contact sheet.")

    print(
        f"Created {args.output} ({actual_width}x{actual_height}, "
        f"{actual_fps:.2f} fps, {actual_frames / actual_fps:.2f} seconds)."
    )


if __name__ == "__main__":
    main()