from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


class MediaCommandError(RuntimeError):
    """Raised when ffmpeg or ffprobe fails."""


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as error:
        raise MediaCommandError(f"Required executable not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip().splitlines()[-1:] or ["unknown media error"]
        raise MediaCommandError(f"Command failed for {command[-1]}: {detail[0]}") from error
    return result.stdout.strip()


def video_duration(video: Path) -> float:
    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ]
    )
    try:
        return max(float(output), 0.1)
    except ValueError as error:
        raise MediaCommandError(f"Invalid duration for {video}: {output!r}") from error


def extract_frame(video: Path, timestamp: float, output: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output),
        ]
    )


def make_temporal_grid(
    video: Path,
    output: Path,
    frame_ratios: list[float],
    cell_size: int = 448,
    jpeg_quality: int = 92,
) -> None:
    if len(frame_ratios) != 4:
        raise ValueError("A 2x2 grid requires exactly four frame ratios")
    if any(not 0.0 < ratio < 1.0 for ratio in frame_ratios):
        raise ValueError("Frame ratios must be between 0 and 1")

    duration = video_duration(video)
    cells = []
    with tempfile.TemporaryDirectory(prefix="home-llm-frames-") as temp_dir:
        for index, ratio in enumerate(frame_ratios, start=1):
            frame_path = Path(temp_dir) / f"frame_{index}.jpg"
            extract_frame(video, duration * ratio, frame_path)
            with Image.open(frame_path) as source:
                image = source.convert("RGB")
                image.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (cell_size, cell_size), "black")
            canvas.paste(image, ((cell_size - image.width) // 2, (cell_size - image.height) // 2))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((8, 8, 62, 38), fill="black")
            draw.text((17, 14), f"T{index}", fill="white", font=ImageFont.load_default())
            cells.append(canvas)

    grid = Image.new("RGB", (cell_size * 2, cell_size * 2), "black")
    for cell, position in zip(cells, ((0, 0), (cell_size, 0), (0, cell_size), (cell_size, cell_size))):
        grid.paste(cell, position)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output, format="JPEG", quality=jpeg_quality)

