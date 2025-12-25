"""Create a simple animated GIF from key plot images.

This is a lightweight way to make the GitHub README feel more alive even
without recording the Streamlit UI.

Usage:
    python scripts/make_demo_gif.py --input-dir assets/plots --output assets/streamlit_demo.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default=str(Path("assets") / "plots"))
    p.add_argument("--output", type=str, default=str(Path("assets") / "streamlit_demo.gif"))
    p.add_argument("--duration", type=int, default=750, help="Frame duration in ms")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Pillow is required to build the GIF. Install with: pip install -r requirements.txt"
        ) from e

    # A small, recruiter-friendly story: performance → segmentation → calibration
    frames = []
    for name in [
        "01_model_performance_comparison.png",
        "05_portfolio_analysis.png",
        "07_calibration_curves.png",
        "08_shap_summary.png",
    ]:
        p = in_dir / name
        if p.exists():
            frames.append(Image.open(p).convert("P"))

    if not frames:
        raise SystemExit(f"No input PNGs found in {in_dir}")

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(args.duration),
        loop=0,
        optimize=True,
    )

    print(f"✅ Wrote GIF: {out_path}")


if __name__ == "__main__":
    main()
