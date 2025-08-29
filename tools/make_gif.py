import argparse
from pathlib import Path

import cv2
import imageio


def make_gif(input_path: Path, output_path: Path, stride: int = 5, max_frames: int = 180, width: int = 480, fps: int = 10):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    frames = []
    frame_idx = 0
    grabbed = True
    while grabbed and len(frames) < max_frames:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        if frame_idx % stride == 0:
            h, w = frame.shape[:2]
            if width and w != width:
                scale = width / float(w)
                frame = cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError("No frames collected to write GIF.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # duration per frame in seconds
    duration = 1.0 / max(1, fps)
    imageio.mimsave(output_path, frames, duration=duration)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create a GIF preview from a video.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video (mp4)")
    parser.add_argument("--output", type=str, required=True, help="Path to output gif")
    parser.add_argument("--stride", type=int, default=5, help="Sample every Nth frame")
    parser.add_argument("--max-frames", type=int, default=180, help="Maximum number of frames to include")
    parser.add_argument("--width", type=int, default=480, help="Resize width for GIF")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    make_gif(input_path, output_path, stride=args.stride, max_frames=args.max_frames, width=args.width, fps=args.fps)
    print(f"GIF saved to: {output_path}")


if __name__ == "__main__":
    main()
