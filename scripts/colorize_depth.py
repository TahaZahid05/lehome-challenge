"""
Colorize top depth maps and encode them as video files to create a 4th camera view.

Creates a NEW dataset by symlinking the heavy files (parquets, existing videos)
from the source dataset and adding colorized depth videos + updated metadata.

Reads observation.top_depth (uint16, 480x640) from parquet data files,
applies the turbo colormap to create 3-channel RGB images, and encodes them
as mp4 video files matching the existing RGB camera structure.

Usage:
    # Dry run to preview what will happen:
    python -m scripts.colorize_depth --dry-run

    # Full run (creates new dataset with symlinks + depth videos):
    python -m scripts.colorize_depth

    # Use h264 if av1 encoding is too slow:
    python -m scripts.colorize_depth --codec libx264
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ── Configuration ────────────────────────────────────────────────────────────

NEW_KEY = "observation.images.top_depth_colorized"
REF_KEY = "observation.images.top_rgb"  # Reference camera for video file grouping
DEPTH_COL = "observation.top_depth"
HEIGHT, WIDTH = 480, 640

DEFAULT_SRC = Path("Datasets/example/four_types_merged_with_depth")
DEFAULT_DST = Path("Datasets/example/four_types_merged_with_depth_colorized")


# ── Turbo Colormap ───────────────────────────────────────────────────────────

def _build_turbo_lut() -> np.ndarray:
    """Build a 256-entry RGB lookup table for the turbo colormap."""
    try:
        import matplotlib.cm as cm
        indices = np.linspace(0, 1, 256)
        rgba = cm.turbo(indices)  # (256, 4) float [0,1]
        return (rgba[:, :3] * 255).astype(np.uint8)
    except ImportError:
        # Fallback: simple blue → cyan → green → yellow → red gradient
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.25:
                s = t / 0.25
                lut[i] = [0, int(255 * s), 255]
            elif t < 0.5:
                s = (t - 0.25) / 0.25
                lut[i] = [0, 255, int(255 * (1 - s))]
            elif t < 0.75:
                s = (t - 0.5) / 0.25
                lut[i] = [int(255 * s), 255, 0]
            else:
                s = (t - 0.75) / 0.25
                lut[i] = [255, int(255 * (1 - s)), 0]
        return lut


TURBO_LUT = _build_turbo_lut()


def colorize_depth_frame(depth_uint16: np.ndarray) -> np.ndarray:
    """Convert a uint16 depth map to a turbo-colorized RGB image.

    Per-frame normalization ensures good contrast regardless of absolute depth range.

    Args:
        depth_uint16: (H, W) uint16 array, depth in millimeters.
    Returns:
        (H, W, 3) uint8 RGB image.
    """
    depth = depth_uint16.astype(np.float32)
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        normalized = (depth - d_min) / (d_max - d_min)
    else:
        normalized = np.zeros_like(depth)

    indices = (normalized * 255).astype(np.uint8)
    return TURBO_LUT[indices]  # (H, W) → (H, W, 3)


# ── Video Encoding ──────────────────────────────────────────────────────────

def encode_frames_to_video(frames: list[np.ndarray], output_path: Path,
                           fps: int, codec: str = "libsvtav1"):
    """Encode a list of RGB frames to an mp4 video by piping raw frames to ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(frames)
    h, w = frames[0].shape[:2]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

    # Write frames to ffmpeg stdin, handling broken pipe if ffmpeg dies early
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass  # ffmpeg crashed; we'll get the error from stderr below
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

    # Wait for ffmpeg to finish and capture stderr (don't use communicate()
    # which tries to flush already-closed stdin)
    proc.wait()
    stderr = proc.stderr.read() if proc.stderr else b""

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {output_path}:\n{stderr.decode('utf-8', errors='replace')}"
        )

    print(f"  ✓ {output_path.name} — {n_frames} frames")


# ── Dataset Setup (symlink + copy) ──────────────────────────────────────────

def setup_output_dataset(src_root: Path, dst_root: Path):
    """Create output dataset directory with symlinks to source heavy files
    and copies of metadata (which will be modified).

    Structure created:
        dst_root/
        ├── data/ → symlink to src_root/data/
        ├── videos/
        │   ├── observation.images.top_rgb/ → symlink to src
        │   ├── observation.images.left_rgb/ → symlink to src
        │   ├── observation.images.right_rgb/ → symlink to src
        │   └── observation.images.top_depth_colorized/  (new, created later)
        └── meta/  (copied from src, will be modified)
    """
    if dst_root.exists():
        print(f"  Output directory {dst_root} already exists, reusing it.")
    else:
        dst_root.mkdir(parents=True)

    # Symlink data/ directory (parquet files — read-only, unchanged)
    dst_data = dst_root / "data"
    if not dst_data.exists():
        src_data = (src_root / "data").resolve()
        os.symlink(src_data, dst_data)
        print(f"  ✓ Symlinked data/ → {src_data}")

    # Symlink existing video cameras
    dst_videos = dst_root / "videos"
    dst_videos.mkdir(exist_ok=True)
    src_videos = src_root / "videos"
    for cam_dir in sorted(src_videos.iterdir()):
        if cam_dir.is_dir():
            dst_cam = dst_videos / cam_dir.name
            if not dst_cam.exists():
                os.symlink(cam_dir.resolve(), dst_cam)
                print(f"  ✓ Symlinked videos/{cam_dir.name}/")

    # Copy meta/ directory (will be modified)
    dst_meta = dst_root / "meta"
    if not dst_meta.exists():
        shutil.copytree(src_root / "meta", dst_meta)
        print(f"  ✓ Copied meta/ directory")
    else:
        print(f"  meta/ already exists, reusing it.")


# ── Episode-to-File Mapping ─────────────────────────────────────────────────

def build_episode_file_mapping(data_dir: Path) -> dict[int, list[Path]]:
    """Scan all data parquet files to build a mapping of episode_index → data file paths.

    This is necessary because the episode metadata's data/file_index field does NOT
    reliably match the actual parquet file index (especially in merged datasets).

    Only reads the lightweight episode_index column, not the depth data.
    """
    all_data_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    ep_to_files: dict[int, list[Path]] = {}

    for df in all_data_files:
        table = pq.read_table(df, columns=["episode_index"])
        episodes_in_file = set(table["episode_index"].to_pylist())
        for ep in episodes_in_file:
            ep_to_files.setdefault(ep, []).append(df)
        del table

    return ep_to_files


# ── Depth Frame Reading ─────────────────────────────────────────────────────

def read_depth_frames_for_episode(
    data_files: list[Path],
    episode_idx: int,
) -> list[np.ndarray]:
    """Read and colorize depth frames for a single episode.

    Reads from the data files that contain this episode (determined by scanning).
    Sorts frames by frame_index to ensure correct ordering.
    """
    all_frame_data = []  # (frame_index, colorized_frame)

    for data_path in data_files:
        table = pq.read_table(data_path, columns=[DEPTH_COL, "episode_index", "frame_index"])
        ep_index_arr = table["episode_index"].to_pylist()
        frame_index_arr = table["frame_index"].to_pylist()
        depth_col = table[DEPTH_COL]

        for i, ep in enumerate(ep_index_arr):
            if ep == episode_idx:
                raw = np.array(depth_col[i].as_py(), dtype=np.uint16).reshape(HEIGHT, WIDTH)
                all_frame_data.append((frame_index_arr[i], colorize_depth_frame(raw)))

        del table

    # Sort by frame_index to ensure correct ordering
    all_frame_data.sort(key=lambda x: x[0])
    return [frame for _, frame in all_frame_data]


# ── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Colorize depth maps and create a new dataset with a 4th camera view"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SRC,
        help=f"Source dataset root (default: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DST,
        help=f"Output dataset root (default: {DEFAULT_DST})",
    )
    parser.add_argument(
        "--codec",
        default="libsvtav1",
        choices=["libsvtav1", "libx264"],
        help="Video codec (default: libsvtav1 for av1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without actually processing",
    )
    args = parser.parse_args()

    src_root = args.source.resolve()
    dst_root = args.output.resolve()
    print(f"Source:  {src_root}")
    print(f"Output:  {dst_root}")

    # ── 1. Read source info.json ─────────────────────────────────────────
    with open(src_root / "meta" / "info.json") as f:
        info = json.load(f)

    fps = info["fps"]
    video_path_template = info["video_path"]
    print(f"FPS: {fps}, Total episodes: {info['total_episodes']}, "
          f"Total frames: {info['total_frames']}")

    # ── 2. Read episode metadata ─────────────────────────────────────────
    print("Reading episode metadata...")
    ep_dir = src_root / "meta" / "episodes"
    ep_files = sorted(ep_dir.glob("chunk-*/file-*.parquet"))
    episodes_df = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    print(f"  Loaded {len(episodes_df)} episodes")

    # Reference camera columns
    ref_chunk_col = f"videos/{REF_KEY}/chunk_index"
    ref_file_col = f"videos/{REF_KEY}/file_index"
    ref_from_col = f"videos/{REF_KEY}/from_timestamp"
    ref_to_col = f"videos/{REF_KEY}/to_timestamp"

    # ── 3. Group episodes by video file ──────────────────────────────────
    grouped = episodes_df.groupby([ref_chunk_col, ref_file_col])
    n_video_files = len(grouped)
    print(f"  {n_video_files} video files to create")

    if args.dry_run:
        print(f"\n[DRY RUN] Would create new dataset at: {dst_root}")
        print(f"  - Symlink data/ and existing videos/")
        print(f"  - Copy and update meta/")
        print(f"  - Create {n_video_files} depth video files:\n")
        for (chunk_idx, file_idx), group in grouped:
            out_path = dst_root / video_path_template.format(
                video_key=NEW_KEY, chunk_index=int(chunk_idx), file_index=int(file_idx)
            )
            n_eps = len(group)
            n_frames = sum(
                int(row["dataset_to_index"]) - int(row["dataset_from_index"])
                for _, row in group.iterrows()
            )
            print(f"    {out_path.name}: {n_eps} episodes, ~{n_frames} frames")
        print("\nDry run complete.")
        return

    # ── 4. Setup output dataset ──────────────────────────────────────────
    print("\nSetting up output dataset...")
    setup_output_dataset(src_root, dst_root)

    # ── 5. Build episode → data file mapping ─────────────────────────────
    # The episode metadata's data/file_index is unreliable in merged datasets,
    # so we scan the actual data parquet files to find which episodes they contain.
    print("\nScanning data files for episode mapping...")
    data_dir = dst_root / "data"  # This is symlinked to source
    ep_to_files = build_episode_file_mapping(data_dir)
    print(f"  Mapped {len(ep_to_files)} episodes across data files")

    # ── 6. Process each video file ───────────────────────────────────────
    print(f"\nEncoding {n_video_files} colorized depth videos...")
    for vid_num, ((chunk_idx, file_idx), group) in enumerate(grouped):
        chunk_idx = int(chunk_idx)
        file_idx = int(file_idx)

        out_path = dst_root / video_path_template.format(
            video_key=NEW_KEY, chunk_index=chunk_idx, file_index=file_idx
        )

        if out_path.exists():
            print(f"[{vid_num + 1}/{n_video_files}] {out_path.name} already exists, skipping")
            continue

        print(f"[{vid_num + 1}/{n_video_files}] Building {out_path.name} "
              f"({len(group)} episodes)...")

        all_frames = []
        for _, ep in group.iterrows():
            ep_idx = int(ep["episode_index"])
            data_files = ep_to_files.get(ep_idx, [])
            if not data_files:
                print(f"  ⚠ Episode {ep_idx} not found in any data file — skipping")
                continue

            ep_frames = read_depth_frames_for_episode(data_files, ep_idx)
            if not ep_frames:
                print(f"  ⚠ Episode {ep_idx} returned 0 frames — skipping")
                continue
            all_frames.extend(ep_frames)

        if not all_frames:
            print(f"  ⚠ No frames for this video file — skipping")
            continue

        encode_frames_to_video(all_frames, out_path, fps, args.codec)
        del all_frames  # Free memory

    # ── 7. Update episode metadata in the OUTPUT dataset ─────────────────
    print("\nUpdating episode metadata...")
    new_chunk_col = f"videos/{NEW_KEY}/chunk_index"
    new_file_col = f"videos/{NEW_KEY}/file_index"
    new_from_col = f"videos/{NEW_KEY}/from_timestamp"
    new_to_col = f"videos/{NEW_KEY}/to_timestamp"

    # Copy the exact same video assignments from the reference camera
    episodes_df[new_chunk_col] = episodes_df[ref_chunk_col]
    episodes_df[new_file_col] = episodes_df[ref_file_col]
    episodes_df[new_from_col] = episodes_df[ref_from_col]
    episodes_df[new_to_col] = episodes_df[ref_to_col]

    # Write to the OUTPUT dataset's episodes file
    dst_ep_dir = dst_root / "meta" / "episodes"
    dst_ep_files = sorted(dst_ep_dir.glob("chunk-*/file-*.parquet"))
    for ep_file in dst_ep_files:
        episodes_df.to_parquet(ep_file)
        break  # Only one episodes file
    print("  ✓ Episode metadata updated")

    # ── 8. Update info.json in the OUTPUT dataset ────────────────────────
    print("Updating info.json...")
    info["features"][NEW_KEY] = {
        "dtype": "video",
        "shape": [HEIGHT, WIDTH, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": HEIGHT,
            "video.width": WIDTH,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }

    dst_info_path = dst_root / "meta" / "info.json"
    with open(dst_info_path, "w") as f:
        json.dump(info, f, indent=4)
    print("  ✓ info.json updated")

    print(f"\n✅ Done! New dataset created at: {dst_root}")
    print(f"   New feature: {NEW_KEY}")
    print(f"\n   Training config usage:")
    print(f"   dataset:")
    print(f"     repo_id: four_types_merged_with_depth_colorized")
    print(f"     root: {args.output}")
    print(f"   ...")
    print(f"     {NEW_KEY}:")
    print(f"       type: VISUAL")
    print(f"       shape: [3, {HEIGHT}, {WIDTH}]")


if __name__ == "__main__":
    main()
