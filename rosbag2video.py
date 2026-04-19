#!/usr/bin/env python3
"""
Module to generate video output given ROS 2 bag folders via direct
sqlite3 extraction from .db3 files and metadata.yaml
e.g. uv run scripts/rosbag-to-lerobot/rosbag2video/rosbag2video.py -r 50 rosbag_dir/
"""
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Maximilian Laiacker.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np


try:
    import cv2
except Exception as exc:
    print("cv2 module not found or failed to import:", exc)
    cv2 = None

try:
    from rosbags.highlevel import AnyReader
    from rosbags.interfaces import Connection
    from rosbags.typesys import Stores, get_typestore
except ModuleNotFoundError:
    print("rosbags module not found")
    #try to run pip to install it
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rosbags"])
        from rosbags.highlevel import AnyReader
        from rosbags.interfaces import Connection
        from rosbags.typesys import Stores, get_typestore
    except Exception as e:
        print("Failed to install rosbags module:", e)
        sys.exit(1)


DEFAULT_TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

IS_VERBOSE = False


def get_ffmpeg_executable() -> Optional[str]:
    """Return an ffmpeg executable path if available.

    Preference order:
    1) System ffmpeg on PATH
    2) Bundled binary from imageio-ffmpeg (if installed)
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin

    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_bin and Path(ffmpeg_bin).exists():
            return ffmpeg_bin
    except Exception:
        pass

    return None


def is_rosbag_path(p: Path) -> bool:
    """Return True if Path *p* looks like a ROS2 bag (file or directory).

    Heuristics: contains metadata.yaml, contains any .db3/.db files or name startswith 'rosbag2'.
    """
    try:
        if not p.exists():
            return False
        if p.is_file():
            return p.suffix in (".db3", ".db", ".bag")
        # directory heuristics
        if (p / "metadata.yaml").exists():
            return True
        for f in p.iterdir():
            if f.is_file() and f.suffix in (".db3", ".db"):
                return True
        if p.name.startswith("rosbag2"):
            return True
    except Exception:
        return False
    return False


def find_bag_paths(path: Path) -> list[Path]:
    """Given a path, return a list of bag paths.

    - If *path* itself looks like a bag, return [path].
    - If *path* is a parent directory, return its child paths that look like bags.
    - If none found, fall back to returning [path] (with a warning).
    """
    path = path.expanduser().resolve()
    if is_rosbag_path(path):
        return [path]
    if path.is_dir():
        candidates = []
        for child in sorted(path.iterdir()):
            if is_rosbag_path(child):
                candidates.append(child)
        if candidates:
            return candidates
        print(f"[WARN] - No rosbag directories found inside {path}; treating as single bag.")
        return [path]
    return []


def get_topic_info(reader: AnyReader, topic_name: str) -> Tuple[int, str, Connection]:
    """Return (message_count, msg_type, connection) for *topic_name*."""
    conn = next((c for c in reader.connections if c.topic == topic_name), None)
    if conn is None:
        sys.exit(f"[ERROR] - Topic '{topic_name}' not found in bag.")
    return conn.msgcount, conn.msgtype, conn


def get_msg_format_from_rosbag(reader: AnyReader, connection: Connection) -> str:
    """Peek at first message to derive ``msg.format``/``msg.encoding``."""
    try:
        _, _, raw = next(reader.messages(connections=[connection]))
    except StopIteration:
        return "", None
    msg = reader.deserialize(raw, connection.msgtype)
    return getattr(msg, "format", getattr(msg, "encoding", "")), msg

def decode_ros_image_message(msg, input_msg_type: str):
    """Convert a ROS image message into an OpenCV image array."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required to decode ROS image messages.")

    if input_msg_type.endswith("CompressedImage"):
        buffer = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Failed to decode compressed image message.")
        return image

    encoding = getattr(msg, "encoding", "").lower()
    height = int(getattr(msg, "height", 0))
    width = int(getattr(msg, "width", 0))
    step = int(getattr(msg, "step", 0))

    encoding_map = {
        "mono8": (np.uint8, 1),
        "8uc1": (np.uint8, 1),
        "bgr8": (np.uint8, 3),
        "rgb8": (np.uint8, 3),
        "bgra8": (np.uint8, 4),
        "16uc1": (np.uint16, 1),
    }
    if encoding not in encoding_map:
        raise ValueError(f"Unsupported image encoding without cv_bridge: {encoding}")

    dtype, channels = encoding_map[encoding]
    itemsize = np.dtype(dtype).itemsize
    row_width = step // itemsize if step else width * channels
    image = np.frombuffer(msg.data, dtype=dtype).reshape(height, row_width)
    image = image[:, : width * channels]
    if channels == 1:
        return image.reshape(height, width)

    image = image.reshape(height, width, channels)
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def save_image_from_rosbag(
    reader: AnyReader,
    connection: Connection,
    input_msg_type: str,
    message_count: int,
    frames_folder: str = "frames",
) -> None:
    """
    Save an image from a ROS bag.

    Args:
        reader: Rosbag reader.
        connection: connection containing the image messages.
        input_msg_type: The type of message in the topic, e.g. "sensor_msgs/msg/Image".
        message_count: Total number of messages in the topic.

    Returns:
        None

    Raises:
        Exception: If an error occurs during image conversion or saving.

    Notes:
        This function queries a ROS 2 database for messages in a specified topic,
        deserializes them into OpenCV images, and saves them as PNG files.
    """
    for i, (conn, ts, raw) in enumerate(reader.messages(connections=[connection])):

        print(f"[INFO] - Extracting [{i+1}/{message_count}] …", end="\r")
        sys.stdout.flush()

        msg = reader.deserialize(raw, connection.msgtype)
        image_file_type = ".jpg" if getattr(msg, "format", "").lower() == "jpeg" else ".png"
        cv_image = decode_ros_image_message(msg, input_msg_type)

        padded_number = f"{i:07d}"
        output_filename = os.path.join(frames_folder, f"{padded_number}{image_file_type}")
        cv2.imwrite(output_filename, cv_image)


def check_and_create_folder(folder_path: str) -> None:
    """
    Check if a directory exists and create it if not.

    Args:
        folder_path: The path of the directory to be checked or created.

    Returns:
        None

    Notes:
        This function attempts to ensure that the specified directory is present.
        If it does not exist, an attempt is made to create it. Any errors during
        creation are logged and reported.

    Raises:
        OSError: If there's a problem creating the folder.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            if IS_VERBOSE:
                print(f"[INFO] - Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"[ERROR] - Failed to create folder '{folder_path}'. {e}")


def clear_folder_if_non_empty(folder_path: str) -> bool:
    """
    Check if a folder is non-empty. If it is, remove all its contents.

    Parameters:
        folder_path: The path of the folder to check and clear.

    Returns:
        True if the folder was cleared, False if it was already empty.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        if IS_VERBOSE:
            print(f"[WARN] - The folder '{folder_path}' does not exist.")
        return False

    # List all files and directories in the folder
    contents = os.listdir(folder_path)
    if contents:
        for item in contents:
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # Remove files
            else:
                shutil.rmtree(item_path)  # Remove directories
        if IS_VERBOSE:
            print(f"[INFO] - Cleared all contents from '{folder_path}'...")
        return True
    if IS_VERBOSE:
        print(f"[INFO] - The folder '{folder_path}' is already empty.")
    return False


def create_video_from_images(image_folder: str, output_video: str, pix_fmt: str, framerate: int = 30):
    """
    Creates a video from a list of images in the specified folder.

    Args:
        image_folder: The path to the folder containing the images.
        output_video: The desired file name for the generated video.
        pix_fmt: ffmpeg pixel format.
        framerate (optional): The frame rate of the resulting video. Defaults to 30.

    Returns:
        True if the operation was successful, False otherwise.
    """
    images = sorted(
        [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int(os.path.splitext(x)[0]),  # Sort by the numeric part of the filename
    )
    if not images:
        print("[WARN] - No images found in the specified folder.")
        return False

    # Create a temporary text file listing all images
    image_list_file = os.path.join(image_folder, "_images.txt")
    with open(image_list_file, "w", encoding="utf-8") as f:
        for path in images:
            f.write(f"file '{path}'\n")

    ffmpeg_bin = get_ffmpeg_executable()
    if ffmpeg_bin is None:
        print("[ERROR] - ffmpeg not found. Install ffmpeg or `uv add imageio-ffmpeg`.")
        os.remove(image_list_file)
        return False

    # Build the ffmpeg command
    command = [
        ffmpeg_bin,
        "-loglevel",
        "error" if not IS_VERBOSE else "info",
        "-stats",
        "-threads",
        "1",
        "-r",
        str(framerate),  # Set frame rate
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        image_list_file,  # Input list of images
        "-c:v",
        "libx264",
        "-pix_fmt",
        pix_fmt,
        output_video,
        "-y",
    ]

    if IS_VERBOSE:
        print("[INFO] -", " ".join(command))
    try:
        subprocess.run(command, check=True)
        print(f"[INFO] - Video written to {output_video}.")
        os.remove(image_list_file)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] - Error occurred: {e}")
        os.remove(image_list_file)
        return False


def create_video_from_jpg(
    reader: AnyReader,
    connection: Connection,
    output_video: str,
    fps: float,
    max_frames: int = -1,
):
    """
    Save an video from a ROS bag with jpg compressed images into a mjpeg video file.

    Args:
        reader: Rosbag reader.
        connection: Connection containing the image messages.
        output_video: Output video filepath.
        fps: The desired video framerate.
        max_frames (int, optional): stops export after this number of frames.

    Returns:
        None

    Raises:


    Notes:

    """
    ffmpeg_bin = get_ffmpeg_executable()
    if ffmpeg_bin is None:
        print("[ERROR] - ffmpeg not found. Install ffmpeg or `uv add imageio-ffmpeg`.")
        return False

    cmd = [
        ffmpeg_bin,
        "-loglevel",
        "error" if not IS_VERBOSE else "info",
        "-stats",
        "-threads",
        "1",
        "-r",
        str(fps),
        "-f",
        "mjpeg",
        "-i",
        "-",  # stdin
        "-c:v",
        "copy",
        "-an",
        output_video,
        "-y",
    ]
    return create_video_ffmpeg(cmd, reader, connection, output_video, max_frames)


def create_video_from_raw_image(
    reader: AnyReader,
    connection: Connection,
    output_video: str,
    fps: float,
    max_frames: int = -1,
):
    """Write raw ROS image messages to an MP4 file using ffmpeg over stdin.

    This avoids OpenCV VideoWriter backend issues (missing codec/container support).
    """
    if cv2 is None:
        print("[ERROR] - OpenCV is required for raw image export.")
        return False

    ffmpeg = None
    ffmpeg_bin = get_ffmpeg_executable()
    use_ffmpeg = ffmpeg_bin is not None
    frame_count = 0
    target_size = None
    writer = None
    writer_output = output_video

    for i, (conn, ts, raw) in enumerate(reader.messages(connections=[connection])):
        if 0 < max_frames <= i:
            break

        msg = reader.deserialize(raw, connection.msgtype)
        frame = decode_ros_image_message(msg, "sensor_msgs/msg/Image")

        # Normalize to 8-bit BGR for stable encoding.
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if frame.dtype != np.uint8:
            if frame.dtype == np.uint16:
                frame = cv2.convertScaleAbs(frame, alpha=255.0 / 65535.0)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        if ffmpeg is None and writer is None:
            height, width = frame.shape[:2]
            target_size = (width, height)
            if use_ffmpeg:
                cmd = [
                    ffmpeg_bin,
                    "-loglevel",
                    "error" if not IS_VERBOSE else "info",
                    "-stats",
                    "-threads",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "bgr24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    output_video,
                    "-y",
                ]
                if IS_VERBOSE:
                    print("[INFO] -", " ".join(cmd))
                ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                print("[WARN] - ffmpeg not found. Falling back to OpenCV VideoWriter codecs.")
                codec_candidates = [
                    ("mp4v", output_video),
                    ("avc1", output_video),
                    ("XVID", str(Path(output_video).with_suffix(".avi"))),
                    ("MJPG", str(Path(output_video).with_suffix(".avi"))),
                ]
                for fourcc_str, candidate_path in codec_candidates:
                    candidate_writer = cv2.VideoWriter(
                        candidate_path,
                        cv2.VideoWriter_fourcc(*fourcc_str),
                        fps,
                        target_size,
                    )
                    if candidate_writer.isOpened():
                        writer = candidate_writer
                        writer_output = candidate_path
                        if candidate_path != output_video:
                            print(f"[WARN] - Using fallback output path: {writer_output}")
                        break
                    candidate_writer.release()

                if writer is None:
                    print(f"[ERROR] - Could not open any VideoWriter codec for {output_video}.")
                    return False

        if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        frame = np.ascontiguousarray(frame)
        if ffmpeg is not None:
            try:
                ffmpeg.stdin.write(frame.tobytes())
            except BrokenPipeError:
                stderr_output = ffmpeg.stderr.read().decode("utf-8", errors="replace")
                print(f"[ERROR] - ffmpeg terminated early while writing raw frames.\n{stderr_output}")
                return False
        else:
            writer.write(frame)

        frame_count += 1

    if ffmpeg is not None:
        ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        if return_code != 0:
            stderr_output = ffmpeg.stderr.read().decode("utf-8", errors="replace")
            print(f"[ERROR] - ffmpeg exited with code {return_code}.\n{stderr_output}")
            return False
    if writer is not None:
        writer.release()

    if frame_count == 0:
        print("[ERROR] - No frames written for raw image topic.")
        return False

    print(f"[INFO] - Video written to {writer_output if writer is not None else output_video}.")
    return True

def create_video_ffmpeg(
    cmd: list[str],
    reader: AnyReader,
    connection: Connection,
    output_video: str,
    max_frames: int = -1,
    frame_serializer: Optional[Callable] = None,
) -> bool:
    if IS_VERBOSE:
        print("[INFO] -", " ".join(cmd))
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    if frame_serializer is None:
        frame_serializer = lambda msg: msg.data

    for i, (conn, ts, raw) in enumerate(reader.messages(connections=[connection])):
        if 0 < max_frames <= i:
            break
        msg = reader.deserialize(raw, connection.msgtype)
        try:
            ffmpeg.stdin.write(frame_serializer(msg))
        except BrokenPipeError:
            stderr_output = ffmpeg.stderr.read().decode("utf-8", errors="replace")
            print(f"[ERROR] - ffmpeg terminated early while writing frames.\n{stderr_output}")
            return False

    ffmpeg.stdin.close()
    return_code = ffmpeg.wait()
    if return_code != 0:
        stderr_output = ffmpeg.stderr.read().decode("utf-8", errors="replace")
        print(f"[ERROR] - ffmpeg exited with code {return_code}.\n{stderr_output}")
        return False

    print(f"[INFO] - Video written to {output_video}.")
    return True

def export_all_image_topics(       
    bag_path: Path,
    args:argparse.ArgumentParser
):
    # Process the bag
    with AnyReader([bag_path], default_typestore=DEFAULT_TYPESTORE) as reader:
        for c in reader.connections:
            message_count, msg_type, conn = get_topic_info(reader, c.topic)
            msg_encoding, msg = get_msg_format_from_rosbag(reader, conn)
            if bag_path.is_file() :
                ofile = bag_path.with_name(bag_path.stem + c.topic.replace("/","_")+".mp4")
            elif bag_path.is_dir() :
                ofile = bag_path / (c.topic.replace("/","_")+".mp4")
            if(ofile.exists()):
                continue
            if (
                msg_type.endswith("CompressedImage")
                and ("jpeg" in msg_encoding.lower() or "jpg" in msg_encoding.lower())
            ):
                if IS_VERBOSE:
                    try:
                        print(f"{c.topic} msg_type: {msg_type} msg_encoding: {msg_encoding}")
                    except:
                        pass
                print(f"exporting {c.topic} to file {ofile}" )
                # we can directly feed the jpg data to ffmpeg to create the video
                create_video_from_jpg(reader, conn, str(ofile), args.rate)
            elif (
                msg_type.endswith("sensor_msgs/msg/Image")
                and msg_encoding != ""
            ):
                try:
                    print(f"exporting {c.topic} to file {ofile}" )
                    create_video_from_raw_image(
                        reader,
                        conn,
                        str(ofile),
                        args.rate,
                        args.frames,
                    )
                except Exception as e:
                    print(f"failed exporting {c.topic} to file {ofile} with error:", e )


if __name__ == "__main__":
    # Parse commandline input arguments.
    parser = argparse.ArgumentParser(
        prog="rosbag2video",
        description="Convert ROS bag (1/2) to video using ffmpeg.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", required=False, default=False,
                        help="Run rosbag2video script in verbose mode.")
    parser.add_argument("-r", "--rate", type=int, required=False, default=30,
                        help="Video framerate")
    parser.add_argument("-t", "--topic", type=str, required=False,
                        help="Topic Name")
    parser.add_argument("-o", "--ofile", type=str, required=False, default="output_video.mp4",
                        help="Output File")
    parser.add_argument("--save_images", action="store_true", required=False, default=False,
                        help="Boolean flag for saving extracted .png frames in frames/")
    parser.add_argument("--frames", type=int, required=False, default=-1,
                        help="Limit the number of frames to export")
    parser.add_argument('rosbag',type=str, help="Input Bag(s)", nargs="+")
    args = parser.parse_args(sys.argv[1:])

    IS_VERBOSE = args.verbose

    # Check if input fps is valid.
    if args.rate <= 0:
        print(f"[WARN] - Invalid rate {args.rate}; using 30 FPS.")
        args.rate = 30

    # Check if bag exists

    if not args.topic:
        for bag in args.rosbag:
            try:
                bag_root = Path(bag).expanduser().resolve()
            except Exception as e:
                print(e)
                continue

            bag_paths = find_bag_paths(bag_root)
            for bag_path in bag_paths:
                if IS_VERBOSE:
                    print(f"extracting from rosbag: {bag_path}")
                try:
                    export_all_image_topics(bag_path, args)
                except Exception as e:
                    print(e)
        exit(0)

    out_basename = Path(args.ofile).name
    for bag in args.rosbag:
        bag_root = Path(bag).expanduser().resolve()
        bag_paths = find_bag_paths(bag_root)
        for bag_path in bag_paths:
            if not bag_path.exists():
                print(f"[ERROR] - Path '{bag_path}' does not exist.")
                continue

            # determine output file for this bag (place inside bag directory)
            if bag_path.is_dir():
                target_ofile = bag_path / out_basename
            else:
                target_ofile = bag_path.with_name(out_basename)

            # Process the bag
            with AnyReader([bag_path], default_typestore=DEFAULT_TYPESTORE) as reader:
                message_count, msg_type, conn = get_topic_info(reader, args.topic)

                msg_encoding, msg = get_msg_format_from_rosbag(reader, conn)
                if (
                    msg_type.endswith("CompressedImage")
                    and not args.save_images
                    and msg_encoding in ("jpeg", "jpg")
                ):
                    # we can directly feed the jpg data to ffmpeg to create the video
                    create_video_from_jpg(reader, conn, str(target_ofile), args.rate, args.frames)
                elif (
                    msg_type.endswith("sensor_msgs/msg/Image")
                    and not args.save_images
                    and msg is not None
                    and msg_encoding != ""
                ):
                    # Stream raw frames directly from bag to ffmpeg to avoid the heavy concat path.
                    if not create_video_from_raw_image(
                        reader,
                        conn,
                        str(target_ofile),
                        args.rate,
                        args.frames,
                    ):
                        print("[ERROR] - Could not generate video.")
                else:
                    # else do the image export stuff - extract frames, then ffmpeg concat
                    FRAMES_FOLDER = str(bag_path / "frames")
                    check_and_create_folder(FRAMES_FOLDER)
                    clear_folder_if_non_empty(FRAMES_FOLDER)

                    save_image_from_rosbag(reader, conn, msg_type, message_count, frames_folder=FRAMES_FOLDER)
                    # Construct video from image sequence
                    pix_fmt = "yuv420p"
                    if not create_video_from_images(FRAMES_FOLDER, str(target_ofile), pix_fmt, framerate=args.rate):
                        print("[ERROR] - Could not generate video.")

                    # Keep or remove frames folder content based on --save-images flag.
                    if not args.save_images:
                        clear_folder_if_non_empty(FRAMES_FOLDER)

