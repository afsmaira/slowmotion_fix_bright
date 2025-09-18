import argparse
import os

import cv2
import numpy as np

from tqdm import tqdm
from typing import Tuple


class FlickerFixer:
    def __init__(self, input_path: str, output_path: str, slices: int = 20):
        self.input = input_path
        self.output = output_path
        self.num_slices = slices
        self.props = None

    def getProps(self):
        """
        Gets video properties
        """
        cap = cv2.VideoCapture(self.input)
        if not cap.isOpened():
            raise Exception(f"Could not open video file at '{self.input}'")

        self.props = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        print(f"Video properties: {self.props['width']}x{self.props['height']}, "
              f"{self.props['frame_count']} frames, {self.props['fps']:.2f} FPS.")

    def slice_idxs(self, i):
        slice_width = self.props['width'] // self.num_slices
        return ((i * slice_width),
                ((i + 1) * slice_width
                 if i < self.num_slices - 1
                 else self.props['width']))


    def analyze(self):
        """
        First pass: Analyzes the video to gather brightness data for each slice
        """
        print("\n--- Starting Step 1: Brightness Analysis ---")
        cap = cv2.VideoCapture(self.input)

        brightness_data_list = []
        slice_width = self.props['width'] // self.num_slices

        for _ in tqdm(range(self.props['frame_count']), desc="Analyzing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_frame_brightness = []
            for i in range(self.num_slices):
                start_x, end_x = self.slice_idxs(i)
                avg_brightness = np.mean(gray_frame[:, start_x:end_x])
                current_frame_brightness.append(avg_brightness)
            brightness_data_list.append(current_frame_brightness)

        cap.release()

        self.brightness_data = np.array(brightness_data_list)
        self.max_brightness_per_slice = np.max(self.brightness_data, axis=0)
        print("--- Analysis complete. Reference brightness values calculated. ---")

    def process(self):
        """
        Second pass: Applies correction based on analysis and writes the new video.
        """
        print("\n--- Starting Step 2: Applying Correction & Writing Video ---")
        cap = cv2.VideoCapture(self.input)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output, fourcc, self.props['fps'],
                              (self.props['width'], self.props['height']))

        for frame_idx in tqdm(range(self.props['frame_count']), desc="Writing fixed frames"):
            ret, frame = cap.read()
            if not ret:
                break

            corrected_frame = frame.copy()
            for slice_idx in range(self.num_slices):
                current_brightness = self.brightness_data[frame_idx, slice_idx]
                target_brightness = self.max_brightness_per_slice[slice_idx]
                correction_factor = target_brightness / current_brightness if current_brightness > 0 else 1.0

                start_x, end_x = self.slice_idxs(slice_idx)
                corrected_slice_float = frame[:, start_x:end_x].astype(np.float32) * correction_factor
                corrected_frame[:, start_x:end_x] = np.clip(corrected_slice_float, 0, 255).astype(np.uint8)

            out.write(corrected_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def fix(self):
        print("Starting flicker process...")
        self.getProps()
        self.analyze()
        self.process()
        print(f"\nProcess finished successfully! Corrected video saved to: '{self.output}'")


def process_input() -> Tuple[str, str, int]:
    """
    Processes the input parameters and returns the main function parameters
    :return: input_path, output_path, slices_number
    """
    parser = argparse.ArgumentParser(
        description="A tool to correct brightness flicker in slow-motion videos caused by artificial lighting.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to the input video file."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to the output video file.\n"
             "Default: '<input>_fixed.mp4' in the same directory."
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "-s", "--slices",
        type=int,
        help="Number of vertical slices to analyze.\n"
             "More slices provide more localized correction but might introduce artifacts."
    )

    group.add_argument(
        "-w", "--slice-width",
        type=int,
        default=20,
        help="Specify the width of each slice in pixels.\n"
             "This is an alternative to --slices. The number of slices will be calculated automatically.\n"
             "Default: 20"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video file '{args.input}' does not exist.")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_fixed.mp4"

    if args.slice_width:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise Exception("Input video file cannot be opened.")
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        if args.slice_width <= 0:
            raise ValueError("Slice width must be a positive integer.")
        if args.slice_width > video_width:
            args.slice_width = video_width

        num_slices = video_width // args.slice_width
        print(f"Using slice width of {args.slice_width}px, resulting in {num_slices} slices for this video.")
        if video_width % args.slice_width != 0:
            print(
                f"Warning: Video width ({video_width}px) is not perfectly divisible by slice width. The last slice will be wider.")

    else:
        num_slices = args.slices
        print(f"Using {num_slices} vertical slices.")

    return args.input, args.output, num_slices


if __name__ == '__main__':
    FlickerFixer(*process_input()).fix()
