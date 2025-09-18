import argparse
import os

import cv2
import numpy as np

from tqdm import tqdm
from typing import Tuple, Dict


class FlickerFixer:
    def __init__(self, input_path: str, output_path: str, mode: str = 'grid',
                 slices: int = 20, grid_dims: tuple = (15, 20)):
        """
        Creates an instance of FlickerFixer
        :param input_path: Path of input file
        :param output_path: Path of output file
        :param mode: Processing mode ("grid" or "slices")
        :param slices: Slices number in case of slices processing
        :param grid_dims: Grid dimensions in case of grid processing
        """
        self.input = input_path
        self.output = output_path
        self.num_slices = slices
        self.mode = mode
        self.grid_dims = grid_dims
        print(self.mode, self.grid_dims, self.num_slices)
        self.props = {}
        self.brightness_data = None
        self.max_brightness = None

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

    def slice_idxs(self, i, j=None):
        if self.mode == 'slices':
            slice_width = self.props['width'] // self.num_slices
            return ((i * slice_width),
                    ((i + 1) * slice_width
                     if i < self.num_slices - 1
                     else self.props['width']))
        slice_width = self.props['width'] // self.grid_dims[0]
        slice_height = self.props['height'] // self.grid_dims[1]
        return ((i * slice_width),
                ((i + 1) * slice_width
                 if i < self.grid_dims[0] - 1
                 else self.props['width']),
                (j * slice_height),
                 ((j + 1) * slice_height
                  if j < self.grid_dims[1] - 1
                  else self.props['height']))

    def _analyze_slices(self, cap):
        brightness_data_list = []
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

        self.brightness_data = np.array(brightness_data_list)
        self.max_brightness_per_slice = np.max(self.brightness_data, axis=0)

    def _analyze_grid(self, cap):
        cols, rows = self.grid_dims
        data = np.zeros((self.props['frame_count'], rows, cols))

        for frame_idx in tqdm(range(self.props['frame_count']), desc="Analyzing (grid mode)"):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for r in range(rows):
                for c in range(cols):
                    start_x, end_x, start_y, end_y = self.slice_idxs(c, r)
                    cell = gray[start_y:end_y, start_x:end_x]
                    data[frame_idx, r, c] = np.mean(cell)
        self.brightness_data = data
        self.max_brightness = np.max(self.brightness_data, axis=0)

    def analyze(self):
        """
        First pass: Analyzes the video to gather brightness data for each slice
        """
        print("\n--- Starting Step 1: Brightness Analysis ---")
        cap = cv2.VideoCapture(self.input)
        if self.mode == 'slices':
            self._analyze_slices(cap)
        else:
            self._analyze_grid(cap)
        cap.release()
        print("--- Analysis complete. Reference brightness values calculated. ---")

    def _process_slices(self, cap, out):
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

    def _process_grid(self, cap, out):
        cols, rows = self.grid_dims
        for frame_idx in tqdm(range(self.props['frame_count']), desc="Correcting (grid mode)"):
            ret, frame = cap.read()
            if not ret:
                break
            corrected = frame.copy()
            for r in range(rows):
                for c in range(cols):
                    current_brightness = self.brightness_data[frame_idx, r, c]
                    if current_brightness > 0:
                        factor = self.max_brightness[r, c] / current_brightness
                        start_x, end_x, start_y, end_y = self.slice_idxs(c, r)
                        region = corrected[start_y:end_y, start_x:end_x]
                        corrected[start_y:end_y, start_x:end_x] = np.clip(region.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            out.write(corrected)

    def process(self):
        """
        Second pass: Applies correction based on analysis and writes the new video.
        """
        print("\n--- Starting Step 2: Applying Correction & Writing Video ---")
        cap = cv2.VideoCapture(self.input)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output, fourcc, self.props['fps'],
                              (self.props['width'], self.props['height']))
        if self.mode == 'slices':
            self._process_slices(cap, out)
        else:
            self._process_grid(cap, out)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def fix(self):
        print("Starting flicker process...")
        self.getProps()
        self.analyze()
        self.process()
        print(f"\nProcess finished successfully! Corrected video saved to: '{self.output}'")


def process_input() -> Dict[str, str | int | Tuple[int, int]]:
    """
    Processes the input parameters and returns the main function parameters
    :return: input_path, output_path, mode, numbers of slices or cells
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

    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="grid",
        choices=['slice', 'grid'],
        help="Correction mode.\n'slice' for vertical slices, 'grid' for a grid of regions.\nDefault: grid"
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

    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument(
        "-g", "--grid-size",
        type=str,
        help="[grid mode] Grid dimensions as COLUMNSxROWS (e.g., '15x20')."
    )
    grid_group.add_argument(
        "--grid-cell-size",
        type=str,
        default='20x20',
        help="[grid mode] Size of each grid cell in pixels as WIDTHxHEIGHT (e.g., '20x20').\nDefault: 20x20"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video file '{args.input}' does not exist.")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_fixed.mp4"

    if args.mode == 'slice':
        if args.slice_width:
            cap = cv2.VideoCapture(args.input)
            if not cap.isOpened():
                raise Exception("Input video file cannot be opened.")
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()

            if args.slice_width <= 0:
                raise ValueError("Slice width must be a positive integer.")
            args.slice_width = min(video_width, args.slice_width)
            num_slices = video_width // args.slice_width
            print(f"Using slice width of {args.slice_width}px, resulting in {num_slices} slices for this video.")
            if video_width % args.slice_width != 0:
                print(
                    f"Warning: Video width ({video_width}px) is not perfectly divisible by slice width. The last slice will be wider.")

        else:
            num_slices = args.slices
            print(f"Using {num_slices} vertical slices.")

        return {'input_path': args.input,
                'output_path': args.output,
                'mode': args.mode,
                'slices': num_slices
                }

    elif args.mode == 'grid':
        if args.grid_cell_size:
            cap = cv2.VideoCapture(args.input)
            if not cap.isOpened():
                raise Exception("Input video file cannot be opened.")
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            dims_grid = tuple(map(int, args.grid_cell_size.split('x')))
            if dims_grid[0] <= 0 or dims_grid[1] <= 0:
                raise ValueError("Grid cells dimensions must be positive integers.")
            dims_grid = (min(dims_grid[0], video_width),
                         min(dims_grid[1], video_height)
                         )

            nums_grid = (video_width // dims_grid[0],
                         video_height // dims_grid[1]
                         )
            print(f"Using grid cells dimensions of {dims_grid[0]}x{dims_grid[1]}, resulting in {nums_grid[0]}x{nums_grid[1]} grid cells for this video.")
            if (video_width % dims_grid[0]) + (video_height % dims_grid[1]) > 0:
                print(
                    "Warning: Video dimensions are not perfectly divisible by grid cells dimensions. The last grid cell will be greater.")
        else:
            nums_grid = tuple(map(int, args.grid_size.split('x')))
            print(f"Using dimensions {nums_grid} for the grid.")

        return {'input_path': args.input,
                'output_path': args.output,
                'mode': args.mode,
                'grid_dims': nums_grid
                }
    else:
        raise ValueError('Mode must be either "grid" or "slices"')


if __name__ == '__main__':
    FlickerFixer(**process_input()).fix()
