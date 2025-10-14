import os
import cv2
import argparse
from typing import Dict, Tuple

from .fixer import FlickerFixer


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
        choices=['slice', 'grid', 'full'],
        help="Correction mode.\n"
             "'slice': for vertical/horizontal slices.\n"
             "'grid': for a grid of regions.\n"
             "'full': analyzes the entire frame as one region.\n"
             "Default: grid"
    )

    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--slice-orientation",
        type=str,
        default="vertical",
        choices=['vertical', 'horizontal'],
        help="[slice mode] Orientation of the slices.\nDefault: vertical"
    )
    group.add_argument(
        "-d", "--divisions",
        type=int,
        help="[slice mode] Number of vertical slices to analyze.\n"
             "More slices provide more localized correction but might introduce artifacts."
    )
    group.add_argument(
        "-w", "--division-size",
        type=int,
        default=20,
        help="[slice mode] Specify the width of each slice in pixels.\n"
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

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of CPU cores to use for processing.\nDefault: 0 (uses all available cores)."
    )

    parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        default=10,
        help="Number of frames per parallel task.\nSmaller values give a smoother progress bar and less use of RAM but may add overhead.\nDefault: 10"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video file '{args.input}' does not exist.")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_fixed.mp4"

    ret = {'input_path': args.input,
           'output_path': args.output,
           'mode': args.mode,
           'workers': args.workers,
           'chunk_size': args.chunk_size
           }
    if args.mode == 'full':
        print("Using full frame (1x1 grid) correction mode.")
        ret['mode'] = 'grid'
        ret['grid_dims'] = (1, 1)
    elif args.mode == 'slice':
        if args.division_size:
            cap = cv2.VideoCapture(args.input)
            if not cap.isOpened():
                raise Exception("Input video file cannot be opened.")
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if args.division_size <= 0:
                raise ValueError("Slice width must be a positive integer.")
            if args.slice_orientation == 'vertical':
                args.division_size = min(video_width, args.division_size)
                num_slices = video_width // args.division_size
                if video_width % args.division_size != 0:
                    print(
                        f"Warning: Video width ({video_width}px) is not perfectly divisible by slice width. The last slice will be wider.")
            else:
                args.division_size = min(video_height, args.division_size)
                num_slices = video_height // args.division_size
                if video_height % args.division_size != 0:
                    print(
                        f"Warning: Video height ({video_height}px) is not perfectly divisible by slice height. The last slice will be higher.")
            print(f"Using slice size of {args.division_size}px, resulting in {num_slices} slices for this video.")

        else:
            num_slices = args.slices
            print(f"Using {num_slices} vertical slices.")

        ret['slices'] = num_slices
        ret['slice_orientation'] = args.slice_orientation

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
            print(
                f"Using grid cells dimensions of {dims_grid[0]}x{dims_grid[1]}, resulting in {nums_grid[0]}x{nums_grid[1]} grid cells for this video.")
            if (video_width % dims_grid[0]) + (video_height % dims_grid[1]) > 0:
                print(
                    "Warning: Video dimensions are not perfectly divisible by grid cells dimensions. The last grid cell will be greater.")
        else:
            nums_grid = tuple(map(int, args.grid_size.split('x')))
            print(f"Using dimensions {nums_grid} for the grid.")

        ret['grid_dims'] = nums_grid
    else:
        raise ValueError('Mode must be either "grid" or "slices"')

    return ret


def main():
    FlickerFixer(**process_input()).fix()
