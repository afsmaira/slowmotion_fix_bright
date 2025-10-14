import os
import math

import cv2
import numpy as np

from tqdm import tqdm
from typing import Tuple

import concurrent.futures


def analyze_frame_worker(args):
    """
    Worker function to analyze a chunk of frames for a given mode.
    """
    video_path, start_frame, end_frame, mode, params = args

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    chunk_data = []

    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if mode == 'slice':
            orientation = params['orientation']
            num_divisions = params['slices']

            if orientation == 'vertical':
                slice_width = width // num_divisions
                frame_data = [np.mean(gray[:, i*slice_width : width if i == num_divisions-1 else (i+1)*slice_width]) for i in range(num_divisions)]
            else:
                row_height = height // num_divisions
                frame_data = [np.mean(gray[i*row_height : height if i == num_divisions-1 else (i+1)*row_height, :]) for i in range(num_divisions)]
            chunk_data.append(frame_data)

        elif mode == 'grid':
            cols, rows = params['grid_dims']
            cell_w = width // cols
            cell_h = height // rows
            frame_data = np.zeros((rows, cols))
            for r in range(rows):
                for c in range(cols):
                    start_y, end_y = r * cell_h, height if r == rows - 1 else (r + 1) * cell_h
                    start_x, end_x = c * cell_w, width if c == cols - 1 else (c + 1) * cell_w
                    cell = gray[start_y:end_y, start_x:end_x]
                    if cell.size > 0:
                        frame_data[r, c] = np.mean(cell)
            chunk_data.append(frame_data)

    cap.release()
    return np.array(chunk_data)


def correct_frames_worker(args):
    """
    Worker function to correct a chunk of frames.
    """
    video_path, start_frame, end_frame, mode, params, brightness_data, max_brightness = args

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    corrected_frames = []

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        corrected = frame.copy()

        if mode == 'slice':
            orientation = params['orientation']
            num_divisions = params['slices']

            if orientation == 'vertical':
                slice_width = width // num_divisions
                for i in range(num_divisions):
                    factor = max_brightness[i] / brightness_data[frame_idx, i] if brightness_data[frame_idx, i] > 0 else 1
                    start_x = i * slice_width
                    end_x = width if i == num_divisions - 1 else (i + 1) * slice_width
                    region = corrected[:, start_x:end_x]
                    corrected[:, start_x:end_x] = np.clip(region.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            else:
                row_height = height // num_divisions
                for i in range(num_divisions):
                    factor = max_brightness[i] / brightness_data[frame_idx, i] if brightness_data[frame_idx, i] > 0 else 1
                    start_y = i * row_height
                    end_y = height if i == num_divisions - 1 else (i + 1) * row_height
                    region = corrected[start_y:end_y, :]
                    corrected[start_y:end_y, :] = np.clip(region.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        elif mode == 'grid':
            cols, rows = params['grid_dims']
            cell_w = width // cols
            cell_h = height // rows
            for r in range(rows):
                for c in range(cols):
                    current_b = brightness_data[frame_idx, r, c]
                    if current_b > 0:
                        factor = max_brightness[r, c] / current_b
                        start_y, end_y = r * cell_h, height if r == rows - 1 else (r + 1) * cell_h
                        start_x, end_x = c * cell_w, width if c == cols - 1 else (c + 1) * cell_w
                        region = corrected[start_y:end_y, start_x:end_x]
                        if region.size > 0:
                            corrected[start_y:end_y, start_x:end_x] = np.clip(region.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        corrected_frames.append(corrected)

    cap.release()
    return corrected_frames


class FlickerFixer:
    def __init__(self, input_path: str, output_path: str, mode: str = 'grid',
                 slices: int = 20, grid_dims: Tuple[int, int] = (15, 20),
                 workers: int = 0, chunk_size: int = 10,
                 slice_orientation: str = "vertical"):
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
        self.props = {}
        self.brightness_data = None
        self.max_brightness = None
        self.workers = workers if workers > 0 else os.cpu_count() or 1
        self.chunk_frames = chunk_size
        self.slice_orientation = slice_orientation

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

    def analyze(self):
        print(f"\n--- Starting Step 1: Analyzing frames in parallel ({self.mode} mode) ---")

        frame_count = self.props['frame_count']
        chunk_size = math.ceil(frame_count / self.workers)
        tasks = []

        if self.mode == 'slice':
            params = {'slices': self.num_slices, 'orientation': self.slice_orientation}
        else:
            params = {'grid_dims': self.grid_dims}

        for start_frame in range(0, frame_count, self.chunk_frames):
            end_frame = min(start_frame + self.chunk_frames, frame_count)
            if start_frame >= end_frame:
                continue
            tasks.append((self.input, start_frame, end_frame, self.mode, params))

        all_chunks_data = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            results = list(tqdm(executor.map(analyze_frame_worker, tasks), total=len(tasks), desc="Analyzing chunks"))
            for chunk_result in results:
                if chunk_result is not None:
                    all_chunks_data.append(chunk_result)

        self.brightness_data = np.vstack(all_chunks_data)
        self.max_brightness = np.max(self.brightness_data, axis=0)
        print("--- Analysis complete. ---")


    def process(self):
        print(f"\n--- Starting Step 2: Correcting frames in parallel and writing video ---")

        frame_count = self.props['frame_count']
        tasks = []

        if self.mode == 'slice':
            params = {'slices': self.num_slices, 'orientation': self.slice_orientation}
        else:
            params = {'grid_dims': self.grid_dims}
        for start_frame in range(0, frame_count, self.chunk_frames):
            end_frame = min(start_frame + self.chunk_frames, frame_count)
            if start_frame >= end_frame:
                continue
            tasks.append((self.input, start_frame, end_frame, self.mode, params, self.brightness_data, self.max_brightness))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output, fourcc, self.props['fps'], (self.props['width'], self.props['height']))

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            # executor.map garante que os resultados cheguem na ordem em que as tarefas foram enviadas
            for corrected_frames_chunk in tqdm(executor.map(correct_frames_worker, tasks), total=len(tasks), desc="Correcting chunks"):
                if corrected_frames_chunk:
                    for frame in corrected_frames_chunk:
                        out.write(frame)

        out.release()
        cv2.destroyAllWindows()

    def fix(self):
        print("Starting flicker process...")
        self.getProps()
        self.analyze()
        self.process()
        print(f"\nProcess finished successfully! Corrected video saved to: '{self.output}'")
