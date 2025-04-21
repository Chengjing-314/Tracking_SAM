"""
Heavily inspired by diffusion_policy.real_world.single_realsense.py
https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/real_world/single_realsense.py
"""

import datetime
import enum
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from multiprocessing.managers import SharedMemoryManager
from typing import Literal, Optional

import cv2
import h5py
import numpy as np
import pyrealsense2 as rs
from jaxtyping import Float

from uni_teleop.constants import DATA_ROOT
from uni_teleop.shared_memory.shared_memory_queue import Empty, SharedMemoryQueue
from uni_teleop.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from uni_teleop.shared_memory.shared_ndarray import SharedNDArray
from uni_teleop.utils.video_io import VideoRecorder, VideoRecorderConfig


def unproject_points(
    pixels: Float[np.ndarray, "... 2"], depth: Float[np.ndarray, "..."], intr_mat: Float[np.ndarray, "... 3 3"]
) -> Float[np.ndarray, "... 3"]:
    """
    Unproject pixels in the image plane to 3D points in the camera space using NumPy arrays.

    Args:
        pixels: Pixels in the image plane, could be Nx2 or BxNx2. The order is uv rather than xy.
        depth: Depth in the camera space, could be N, Nx1, BxN or BxNx1.
        intr_mat: Intrinsic matrix, could be 3x3 or Bx3x3.

    Returns:
        pts: 3D points, Nx3 or BxNx3.
    """
    if depth.ndim < pixels.ndim:
        depth = depth[..., None]
    principal_point = np.expand_dims(intr_mat[..., :2, 2], axis=-2)
    focal_length = np.concatenate([intr_mat[..., 0:1, 0:1], intr_mat[..., 1:2, 1:2]], axis=-1)
    xys = (pixels - principal_point) * depth / focal_length
    pts = np.concatenate([xys, depth], axis=-1)
    return pts


class Command(enum.Enum):
    START_RECORDING = 0
    STOP_RECORDING = 1


@dataclass
class RealsenseConfig:
    color_width: int = 960
    color_height: int = 540
    depth_width: int = 640
    depth_height: int = 480
    capture_fps: int = 30
    align_to: Literal["color", "depth"] = "color"
    serial_number: Optional[str] = None
    get_max_k: int = 30
    depth_zrange = (0.1, 2.0)

    enable_color: bool = True
    enable_depth: bool = True
    enable_visualization: bool = True
    enable_visualization_color_clipping: bool = True
    enable_recording: bool = True
    enable_depth_recording: bool = True  # only used if enable_recording is True
    enable_point_cloud_recording: bool = True  # only used if enable_recording is True
    num_points_per_frame: int = 16384

    visualization_depth_clipping_distance_meters: float = 1.0
    recording_output_root: str = str(DATA_ROOT / "realsense_recordings")
    recording_date: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    recording_date_time: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    verbose: bool = False

    @property
    def aligned_width(self):
        return self.color_width if self.align_to == "color" else self.depth_width

    @property
    def aligned_height(self):
        return self.color_height if self.align_to == "color" else self.depth_height

    @property
    def recording_output_dir(self):
        return os.path.join(self.recording_output_root, self.recording_date, self.recording_date_time)

    def __post_init__(self):
        if not self.enable_recording and (self.enable_depth_recording or self.enable_point_cloud_recording):
            raise ValueError(
                "enable_recording is False, but enable_depth_recording or enable_point_cloud_recording is True"
            )
        if self.enable_depth_recording and not self.enable_depth:
            raise ValueError("enable_depth_recording is True, but enable_depth is False")
        if self.enable_point_cloud_recording and not self.enable_depth:
            raise ValueError("enable_point_cloud_recording is True, but enable_depth is False")


class Realsense(mp.Process):
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        config: Optional[RealsenseConfig] = None,
    ):
        super().__init__()

        self.shm_manager = shm_manager
        if config is None:
            self.config = RealsenseConfig()
        else:
            self.config = config

        shape = (self.config.aligned_height, self.config.aligned_width)
        examples = {}
        if self.config.enable_color:
            examples["color"] = np.empty(shape=shape + (3,), dtype=np.uint8)
        if self.config.enable_depth:
            examples["depth"] = np.empty(shape=shape, dtype=np.uint16)
        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0

        self.data_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=self.config.get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=self.config.capture_fps * 2,
        )
        self.vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=self.config.capture_fps * 2,
        )

        # create command queue
        examples = {
            "cmd": Command.START_RECORDING.value,
            "option_enum": rs.option.exposure.value,
            "option_value": 0.0,
            "video_path": np.array("a" * 4096),  # linux path has a limit of 4096 bytes
            "recording_start_time": 0.0,
            "put_start_time": 0.0,
        }
        self.command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=examples, buffer_size=128
        )

        # (fx, fy, ppx, ppy, height, width, depth_scale)
        self.intrinsics_array = SharedNDArray.create_from_shape(mem_mgr=shm_manager, shape=(7,), dtype=np.float64)
        self.intrinsics_array.get()[:] = 0
        self.intrinsics_array_depth = SharedNDArray.create_from_shape(mem_mgr=shm_manager, shape=(7,), dtype=np.float64)
        self.intrinsics_array_depth.get()[:] = 0
        # self.vertices = SharedNDArray.create_from_shape(mem_mgr=shm_manager, shape=(307200,3), dtype=np.float32)
        self.vertices = SharedNDArray.create_from_shape(mem_mgr=shm_manager, shape=(518400, 3), dtype=np.float32)

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.start_recording_event = mp.Event()

    # ========= user API ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    @property
    def is_stopped(self):
        return self.stop_event.is_set()

    def get(self, k: Optional[int] = None, out=None):
        if k is None:
            return self.data_ring_buffer.get(out=out)
        else:
            return self.data_ring_buffer.get_last_k(k, out=out)

    def start_recording(self):
        assert self.config.enable_color
        self.command_queue.put({"cmd": Command.START_RECORDING.value})

    def stop_recording(self):
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main process ===========
    def run(self):
        if self.config.enable_visualization:
            display_proc = mp.Process(target=self._visualize)
            display_proc.start()

        if self.config.enable_recording:
            video_recorder = None
            depth_video_recorder = None
            point_cloud_h5 = None
            point_cloud_dataset = None
            self.start_recording_event.clear()

        if self.config.align_to == "color":
            rs_align = rs.align(rs.stream.color)
        elif self.config.align_to == "depth":
            rs_align = rs.align(rs.stream.depth)

        rs_config = rs.config()
        if self.config.enable_color:
            rs_config.enable_stream(
                rs.stream.color,
                self.config.color_width,
                self.config.color_height,
                rs.format.bgr8,
                self.config.capture_fps,
            )
        if self.config.enable_depth:
            rs_config.enable_stream(
                rs.stream.depth,
                self.config.depth_width,
                self.config.depth_height,
                rs.format.z16,
                self.config.capture_fps,
            )

        if self.config.verbose:
            print(f"[Realsense] {self.config.serial_number or ''} starting...")

        # start pipeline
        try:
            if self.config.serial_number is not None:
                rs_config.enable_device(self.config.serial_number)
            else:
                rs_pipeline = rs.pipeline()
                rs_pipeline_profile = rs_pipeline.start(rs_config)

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            d = rs_pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)

            color_stream = rs_pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ["fx", "fy", "ppx", "ppy", "height", "width"]
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.config.enable_depth:
                depth_sensor = rs_pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale

                # intrinsics for depth
                depth_stream = rs_pipeline_profile.get_stream(rs.stream.depth)
                intr_depth = depth_stream.as_video_stream_profile().get_intrinsics()
                order = ["fx", "fy", "ppx", "ppy", "height", "width"]
                for i, name in enumerate(order):
                    self.intrinsics_array_depth.get()[i] = getattr(intr_depth, name)
                self.intrinsics_array_depth.get()[-1] = depth_scale

            intr_mat = np.eye(3)
            intr_mat[0, 0] = self.intrinsics_array.get()[0]
            intr_mat[1, 1] = self.intrinsics_array.get()[1]
            intr_mat[0, 2] = self.intrinsics_array.get()[2]
            intr_mat[1, 2] = self.intrinsics_array.get()[3]
            pixels = np.stack(
                np.meshgrid(np.arange(self.config.aligned_width), np.arange(self.config.aligned_height)), axis=-1
            )

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                # wait for frames to come in
                frameset = rs_pipeline.wait_for_frames()
                receive_time = time.time()

                # align frames
                frameset = rs_align.process(frameset)

                # grab data
                data = dict()
                data["camera_receive_timestamp"] = receive_time
                # realsense report in ms
                data["camera_capture_timestamp"] = frameset.get_timestamp() / 1000

                if self.config.enable_color:
                    color_frame = frameset.get_color_frame()
                    data["color"] = np.asarray(color_frame.get_data())
                    t = color_frame.get_timestamp() / 1000
                    data["camera_capture_timestamp"] = t
                if self.config.enable_depth:
                    data["depth"] = np.asarray(frameset.get_depth_frame().get_data())

                step_idx = int((receive_time - t_start) * self.config.capture_fps)
                data["step_idx"] = step_idx
                data["timestamp"] = receive_time

                # ------ get point cloud from realsense sdk ------
                pts = rs.pointcloud()
                pts.map_to(color_frame)
                # print("pts", pts)
                depth_frame = frameset.get_depth_frame()
                # print("depth_data", depth_data)
                points = pts.calculate(depth_frame)
                vertices = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
                # print("all zero", np.all(vertices == 0))
                # print("vertices origin shape", vertices.shape)
                # indices = np.random.choice(vertices.shape[0], 4096, replace=False)
                # self.vertices = vertices[indices]
                self.vertices.get()[:] = vertices
                # -------------------------------------------------

                # put data to ring buffer
                self.data_ring_buffer.put(data, wait=False)
                self.vis_ring_buffer.put(data, wait=False)


                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.config.verbose and iter_idx % 10 == 0:
                    print(f"[Realsense] {self.config.serial_number or ''} FPS {frequency}")

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command["cmd"]
                    # execute command if needed.
                    if cmd == Command.START_RECORDING.value:
                        recording_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        serial_number_str = f"{self.config.serial_number}_" if self.config.serial_number else ""
                        output_path = os.path.join(
                            self.config.recording_output_dir,
                            f"color_{serial_number_str}{recording_time_str}.mp4",
                        )
                        config = VideoRecorderConfig(
                            output_path=output_path,
                            width=self.config.aligned_width,
                            height=self.config.aligned_height,
                            fps=self.config.capture_fps,
                            from_fmt="bgr24",
                        )
                        video_recorder = VideoRecorder(config)
                        if self.config.enable_depth_recording:
                            depth_output_path = os.path.join(
                                self.config.recording_output_dir,
                                f"depth_{serial_number_str}{recording_time_str}.mp4",
                            )
                            depth_video_recorder = VideoRecorder(
                                VideoRecorderConfig(
                                    output_path=depth_output_path,
                                    width=self.config.aligned_width,
                                    height=self.config.aligned_height,
                                    fps=self.config.capture_fps,
                                    from_fmt="rgb24",
                                )
                            )
                        if self.config.enable_point_cloud_recording:
                            point_cloud_path = os.path.join(
                                self.config.recording_output_dir,
                                f"point_cloud_{serial_number_str}{recording_time_str}.h5",
                            )
                            point_cloud_h5 = h5py.File(point_cloud_path, "w")
                            point_cloud_dataset = point_cloud_h5.create_dataset(
                                "point_cloud",
                                shape=(0, self.config.num_points_per_frame, 3),
                                maxshape=(None, self.config.num_points_per_frame, 3),
                                dtype=np.float32,
                                chunks=(1, self.config.num_points_per_frame, 3),
                                compression="gzip",
                            )
                        self.start_recording_event.set()
                    elif cmd == Command.STOP_RECORDING.value:
                        if self.config.enable_recording and video_recorder is not None:
                            video_recorder.close()
                        if depth_video_recorder is not None:
                            depth_video_recorder.close()
                        if point_cloud_h5 is not None:
                            point_cloud_h5.close()
                        self.start_recording_event.clear()
                        video_recorder = None
                        depth_video_recorder = None

                iter_idx += 1
        finally:
            if self.config.verbose:
                print(f"[Realsense] {self.config.serial_number or ''} stopping...")
            rs_config.disable_all_streams()
            self.ready_event.set()
            self.stop_event.set()
            if self.config.enable_visualization:
                display_proc.join()
            if self.config.enable_recording and video_recorder is not None:
                video_recorder.close()
            if self.config.enable_depth_recording and depth_video_recorder is not None:
                depth_video_recorder.close()
            if self.config.enable_point_cloud_recording and point_cloud_h5 is not None:
                point_cloud_h5.close()

    def _visualize(self):
        cv2.namedWindow("RealSense Stream", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RealSense Stream", 1080, 720)
        data = None
        while not self.stop_event.is_set():
            try:
                data = self.vis_ring_buffer.get(out=data)
            except Empty:
                continue

            if "color" not in data or "depth" not in data:
                continue

            color = data["color"]
            depth = data["depth"]

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

            if self.config.enable_visualization_color_clipping:
                grey_color = 153
                depth_3d = np.dstack((depth, depth, depth))
                intrinsics = self.intrinsics_array.get()
                depth_scale = intrinsics[-1] if intrinsics[-1] != 0 else 1.0
                clipping_distance = self.config.visualization_depth_clipping_distance_meters / depth_scale
                bg_removed = np.where(
                    (depth_3d > clipping_distance) | (depth_3d <= 0),
                    grey_color,
                    color,
                )
                combined_image = np.hstack((bg_removed, depth_colormap))
            else:
                combined_image = np.hstack((color, depth_colormap))

            if self.config.enable_recording and self.start_recording_event.is_set():
                border_thickness = 15  # adjust the thickness as desired
                cv2.rectangle(
                    combined_image,
                    (0, 0),
                    (combined_image.shape[1] - 1, combined_image.shape[0] - 1),
                    (0, 0, 255),
                    thickness=border_thickness,
                )
            cv2.imshow("RealSense Stream", combined_image)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                self.stop_event.set()
                break
            elif key & 0xFF == ord("s"):
                print("Start recording triggered via keyboard.")
                self.start_recording()
            elif key & 0xFF == ord("q"):
                print("Stop recording triggered via keyboard.")
                self.stop_recording()

            time.sleep(0.01)  # avoid 100% CPU usage

        cv2.destroyAllWindows()

    # ========= function to be overridden ===========
    def process_point_cloud(self, pixels: np.ndarray, depth: np.ndarray, intr_mat: np.ndarray):
        # NOTE this is an example function to process point cloud,
        # you can override this function to process point cloud in a different way
        mask = np.logical_and(depth > self.config.depth_zrange[0], depth < self.config.depth_zrange[1])
        rand_indices = np.random.randint(0, np.sum(mask), size=self.config.num_points_per_frame)
        selected_pixels = pixels[mask][rand_indices]
        selected_depth = depth[mask][rand_indices]
        points = unproject_points(selected_pixels, selected_depth, intr_mat).astype(np.float32)
        return points
