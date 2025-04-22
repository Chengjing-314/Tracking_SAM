import pyrealsense2 as rs
import cv2
import time
import numpy as np
import tracking_SAM

def main(sam_ckpt, aot_ckpt, dino_ckpt, play_delay=1):
    # 1) Initialize your tracker
    tracker = tracking_SAM.main_tracker(sam_ckpt, aot_ckpt, dino_ckpt)

    # 2) Setup RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    pipeline.start(config)
    # warm‑up
    for _ in range(30):
        pipeline.wait_for_frames()

    cv2.namedWindow('Live', cv2.WINDOW_AUTOSIZE)
    print("Press 'a' to annotate the first frame, 'q' to quit.")

    # 3) Wait for initial annotation
    while True:
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        bgr = np.asanyarray(color.get_data())
        cv2.imshow('Live', bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tracker.annotate_init_frame(rgb)
            break
        elif key == ord('q'):
            pipeline.stop()
            return

    while True:
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        bgr = np.asanyarray(color.get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Show raw frame
        cv2.imshow('Live', bgr)

        # If already tracking, propagate and overlay
        if tracker.is_tracking():
            cv2.destroyWindow('Live')
            start = time.time()
            raw_mask = tracker.propagate_one_frame(rgb)
            elapsed = time.time() - start

            # Threshold → boolean → uint8 → scale to 255
            mask = (raw_mask > 0).astype(np.uint8) * 255

            # Build a 3‑channel overlay of the same shape & dtype as bgr
            overlay = np.zeros_like(bgr, dtype=bgr.dtype)
            overlay[..., 2] = mask    # fill red channel

            # Blend (both are uint8)
            vis = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
            cv2.putText(vis, f"Latency {elapsed:.2f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Tracked', vis)

        # Key controls
        k = cv2.waitKey(play_delay) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('a'):
            tracker.reset_engine()
            tracker.annotate_init_frame(rgb)
        elif k == ord('d'):
            tracker.reset_engine()

    
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sam_checkpoint',    default="./pretrained_weights/sam_vit_h_4b8939.pth")
    p.add_argument('--aot_checkpoint',    default="./pretrained_weights/AOTT_PRE_YTB_DAV.pth")
    p.add_argument('--ground_dino_checkpoint',
                   default="./pretrained_weights/groundingdino_swint_ogc.pth")
    p.add_argument('--play_delay', type=int, default=1)
    args = p.parse_args()
    main(args.sam_checkpoint, args.aot_checkpoint,
         args.ground_dino_checkpoint, args.play_delay)
