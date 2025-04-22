

from multiprocessing.managers import SharedMemoryManager
import time
import cv2
import numpy as np
from cameras.realsense import Realsense, RealsenseConfig, unproject_points
import tracking_SAM
from fastdev.sim_webui import SimWebUI
import open3d as o3d


def filter_and_cluster_pointcloud(points, voxel_size=0.005, eps=0.01, min_points=10, min_cluster_size=10):
    # Build Open3D pointcloud and downsample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Cluster with DBSCAN
    labels = np.array(pcd_down.cluster_dbscan(eps=eps, min_points=min_points))

    # 1) mask out noise
    good_mask = labels != -1

    # 2) compute cluster sizes
    unique_labels, counts = np.unique(labels[good_mask], return_counts=True)
    # keep only clusters with enough points
    large_clusters = unique_labels[counts >= min_cluster_size]

    # final mask: point is non‑noise AND belongs to a large cluster
    keep_mask = np.isin(labels, large_clusters)

    # extract the surviving points
    indices = np.nonzero(keep_mask)[0].tolist()
    pcd_clean = pcd_down.select_by_index(indices)

    pcd_clean = np.asarray(pcd_clean.points)

    return pcd_clean, labels

def main():
    # 1) Launch your Realsense worker
    with SharedMemoryManager() as smm:
       
        rs_cfg = RealsenseConfig(enable_recording=False, enable_depth_recording=False,
                                 enable_point_cloud_recording=False,enable_visualization=False)

        rs = Realsense(smm, config=rs_cfg)
        rs.start()          # starts the processor
        rs.start_wait()     # wait until first frame is in


        tracker = tracking_SAM.main_tracker(
            "./pretrained_weights/sam_vit_h_4b8939.pth",
            "./pretrained_weights/AOTT_PRE_YTB_DAV.pth",
            "./pretrained_weights/groundingdino_swint_ogc.pth",
        )

        sample = rs.get()  
        color = sample["color"]
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        tracker.annotate_init_frame(rgb)

        

        H, W = sample["color"].shape[:2]
        __import__('IPython').embed(header='real_sense_module.py:35')
        pixels = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(np.float32)
        intr = np.eye(3)
        intr[0,0], intr[1,1] = rs.intrinsics_array.get()[0], rs.intrinsics_array.get()[1]
        intr[0,2], intr[1,2] = rs.intrinsics_array.get()[2], rs.intrinsics_array.get()[3]
        depth_scale = rs.intrinsics_array.get()[-1]

        # # 3) Main loop: fetch → track → unproject mask → visualize
        # cv2.namedWindow("SegPointCloud", cv2.WINDOW_NORMAL)

        webui = SimWebUI()


        while True:
            frame = rs.get()  # latest frame
            color = frame["color"]
            depth_raw = frame["depth"].astype(np.float32) * depth_scale

            # 3a) 2D tracking
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            mask2d = (tracker.propagate_one_frame(rgb) > 0)

            uv = pixels[mask2d]   
            d  = depth_raw[mask2d]
            pts3d = unproject_points(uv, d, intr)  

            cleaned_pcd, labels = filter_and_cluster_pointcloud(pts3d, voxel_size=0.005, eps=0.01, min_points=10, min_cluster_size=50)

            temp_mask =cleaned_pcd[:, 2] < 0.75
            cleaned_pcd = cleaned_pcd[temp_mask]

            pt_id = webui.add_point_cloud_asset(cleaned_pcd)
            webui.set_point_cloud_state(pt_id, point_size=0.002)
            __import__('IPython').embed(header='real_sense_module.py:66')
            np.savez("cleaned_pcd_1.npy", cleaned_pcd)



            # exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # # cleanup
        # rs.stop()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
