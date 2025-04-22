import numpy as np 

import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    raise RuntimeError("No RealSense devices connected.")

sensor = devices[0].query_sensors()
color_sensor = None
depth_sensor = None
for s in sensor:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        color_sensor = s
        break

for s in sensor:
    if s.get_info(rs.camera_info.name) == 'Depth Camera':
        depth_sensor = s
        break

if color_sensor:
    print("Supported color stream modes:")
    for profile in color_sensor.get_stream_profiles():
        if profile.stream_type() == rs.stream.color:
            vprofile = profile.as_video_stream_profile()
            print("color stream profile:")
            print(f"{vprofile.width()}x{vprofile.height()} @ {vprofile.fps()} FPS")
    
    for profile in depth_sensor.get_stream_profiles():
        if profile.stream_type() == rs.stream.depth:
            vprofile = profile.as_video_stream_profile()
            print("depth stream profile:")
            print(f"{vprofile.width()}x{vprofile.height()} @ {vprofile.fps()} FPS")
else:
    print("No RGB camera found on device.")
