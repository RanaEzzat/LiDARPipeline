#!/usr/bin/env python

import glob
import os
import sys
import numpy as np
import carla
import argparse
import matplotlib.pyplot as plt
from queue import Queue, Empty

# Callback function for Lidar sensor
def sensor_callback(data, queue):
    """ Store sensor data in a thread-safe queue. """
    queue.put(data)

# Function to save Lidar point cloud data
def save_lidar_data(lidar_data, frame):
    """ Save Lidar data as a .npy file. """
    points_dir = "_out/points"
    os.makedirs(points_dir, exist_ok=True)

    p_cloud_size = len(lidar_data)
    p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))  # (x, y, z, intensity)

    file_path = os.path.join(points_dir, f"lidar_{frame:08d}.npy")
    np.save(file_path, p_cloud)
    print(f"\rSaved Lidar point cloud: {file_path}", end="")

# Function to visualize and save Lidar point cloud
def save_lidar_visualization(lidar_data, frame):
    """ Generate and save a 2D visualization of the Lidar point cloud. """
    image_dir = "_out/images"
    os.makedirs(image_dir, exist_ok=True)

    p_cloud_size = len(lidar_data)
    p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))  # (x, y, z, intensity)

    # Extract X, Y points (top-down view)
    x_points = p_cloud[:, 0]
    y_points = p_cloud[:, 1]

    # Plot Lidar points
    plt.figure(figsize=(10, 10))
    plt.scatter(x_points, y_points, s=1, c='red', alpha=0.7)
    plt.xlim(-50, 50)  # Adjust based on Lidar range
    plt.ylim(-50, 50)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title(f"Lidar Point Cloud Frame {frame}")
    plt.grid(True)

    # Save the visualization
    image_path = os.path.join(image_dir, f"lidar_{frame:08d}.png")
    plt.savefig(image_path)
    plt.close()
    print(f"\rSaved Lidar visualization: {image_path}", end="")

# Main function to run CARLA with Lidar only
def run_lidar_only(args):
    """ Run the CARLA simulation with only Lidar. """
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    # Enable synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # Adjust simulation speed
    world.apply_settings(settings)

    vehicle = None
    lidar = None

    try:
        # Ensure output directories exist
        os.makedirs("_out", exist_ok=True)

        # Spawn the vehicle
        vehicle_bp = bp_lib.filter("vehicle.lincoln.mkz_2017")[0]
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[0])
        vehicle.set_autopilot(True)

        # Configure the Lidar sensor
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]
        lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(args.lower_fov))
        lidar_bp.set_attribute('channels', str(args.channels))
        lidar_bp.set_attribute('range', str(args.range))
        lidar_bp.set_attribute('points_per_second', str(args.points_per_second))

        # Spawn the Lidar sensor
        lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=1.0, z=1.8)),
            attach_to=vehicle)

        # Lidar data queue
        lidar_queue = Queue()
        lidar.listen(lambda data: sensor_callback(data, lidar_queue))

        # Run simulation
        for frame in range(args.frames):
            world.tick()
            world_frame = world.get_snapshot().frame

            try:
                lidar_data = lidar_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            assert lidar_data.frame == world_frame
            save_lidar_data(lidar_data, frame)  # Save Lidar point cloud
            save_lidar_visualization(lidar_data, frame)  # Save Lidar visualization

    finally:
        # Restore settings and clean up
        world.apply_settings(original_settings)
        if lidar:
            lidar.destroy()
        if vehicle:
            vehicle.destroy()

# Argument parser
def main():
    """ Start the Lidar-only script. """
    argparser = argparse.ArgumentParser(description='CARLA Lidar-only recording')
    argparser.add_argument('--host', default='127.0.0.1', help='IP of host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port (default: 2000)')
    argparser.add_argument('-f', '--frames', default=300, type=int, help='Number of frames to record')
    argparser.add_argument('--upper-fov', default=30.0, type=float, help='Lidar upper FOV (default: 30.0)')
    argparser.add_argument('--lower-fov', default=-25.0, type=float, help='Lidar lower FOV (default: -25.0)')
    argparser.add_argument('-c', '--channels', default=64, type=int, help='Lidar channel count (default: 64)')
    argparser.add_argument('-r', '--range', default=100.0, type=float, help='Lidar range in meters (default: 100.0)')
    argparser.add_argument('--points-per-second', default=100000, type=int, help='Lidar points per second')

    args = argparser.parse_args()

    try:
        run_lidar_only(args)
    except KeyboardInterrupt:
        print("\nCancelled by user. Exiting.")

if __name__ == '__main__':
    main()
