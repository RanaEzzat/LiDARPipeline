import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

# Paths to original data
BASE_PATH = r"C:\Users\<username>\Downloads\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\_out"
POINTS_PATH = os.path.join(BASE_PATH, "points")  # Original .npy Lidar points
IMAGES_PATH = os.path.join(BASE_PATH, "images")  # Original .png Lidar images

# Paths for attacked outputs
ATTACK_PATHS = {
    "original": os.path.join(BASE_PATH, "original"),
    "spoofing": os.path.join(BASE_PATH, "spoofing"),
    "adversarial_perturbation": os.path.join(BASE_PATH, "adversarial_perturbation"),
    "occlusion_attack": os.path.join(BASE_PATH, "occlusion_attack"),
    "misalignment_attack": os.path.join(BASE_PATH, "misalignment_attack"),
    "scaling_attack": os.path.join(BASE_PATH, "scaling_attack"),
}

# Ensure attack directories exist
for attack in ATTACK_PATHS.values():
    os.makedirs(os.path.join(attack, "points"), exist_ok=True)
    os.makedirs(os.path.join(attack, "images"), exist_ok=True)

# Load all .npy files
lidar_files = sorted(glob.glob(os.path.join(POINTS_PATH, "*.npy")))

# Attack functions
#def lidar_spoofing(lidar_data):
#    """ Inject fake Lidar points. """
#    fake_points = np.array([[10 + x, y, 2, 1] for x in np.linspace(-5, 5, 200) for y in np.linspace(-5, 5, 200)])
#    return np.vstack((lidar_data, fake_points)), fake_points  # Returning fake points separately
def lidar_spoofing(lidar_data):
    """ Inject fake Lidar points to simulate a spoofing attack. """
    # Generate a smaller, less dense grid of fake points
    fake_points = np.array([[10 + x, y, 2, 1] for x in np.linspace(-2, 2, 20) for y in np.linspace(-2, 2, 20)])
    
    # Alternatively, simulate a small cluster of points (e.g., a fake car)
    # fake_points = np.array([[10 + x, y, 2, 1] for x in np.linspace(-1, 1, 10) for y in np.linspace(-1, 1, 10)])
    
    return np.vstack((lidar_data, fake_points)), fake_points  # Returning fake points separately

def adversarial_noise(lidar_data):
    """ Add noise to Lidar points. """
    noise = np.random.normal(0, 0.5, lidar_data[:, :3].shape)  # Increased noise magnitude
    modified_data = lidar_data.copy()
    modified_data[:, :3] += noise  # Apply noise
    return modified_data, modified_data[:, :3]  # Returning modified points separately

def occlusion_attack(lidar_data):
    # More intense occlusion: Increase the threshold
    mask = lidar_data[:, 0] > 10  # Increased threshold
    return lidar_data[mask], None

def misalignment_attack(lidar_data):
    # More intense misalignment: Increase the shift
    modified_data = lidar_data.copy()
    modified_data[:, 0] += 10  # Increased shift
    return modified_data, None

def scaling_attack(lidar_data):
    # More intense scaling: Increase the scaling factor
    modified_data = lidar_data.copy()
    modified_data[:, :3] *= 0.1  # Increased scaling factor
    return modified_data, None

# Visualization function
def save_lidar_visualization(lidar_data, image_path):
    """ Save Lidar 2D visualization with all points in the same color. """
    x_points, y_points = lidar_data[:, 0], lidar_data[:, 1]
    plt.figure(figsize=(10, 10))
    plt.scatter(x_points, y_points, s=1, c='red', alpha=0.7)  # All points are red

    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True)
    plt.savefig(image_path)
    plt.close()

# Process all frames
for lidar_file in lidar_files:
    frame_name = os.path.basename(lidar_file)
    frame_num = frame_name.split('.')[0]

    lidar_data = np.load(lidar_file)

    original_image_path = os.path.join(ATTACK_PATHS["original"], "images", f"{frame_num}.png")
    save_lidar_visualization(lidar_data, original_image_path)

    # Ensure all attack functions return consistent outputs
    attacks = {
        "spoofing": lidar_spoofing(lidar_data.copy()),
        "adversarial_perturbation": adversarial_noise(lidar_data.copy()),
        "occlusion_attack": occlusion_attack(lidar_data.copy()),
        "misalignment_attack": misalignment_attack(lidar_data.copy()),
        "scaling_attack": scaling_attack(lidar_data.copy()),
    }

    for attack_name, attack_result in attacks.items():
        if isinstance(attack_result, tuple):
            attacked_data, _ = attack_result  # Ignore attack_points
        else:
            attacked_data = attack_result

        npy_output_path = os.path.join(ATTACK_PATHS[attack_name], "points", f"{frame_num}.npy")
        png_output_path = os.path.join(ATTACK_PATHS[attack_name], "images", f"{frame_num}.png")

        # Skip if files already exist (to avoid re-processing)
        if os.path.exists(npy_output_path) and os.path.exists(png_output_path):
            print(f"Skipping frame {frame_num} for {attack_name} (already processed)")
            continue

        np.save(npy_output_path, attacked_data)

        # Save visualization with all points in the same color
        save_lidar_visualization(attacked_data, png_output_path)

    print(f"Processed frame {frame_num} with all attacks.")

print("All frames processed and attacks applied!")

# Convert attack images + original images to videos
def create_video(image_folder, output_video):
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not images:
        print(f"No images found in {image_folder}")
        return

    first_frame = cv2.imread(images[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 10, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video saved: {output_video}")

attack_videos = {}
for attack_name, path in ATTACK_PATHS.items():
    video_path = os.path.join(BASE_PATH, f"{attack_name}.mp4")
    create_video(os.path.join(path, "images"), video_path)
    attack_videos[attack_name] = video_path

import numpy as np
import cv2
import os

# Combine attack videos + original video in 3x2 grid with titles below each video
def combine_videos(video_files, output_video):
    caps = [cv2.VideoCapture(video) for video in video_files]

    # Check if all videos were opened successfully
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Could not open video {video_files[i]}")
            return

    # Get the dimensions of the first frame of the first video
    ret, first_frame = caps[0].read()
    if not ret:
        print("Error: Could not read the first frame of the first video.")
        return

    height, width, _ = first_frame.shape

    # Define grid dimensions
    grid_width = width // 2
    grid_height = height // 2
    title_height = 50  # Space for titles below each video
    final_width = grid_width * 3
    final_height = (grid_height + title_height) * 2

    # Define the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 10, (final_width, final_height))

    # Extract titles from video filenames
    titles = [os.path.splitext(os.path.basename(video))[0] for video in video_files]

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                # End of video
                break
            # Resize frame to fit the grid
            frame = cv2.resize(frame, (grid_width, grid_height))
            frames.append(frame)

        # Break if any video has ended
        if len(frames) < len(caps):
            break

        # Create the grid
        grid = np.zeros((final_height, final_width, 3), dtype=np.uint8)

        # Place frames in the grid and add titles BELOW each video
        for i, frame in enumerate(frames):
            row = i // 3  # 0 for top row, 1 for bottom row
            col = i % 3   # 0 for left, 1 for middle, 2 for right

            x_offset = col * grid_width
            y_offset = row * (grid_height + title_height)

            # Place the frame in the grid
            grid[y_offset:y_offset + grid_height, x_offset:x_offset + grid_width] = frame

            # Add black background for the title
            grid[y_offset + grid_height:y_offset + grid_height + title_height, x_offset:x_offset + grid_width] = (0, 0, 0)

            # Add title BELOW the frame (Extracted from filename)
            title = titles[i]
            text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]  # Smaller font size
            text_x = x_offset + (grid_width - text_size[0]) // 2  # Center horizontally
            text_y = y_offset + grid_height + (title_height // 2) + (text_size[1] // 2)  # Adjust position below video
            cv2.putText(grid, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write the grid frame to the output video
        out.write(grid)

    # Release all video captures and the output video writer
    for cap in caps:
        cap.release()
    out.release()

    print(f"Combined attack video saved: {output_video}")

combine_videos(list(attack_videos.values()), os.path.join(BASE_PATH, "combined_attacks.mp4"))
