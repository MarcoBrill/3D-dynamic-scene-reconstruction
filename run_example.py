from main import main

if __name__ == "__main__":
    # Example inputs
    video_paths = ["data/video_1.mp4", "data/video_2.mp4"]
    camera_params = {
        "intrinsic": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],  # Example intrinsic matrix
        "extrinsic": [np.eye(4) for _ in range(2)]  # Example extrinsic matrices
    }

    # Run the pipeline
    main(video_paths, camera_params)
