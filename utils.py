import os
import subprocess
import asyncio

def facefusion_swap(source_image_path: str, target_image_path: str) -> str:
    """
    Calls FaceFusion face swap using your detailed CLI wrapper logic.
    Ensures all paths exist and are absolute, manages output folder.
    """

    # Resolve absolute paths
    abs_source_path = os.path.abspath(source_image_path)
    abs_target_path = os.path.abspath(target_image_path)
    output_dir = os.path.abspath("output")
    os.makedirs(output_dir, exist_ok=True)
    abs_output_path = os.path.join(output_dir, "faceswapped.jpg")

    # Validate input files
    if not os.path.exists(abs_source_path):
        raise FileNotFoundError(f"Source image not found: {abs_source_path}")
    if not os.path.exists(abs_target_path):
        raise FileNotFoundError(f"Target image not found: {abs_target_path}")

    # Possible locations of facefusion.py, adjust as needed
    facefusion_paths = [
        "../fake_news_service",
        "./fake_news_service",
        ".",
        "/home/merlotllm/Documents/fake_news_service"
    ]
    facefusion_dir = None
    for path in facefusion_paths:
        full_path = os.path.abspath(path)
        if os.path.exists(os.path.join(full_path, "facefusion.py")):
            facefusion_dir = full_path
            break
    if not facefusion_dir:
        raise RuntimeError("Could not find facefusion.py in any expected location")

    # Build FaceFusion command
    cmd = [
        "python", "facefusion.py",
        "headless-run",
        "--source-paths", abs_source_path,
        "--target-path", abs_target_path,
        "--output-path", abs_output_path,
        "--processors", "face_swapper",
        "--face-swapper-model", "inswapper_128_fp16",
        "--face-swapper-pixel-boost", "256x256",
        "--output-image-quality", "80",
        "--execution-providers", "cuda",  # or "cpu" if no GPU
    ]

    # Run command synchronously with timeout
    try:
        result = subprocess.run(
            cmd,
            cwd=facefusion_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("FaceFusion process timed out")

    # Debug output (optional)
    print(f"FaceFusion stdout:\n{result.stdout}")
    print(f"FaceFusion stderr:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"FaceFusion failed with exit code {result.returncode}")

    if not os.path.exists(abs_output_path):
        raise RuntimeError("FaceFusion output file not created")

    return abs_output_path
