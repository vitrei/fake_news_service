import os
import subprocess
import asyncio
import uuid
from pathlib import Path

async def facefusion_swap(source_image_path: str, target_image_path: str, quality: int = 80, user_id: str = None) -> str:
    print(f"DEBUG: user_id value: '{user_id}', type: {type(user_id)}, bool: {bool(user_id)}")

    try:
        # print(f"DEBUG: Starting face swap process")
        # print(f"DEBUG: Source: {source_image_path}")
        # print(f"DEBUG: Target: {target_image_path}")
        # print(f"DEBUG: Quality: {quality}")
        print(f"DEBUG: User ID: {user_id}")
        
        # Resolve absolute paths
        abs_source_path = os.path.abspath(source_image_path)
        abs_target_path = os.path.abspath(target_image_path)
        
        # Create output directory
        output_dir = os.path.abspath("output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename based on user_id or unique ID
        if user_id:
            filename = f"{user_id}.jpg"
        else:
            request_id = str(uuid.uuid4())
            filename = f"swapped_{request_id}.jpg"
        
        abs_output_path = os.path.join(output_dir, filename)
        
        # print(f"DEBUG: Absolute source path: {abs_source_path}")
        # print(f"DEBUG: Absolute target path: {abs_target_path}")
        # print(f"DEBUG: Absolute output path: {abs_output_path}")
        
        # Validate input files
        if not os.path.exists(abs_source_path):
            raise FileNotFoundError(f"Source image not found: {abs_source_path}")
        if not os.path.exists(abs_target_path):
            raise FileNotFoundError(f"Target image not found: {abs_target_path}")
            
        # Check file sizes
        source_size = os.path.getsize(abs_source_path)
        target_size = os.path.getsize(abs_target_path)
        print(f"DEBUG: Source file size: {source_size} bytes")
        print(f"DEBUG: Target file size: {target_size} bytes")
        
        if source_size == 0 or target_size == 0:
            raise RuntimeError("One or both input files are empty")
        
        # Possible locations of facefusion.py
        facefusion_paths = [
            "../facefusion",
            "./facefusion",
            ".",
            "/home/merlotllm/Documents/facefusion",
            "../fake_news_service",
            "./fake_news_service",
            "/home/merlotllm/Documents/fake_news_service"
        ]
        
        facefusion_dir = None
        for path in facefusion_paths:
            full_path = os.path.abspath(path)
            facefusion_file = os.path.join(full_path, "facefusion.py")
            print(f"DEBUG: Checking for facefusion.py at: {facefusion_file}")
            if os.path.exists(facefusion_file):
                facefusion_dir = full_path
                break
        
        if not facefusion_dir:
            print("ERROR: Could not find facefusion.py in any expected location")
            print("DEBUG: Searched paths:")
            for path in facefusion_paths:
                print(f"  - {os.path.abspath(path)}")
            raise RuntimeError("Could not find facefusion.py in any expected location")
        
        print(f"DEBUG: Using FaceFusion directory: {facefusion_dir}")
        
        # Detect available execution providers
        # available_providers = await detect_execution_providers(facefusion_dir)
        # execution_provider = "cuda" if "cuda" in available_providers else "cpu"
        
        
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
            "--output-image-quality", str(quality),
            "--execution-providers", "cuda",
        ]
        
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        print(f"DEBUG: Working directory: {facefusion_dir}")
        
        # Run command asynchronously with timeout
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                cwd=facefusion_dir,
                timeout=300  # 5 minute timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError("FaceFusion process timed out")
        
        print(f"DEBUG: Command completed with return code: {result.returncode}")
        
        # Debug output
        if result.stdout:
            print(f"DEBUG: STDOUT: {result.stdout}")
        if result.stderr:
            print(f"DEBUG: STDERR: {result.stderr}")
        
        if result.returncode != 0:
            raise RuntimeError(f"FaceFusion failed with exit code {result.returncode}: {result.stderr}")
        
        if not os.path.exists(abs_output_path):
            raise RuntimeError("FaceFusion output file not created")
            
        # Verify output file has content
        output_size = os.path.getsize(abs_output_path)
        print(f"DEBUG: Face swap completed successfully, output size: {output_size} bytes")
        
        if output_size == 0:
            raise RuntimeError("FaceFusion created empty output file")
        
        return abs_output_path
        
    except Exception as e:
        print(f"ERROR in facefusion_swap: {str(e)}")
        print(f"ERROR type: {type(e).__name__}")
        import traceback
        print(f"ERROR traceback: {traceback.format_exc()}")
        raise

async def facefusion_video_swap(source_image_path: str, target_video_path: str, quality: int = 80, user_id: str = None) -> str:
    print(f"DEBUG: user_id value: '{user_id}', type: {type(user_id)}, bool: {bool(user_id)}")
    
    try:
        # print(f"DEBUG: Starting video face swap process")
        # print(f"DEBUG: Source: {source_image_path}")
        # print(f"DEBUG: Target: {target_video_path}")
        # print(f"DEBUG: Quality: {quality}")
        # print(f"DEBUG: User ID: {user_id}")
        
        # Resolve absolute paths
        abs_source_path = os.path.abspath(source_image_path)
        abs_target_path = os.path.abspath(target_video_path)
        
        # Create output directory
        output_dir = os.path.abspath("output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename based on user_id or unique ID
        if user_id:
            filename = f"{user_id}_video.mp4"
        else:
            request_id = str(uuid.uuid4())
            filename = f"swapped_video_{request_id}.mp4"
        
        abs_output_path = os.path.join(output_dir, filename)
        
        # print(f"DEBUG: Absolute source path: {abs_source_path}")
        # print(f"DEBUG: Absolute target path: {abs_target_path}")
        # print(f"DEBUG: Absolute output path: {abs_output_path}")
        
        # Validate input files
        if not os.path.exists(abs_source_path):
            raise FileNotFoundError(f"Source image not found: {abs_source_path}")
        if not os.path.exists(abs_target_path):
            raise FileNotFoundError(f"Target video not found: {abs_target_path}")
            
        # Check file sizes
        source_size = os.path.getsize(abs_source_path)
        target_size = os.path.getsize(abs_target_path)
        print(f"DEBUG: Source file size: {source_size} bytes")
        print(f"DEBUG: Target file size: {target_size} bytes")
        
        if source_size == 0 or target_size == 0:
            raise RuntimeError("One or both input files are empty")
        
        # Find facefusion directory (same logic as your image swap)
        facefusion_paths = [
            "../facefusion",
            "./facefusion",
            ".",
            "/home/merlotllm/Documents/facefusion",
            "../fake_news_service",
            "./fake_news_service",
            "/home/merlotllm/Documents/fake_news_service"
        ]
        
        facefusion_dir = None
        for path in facefusion_paths:
            full_path = os.path.abspath(path)
            facefusion_file = os.path.join(full_path, "facefusion.py")
            print(f"DEBUG: Checking for facefusion.py at: {facefusion_file}")
            if os.path.exists(facefusion_file):
                facefusion_dir = full_path
                break
        
        if not facefusion_dir:
            print("ERROR: Could not find facefusion.py in any expected location")
            print("DEBUG: Searched paths:")
            for path in facefusion_paths:
                print(f"  - {os.path.abspath(path)}")
            raise RuntimeError("Could not find facefusion.py in any expected location")
        
        print(f"DEBUG: Using FaceFusion directory: {facefusion_dir}")
        
        # Build FaceFusion command for video
        cmd = [
            "python", "facefusion.py",
            "headless-run",
            "--source-paths", abs_source_path,
            "--target-path", abs_target_path,
            "--output-path", abs_output_path,
            "--processors", "face_swapper",
            "--face-swapper-model", "inswapper_128_fp16",
            "--face-swapper-pixel-boost", "128x128",  # Lower for video performance
            "--output-video-quality", str(quality),
            "--output-video-encoder", "libx264",
            "--execution-providers", "cuda",
        ]
        
        print(f"DEBUG: Running video command: {' '.join(cmd)}")
        print(f"DEBUG: Working directory: {facefusion_dir}")
        
        # Run command asynchronously with longer timeout for video
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                cwd=facefusion_dir,
                timeout=600  # 10 minute timeout for video
            )
        except asyncio.TimeoutError:
            raise RuntimeError("FaceFusion video process timed out")
        
        print(f"DEBUG: Command completed with return code: {result.returncode}")
        
        # Debug output
        if result.stdout:
            print(f"DEBUG: STDOUT: {result.stdout}")
        if result.stderr:
            print(f"DEBUG: STDERR: {result.stderr}")
        
        if result.returncode != 0:
            raise RuntimeError(f"FaceFusion video failed with exit code {result.returncode}: {result.stderr}")
        
        if not os.path.exists(abs_output_path):
            raise RuntimeError("FaceFusion video output file not created")
            
        # Verify output file has content
        output_size = os.path.getsize(abs_output_path)
        print(f"DEBUG: Video face swap completed successfully, output size: {output_size} bytes")
        
        if output_size == 0:
            raise RuntimeError("FaceFusion created empty video output file")
        
        return abs_output_path
        
    except Exception as e:
        print(f"ERROR in facefusion_video_swap: {str(e)}")
        print(f"ERROR type: {type(e).__name__}")
        import traceback
        print(f"ERROR traceback: {traceback.format_exc()}")
        raise
