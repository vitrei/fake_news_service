from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uuid
from pathlib import Path
import asyncio
from typing import Optional
import shutil
from user_client import get_user_profile
from utils import swap_faces_with_user_logic, swap_faces_logic


import sys
import subprocess


sys.path.append('./facefusion') 

try:
    from facefusion.processors.modules.face_swapper import process_frame, get_reference_faces
    from facefusion import core
except ImportError as e:
    print(f"Error importing FaceFusion modules: {e}")
    print("Make sure FaceFusion is properly installed and paths are correct")

app = FastAPI(title="FaceFusion API", description="Face swapping API using FaceFusion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TEMP_DIR = Path("./temp")
OUTPUT_DIR = Path("./output")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.post("/swap-faces")
async def swap_faces(
    source_image: UploadFile = File(..., description="Source face image"),
    target_image: UploadFile = File(..., description="Target image"),
    output_quality: Optional[int] = 80,
    user_id: Optional[str] = None  # Add user_id parameter if needed
):
    """
    Swap faces between source and target images
    """
    request_id = str(uuid.uuid4())
    
    # Initialize paths to None for proper cleanup
    source_path = None
    target_path = None
    output_path = None
   
    try:
        # Get user profile if user_id is provided
        user_profile = None
        if user_id:
            user_profile = get_user_profile(user_id)
            if not user_profile:
                print(f"WARNING: Could not fetch user profile for user_id: {user_id}")
        
        # Define file paths
        source_path = TEMP_DIR / f"{request_id}_source.jpg"
        target_path = TEMP_DIR / f"{request_id}_target.jpg"
        output_path = OUTPUT_DIR / f"swapped_{request_id}.jpg"  
        
        print(f"DEBUG: Source path: {source_path}")
        print(f"DEBUG: Target path: {target_path}")
        print(f"DEBUG: Output path: {output_path}")
       
        # Save uploaded files
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_image.file, f)
        print(f"DEBUG: Saved source image to {source_path}")
       
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_image.file, f)
        print(f"DEBUG: Saved target image to {target_path}")
        
        # Verify files were saved correctly
        if not os.path.exists(source_path):
            raise HTTPException(status_code=500, detail="Failed to save source image")
        if not os.path.exists(target_path):
            raise HTTPException(status_code=500, detail="Failed to save target image")
            
        print(f"DEBUG: Source file size: {os.path.getsize(source_path)} bytes")
        print(f"DEBUG: Target file size: {os.path.getsize(target_path)} bytes")
        
        # Process face swap
        success = await process_face_swap(
            str(source_path),
            str(target_path),
            str(output_path),
            quality=output_quality
        )
       
        if not success:
            raise HTTPException(status_code=500, detail="Face swapping failed during processing")
            
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Face swapping failed - output file not created")
       
        # Clean up temporary files
        cleanup_temp_files([source_path, target_path])
        
        response_data = {
            "success": True,
            "message": "Face swapping completed successfully!",
            "output_file": str(output_path),
            "filename": f"swapped_{request_id}.jpg",
            "request_id": request_id
        }
        
        # Add user info if available
        if user_profile:
            response_data["user_id"] = user_id
       
        return response_data
       
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        cleanup_temp_files([source_path, target_path])
        raise
    except Exception as e:
        print(f"ERROR in swap_faces: {str(e)}")
        print(f"ERROR type: {type(e).__name__}")
        cleanup_temp_files([source_path, target_path])
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


async def process_face_swap(source_path: str, target_path: str, output_path: str, quality: int = 80):
    """
    Wrapper function for FaceFusion face swapping using CLI
    """
    try:
        print(f"DEBUG: Starting face swap process")
        print(f"DEBUG: Source: {source_path}")
        print(f"DEBUG: Target: {target_path}")
        print(f"DEBUG: Output: {output_path}")
        print(f"DEBUG: Quality: {quality}")
       
        # Verify input files exist and are readable
        if not os.path.exists(source_path):
            print(f"ERROR: Source file does not exist: {source_path}")
            return False
        if not os.path.exists(target_path):
            print(f"ERROR: Target file does not exist: {target_path}")
            return False
            
        # Check file sizes
        source_size = os.path.getsize(source_path)
        target_size = os.path.getsize(target_path)
        print(f"DEBUG: Source file size: {source_size} bytes")
        print(f"DEBUG: Target file size: {target_size} bytes")
        
        if source_size == 0 or target_size == 0:
            print("ERROR: One or both input files are empty")
            return False
       
        facefusion_paths = [
            "../facefusion",  
            "./facefusion",  
            ".",              
            "/home/merlotllm/Documents/facefusion"
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
            return False
       
        print(f"DEBUG: Using FaceFusion directory: {facefusion_dir}")
        
        # Convert paths to absolute paths for better compatibility
        abs_source_path = os.path.abspath(source_path)
        abs_target_path = os.path.abspath(target_path)
        abs_output_path = os.path.abspath(output_path)
        
        print(f"DEBUG: Absolute source path: {abs_source_path}")
        print(f"DEBUG: Absolute target path: {abs_target_path}")
        print(f"DEBUG: Absolute output path: {abs_output_path}")
       
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
            "--execution-providers", "cuda",  # Change to "cpu" if no GPU
        ]
       
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        print(f"DEBUG: Working directory: {facefusion_dir}")
       
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(abs_output_path)
        os.makedirs(output_dir, exist_ok=True)
        print(f"DEBUG: Output directory: {output_dir}")
       
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            cwd=facefusion_dir,
            timeout=300  # 5 minute timeout
        )
       
        print(f"DEBUG: Command completed with return code: {result.returncode}")
        
        if result.stdout:
            print(f"DEBUG: STDOUT: {result.stdout}")
        if result.stderr:
            print(f"DEBUG: STDERR: {result.stderr}")
       
        if result.returncode == 0:
            # Verify output file was created
            if os.path.exists(abs_output_path):
                output_size = os.path.getsize(abs_output_path)
                print(f"DEBUG: Face swap completed successfully, output size: {output_size} bytes")
                return output_size > 0  # Return True only if file has content
            else:
                print("ERROR: Face swap reported success but output file was not created")
                return False
        else:
            print(f"ERROR: Face swap failed with return code: {result.returncode}")
            return False
           
    except asyncio.TimeoutError:
        print("ERROR: Face swap process timed out")
        return False
    except Exception as e:
        print(f"ERROR in process_face_swap: {str(e)}")
        print(f"ERROR type: {type(e).__name__}")
        import traceback
        print(f"ERROR traceback: {traceback.format_exc()}")
        return False

# @app.post("/swap-faces")
# async def swap_faces(
#     source_image: UploadFile = File(..., description="Source face image"),
#     target_image: UploadFile = File(..., description="Target image"),
#     output_quality: Optional[int] = 80
# ):
#     """
#     Swap faces between source and target images
#     """
#     request_id = str(uuid.uuid4())
    
#     try:
#         source_path = TEMP_DIR / f"{request_id}_source.jpg"
#         target_path = TEMP_DIR / f"{request_id}_target.jpg"
#         output_path = OUTPUT_DIR / f"swapped_{request_id}.jpg"  
        
#         with open(source_path, "wb") as f:
#             shutil.copyfileobj(source_image.file, f)
        
#         with open(target_path, "wb") as f:
#             shutil.copyfileobj(target_image.file, f)

#         success = await process_face_swap(
#             str(source_path), 
#             str(target_path), 
#             str(output_path),
#             quality=output_quality
#         )
        
#         if not success or not os.path.exists(output_path):
#             raise HTTPException(status_code=500, detail="Face swapping failed - output file not created")
        
#         cleanup_temp_files([source_path, target_path])
        
#         return {
#             "success": True,
#             "message": "Face swapping completed successfully!",
#             "output_file": str(output_path),
#             "filename": f"swapped_{request_id}.jpg"
#         }
        
#     except Exception as e:
#         cleanup_temp_files([source_path, target_path])
#         raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    

@app.post("/swap-faces-with-user/{user_id}")
async def swap_faces_with_user(
    user_id: str,
    target_image: UploadFile = File(..., description="Target image"),
    output_quality: Optional[int] = 80
):
    """
    1. Get user data from user_profile_service (port 8010) and download their image
    2. Use that data to call /swap-faces endpoint
    """
    return await swap_faces_with_user_logic(user_id, target_image, None, output_quality)

# @app.post("/swap-faces-video")
# async def swap_faces_video(
#     source_image: UploadFile = File(..., description="Source face image"),
#     target_video: UploadFile = File(..., description="Target video"),
#     output_quality: Optional[int] = 80,
#     template: Optional[str] = None
# ):
#     """
#     Swap face in a video
#     """
#     request_id = str(uuid.uuid4())
    
#     try:
#         source_path = TEMP_DIR / f"{request_id}_source.jpg"
#         target_path = TEMP_DIR / f"{request_id}_target.mp4"
#         output_path = OUTPUT_DIR / f"swapped_video_{request_id}.mp4"
        
#         with open(source_path, "wb") as f:
#             shutil.copyfileobj(source_image.file, f)
        
#         with open(target_path, "wb") as f:
#             shutil.copyfileobj(target_video.file, f)
        
#         success = await process_video_face_swap(
#             str(source_path),
#             str(target_path),
#             str(output_path),
#             quality=output_quality
#         )
        
#         if not success or not os.path.exists(output_path):
#             raise HTTPException(status_code=500, detail="Video face swapping failed - output file not created")
        
#         return FileResponse(
#             output_path,
#             media_type="video/mp4",
#             filename=f"swapped_video_{request_id}.mp4",
#             background=cleanup_files([source_path, target_path, output_path])
#         )
        
#     except Exception as e:
#         cleanup_temp_files([source_path, target_path])
#         raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

@app.post("/swap-faces-video")
async def swap_faces_video(
    source_image: UploadFile = File(..., description="Source face image"),
    target_video: UploadFile = File(..., description="Target video"),
    output_quality: Optional[int] = 80
):
    """Swap face in a video"""
    request_id = str(uuid.uuid4())
    
    try:
        source_path = TEMP_DIR / f"{request_id}_source.jpg"
        target_path = TEMP_DIR / f"{request_id}_target.mp4"
        output_path = OUTPUT_DIR / f"swapped_video_{request_id}.mp4"
        
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_image.file, f)
        
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_video.file, f)
        
        success = await process_video_face_swap(
            str(source_path),
            str(target_path),
            str(output_path),
            quality=output_quality
        )
        
        if not success or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Video face swapping failed")
        
        cleanup_temp_files([source_path, target_path])
        
        return {"success": True, "video_filename": f"swapped_video_{request_id}.mp4"}
        
    except Exception as e:
        cleanup_temp_files([source_path, target_path])
        raise HTTPException(status_code=500, detail=str(e))

# async def process_face_swap(source_path: str, target_path: str, output_path: str, quality: int = 80):
#     """
#     Wrapper function for FaceFusion face swapping using CLI
#     """
#     try:
#         print(f"Processing face swap: {source_path} -> {target_path} -> {output_path}")
        
#         facefusion_paths = [
#             "../facefusion",  
#             "./facefusion",   
#             ".",              
#             "/home/merlotllm/Documents/facefusion" 
#         ]
        
#         facefusion_dir = None
#         for path in facefusion_paths:
#             if os.path.exists(os.path.join(path, "facefusion.py")):
#                 facefusion_dir = path
#                 break
        
#         if not facefusion_dir:
#             print("Could not find facefusion.py in any expected location")
#             return False
        
#         print(f"Using FaceFusion directory: {facefusion_dir}")
        
#         # Build FaceFusion commands
#         cmd = [
#             "python", "facefusion.py", 
#             "headless-run",  # Run in headless mode (no GUI)
#             "--source-paths", source_path,  # Source face image
#             "--target-path", target_path,   # Target image to swap face into
#             "--output-path", output_path,   # Output file path
#             "--processors", "face_swapper", # Enable face swapper processor
#             "--face-swapper-model", "inswapper_128_fp16",  # Default model (fast and good quality)
#             "--face-swapper-pixel-boost", "256x256",  # Good balance of quality and speed
#             "--output-image-quality", str(quality),  # Image quality setting
#             "--execution-providers", "cuda",  # Use CPU (change to "cuda" if you have GPU)
#         ]
        
#         print(f"Running command: {' '.join(cmd)}")
#         print(f"Working directory: {facefusion_dir}")
        
#         result = await asyncio.to_thread(
#             subprocess.run, 
#             cmd, 
#             capture_output=True, 
#             text=True,
#             cwd=facefusion_dir 
#         )
        
#         if result.returncode == 0:
#             print(f"Face swap completed successfully")
#             print(f"STDOUT: {result.stdout}")
#             return True
#         else:
#             print(f"Face swap failed with return code: {result.returncode}")
#             print(f"STDERR: {result.stderr}")
#             print(f"STDOUT: {result.stdout}")
#             return False
            
#     except Exception as e:
#         print(f"Face swap error: {e}")
#         return False

async def process_video_face_swap(source_path: str, target_path: str, output_path: str, quality: int = 80):
    """
    Process video face swapping using FaceFusion CLI
    """
    try:
        print(f"Processing video face swap: {source_path} -> {target_path} -> {output_path}")
        
        # Build FaceFusion commands
        cmd = [
            "python", "facefusion.py",
            "headless-run",  # Run in headless mode
            "--source-paths", source_path,  # Source face image
            "--target-path", target_path,   # Target video
            "--output-path", output_path,   # Output video path
            "--processors", "face_swapper", # Enable face swapper processor
            "--face-swapper-model", "inswapper_128_fp16",  # Fast model for video
            "--face-swapper-pixel-boost", "128x128",  # Lower resolution for video (faster)
            "--output-video-quality", str(quality),  # Video quality setting
            "--output-video-encoder", "libx264",  # Standard video encoder
            "--execution-providers", "cuda",  # Use CPU (change to "cuda" for GPU)
        ]
        
        print(f"Running video command: {' '.join(cmd)}")
        
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            cwd="../facefusion",  
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"Video face swap completed successfully")
            print(f"STDOUT: {result.stdout}")
            return True
        else:
            print(f"Video face swap failed with return code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return False
            
    except asyncio.TimeoutError:
        print("Video processing timed out")
        return False
    except Exception as e:
        print(f"Video face swap error: {e}")
        return False

def cleanup_temp_files(file_paths):
    """
    Background task to cleanup temporary files (not output files)
    """
    async def cleanup():
        await asyncio.sleep(1) 
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Cleaned up temp file: {path}")
            except Exception as e:
                print(f"Cleanup error: {e}")

    asyncio.create_task(cleanup())

@app.get("/api/videos/first")
async def get_first_video():
    try:
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        files = []
        
        for file_path in OUTPUT_DIR.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                files.append((file_path, file_path.stat().st_ctime))
        
        if not files:
            raise HTTPException(status_code=404, detail="No videos found")
        
        # Neueste Datei
        latest_file = max(files, key=lambda x: x[1])[0]
        
        return {"url": f"/videos/{latest_file.name}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static files mount hinzufügen (nach den anderen imports)
app.mount("/videos", StaticFiles(directory=str(OUTPUT_DIR)), name="videos")

@app.get("/template/{template_name}", response_class=HTMLResponse)
async def serve_template(template_name: str):
    """Dynamisches Template laden"""
    try:
        # template_name = "insta_fake_mobile"  # Hardcoded
        template_file = f"template/insta_fake_mobile.html"  # Korrekter Pfad
        
        with open(template_file, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' existiert nicht")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FaceFusion API"}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with upload interface"""
    with open("insta_fake_mobile.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)