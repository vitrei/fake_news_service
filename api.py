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
    output_quality: Optional[int] = 80
):
    """
    Swap faces between source and target images
    """
    request_id = str(uuid.uuid4())
    
    try:
        source_path = TEMP_DIR / f"{request_id}_source.jpg"
        target_path = TEMP_DIR / f"{request_id}_target.jpg"
        output_path = OUTPUT_DIR / f"swapped_{request_id}.jpg"  
        
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_image.file, f)
        
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_image.file, f)

        success = await process_face_swap(
            str(source_path), 
            str(target_path), 
            str(output_path),
            quality=output_quality
        )
        
        if not success or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Face swapping failed - output file not created")
        
        cleanup_temp_files([source_path, target_path])
        
        return {
            "success": True,
            "message": "Face swapping completed successfully!",
            "output_file": str(output_path),
            "filename": f"swapped_{request_id}.jpg"
        }
        
    except Exception as e:
        cleanup_temp_files([source_path, target_path])
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

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

async def process_face_swap(source_path: str, target_path: str, output_path: str, quality: int = 80):
    """
    Wrapper function for FaceFusion face swapping using CLI
    """
    try:
        print(f"Processing face swap: {source_path} -> {target_path} -> {output_path}")
        
        facefusion_paths = [
            "../facefusion",  
            "./facefusion",   
            ".",              
            "/home/merlotllm/Documents/facefusion" 
        ]
        
        facefusion_dir = None
        for path in facefusion_paths:
            if os.path.exists(os.path.join(path, "facefusion.py")):
                facefusion_dir = path
                break
        
        if not facefusion_dir:
            print("Could not find facefusion.py in any expected location")
            return False
        
        print(f"Using FaceFusion directory: {facefusion_dir}")
        
        # Build FaceFusion commands
        cmd = [
            "python", "facefusion.py", 
            "headless-run",  # Run in headless mode (no GUI)
            "--source-paths", source_path,  # Source face image
            "--target-path", target_path,   # Target image to swap face into
            "--output-path", output_path,   # Output file path
            "--processors", "face_swapper", # Enable face swapper processor
            "--face-swapper-model", "inswapper_128_fp16",  # Default model (fast and good quality)
            "--face-swapper-pixel-boost", "256x256",  # Good balance of quality and speed
            "--output-image-quality", str(quality),  # Image quality setting
            "--execution-providers", "cuda",  # Use CPU (change to "cuda" if you have GPU)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {facefusion_dir}")
        
        result = await asyncio.to_thread(
            subprocess.run, 
            cmd, 
            capture_output=True, 
            text=True,
            cwd=facefusion_dir 
        )
        
        if result.returncode == 0:
            print(f"Face swap completed successfully")
            print(f"STDOUT: {result.stdout}")
            return True
        else:
            print(f"Face swap failed with return code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"Face swap error: {e}")
        return False

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