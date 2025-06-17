from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from data_models.requests import FaceSwapRequest, FaceSwapVideoRequest
from fastapi.middleware.cors import CORSMiddleware
from utils import facefusion_swap, facefusion_video_swap
from pathlib import Path


import httpx
import os
import shutil
import uuid

app = FastAPI(title="FaceFusion API", description="Face swapping API using FaceFusion")

user_profile_builder_service = "http://localhost:8010"

# Create directories
TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("output")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cleanup_temp_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"DEBUG: Cleaned up temp file: {file_path}")
            except Exception as e:
                print(f"WARNING: Could not clean up {file_path}: {e}")

def get_user_profile(user_id: str):
    """Get user profile from user profile builder service"""
    try:
        import requests
        url = f"{user_profile_builder_service}/users/{user_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"ERROR: Could not fetch user profile: {e}")
        return None

@app.post("/faceswap")
async def faceswap_api(request: FaceSwapRequest):
    """
    Face swap using user's profile image and provided target image path
    """
    try:
        async with httpx.AsyncClient() as client:
            url = f"{user_profile_builder_service}/users/{request.user_id}"
            response = await client.get(url)
            
            print(request.user_id)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
            
            response.raise_for_status()
            profile = response.json()
            
            source_image_path = profile.get("image_path")
            if not source_image_path:
                raise HTTPException(status_code=404, detail="User profile missing 'image_path'")
            
            print(f"API DEBUG: About to call with user_id='{request.user_id}', type={type(request.user_id)}, repr={repr(request.user_id)}")
            swapped_image_path = await facefusion_swap(
                source_image_path, 
                request.target_face_path, 
                quality=80,
                user_id=request.user_id
            )
            
            return {
                "success": True,
                "message": "Face swapping completed successfully!",
                "swapped_image_path": swapped_image_path,
                "user_id": request.user_id
            }
            
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Failed to call user profile service: {str(e)}")
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face swap failed: {str(e)}")

@app.post("/faceswap-video")
async def faceswap_video_api(request: FaceSwapVideoRequest):
    """
    Face swap video using user's profile image and provided target video path
    """
    try:
        async with httpx.AsyncClient() as client:
            url = f"{user_profile_builder_service}/users/{request.user_id}"
            response = await client.get(url)
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
            
            response.raise_for_status()
            profile = response.json()
            source_image_path = profile.get("image_path")
            
            if not source_image_path:
                raise HTTPException(status_code=404, detail="User profile missing 'image_path'")
            
            print(f"API DEBUG: About to call video swap with user_id='{request.user_id}', type={type(request.user_id)}, repr={repr(request.user_id)}")
            
            swapped_video_path = await facefusion_video_swap(
                source_image_path,
                request.target_video_path,
                quality=request.output_quality or 80,
                user_id=request.user_id
            )
            
            return {
                "success": True,
                "message": "Video face swapping completed successfully!",
                "swapped_video_path": swapped_video_path,
                "user_id": request.user_id
            }
            
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Failed to call user profile service: {str(e)}")
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video face swap failed: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated face swap image"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "FaceFusion API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "facefusion_api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)