from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
from fastapi.middleware.cors import CORSMiddleware
from utils import facefusion_swap
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

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

class FaceSwapRequest(BaseModel):
    user_id: str
    target_face_path: str

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

@app.post("/swap-faces")
async def swap_faces(
    source_image: UploadFile = File(..., description="Source face image"),
    target_image: UploadFile = File(..., description="Target image"),
    output_quality: Optional[int] = 80,
    user_id: Optional[str] = None
):
    """
    Swap faces between uploaded source and target images
    """
    request_id = str(uuid.uuid4())
    
    # Initialize paths to None for proper cleanup
    source_path = None
    target_path = None
    
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
        
        print(f"DEBUG: Source path: {source_path}")
        print(f"DEBUG: Target path: {target_path}")
        
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
        
        # Process face swap using the improved function with user_id
        output_path = await facefusion_swap(
            str(source_path),
            str(target_path),
            quality=output_quality,
            user_id=user_id
        )
        
        # Clean up temporary files
        cleanup_temp_files([source_path, target_path])
        
        response_data = {
            "success": True,
            "message": "Face swapping completed successfully!",
            "output_file": str(output_path),
            "filename": os.path.basename(output_path),
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