from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import tempfile
import uuid
from pathlib import Path
import asyncio
from typing import Optional
import shutil
import requests
import aiohttp

# Configuration
MIDDLEWARE_URL = "http://localhost:5010"  # Your middleware endpoint

# Import FaceFusion modules (adjust paths based on actual structure)
import sys
import subprocess
sys.path.append('./facefusion')  # Add facefusion to path

async def fetch_image_from_middleware(face_id: str, save_path: str) -> bool:
    """
    Fetch image from middleware using faceId and save it locally
    """
    try:
        print(f"Fetching image for faceId: {face_id} from {MIDDLEWARE_URL}")
        
        # Construct the middleware URL - adjust this based on your middleware API
        # Common patterns:
        # - GET /image/{face_id}
        # - GET /faces/{face_id}/image  
        # - POST /get-image with {"faceId": face_id}
        
        async with aiohttp.ClientSession() as session:
            # Option 1: Simple GET request with faceId in URL
            url = f"{MIDDLEWARE_URL}/image/{face_id}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    # Save the image data
                    with open(save_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    print(f"Successfully fetched and saved image to: {save_path}")
                    return True
                else:
                    print(f"Failed to fetch image: HTTP {response.status}")
                    
                    # Try alternative endpoint patterns
                    alt_urls = [
                        f"{MIDDLEWARE_URL}/faces/{face_id}/image",
                        f"{MIDDLEWARE_URL}/get-image/{face_id}",
                        f"{MIDDLEWARE_URL}/api/image/{face_id}"
                    ]
                    
                    for alt_url in alt_urls:
                        print(f"Trying alternative URL: {alt_url}")
                        async with session.get(alt_url) as alt_response:
                            if alt_response.status == 200:
                                with open(save_path, 'wb') as f:
                                    async for chunk in alt_response.content.iter_chunked(8192):
                                        f.write(chunk)
                                print(f"Successfully fetched from alternative URL: {alt_url}")
                                return True
                    
                    return False
                    
    except Exception as e:
        print(f"Error fetching image from middleware: {e}")
        return False

async def fetch_image_from_middleware_post(face_id: str, save_path: str) -> bool:
    """
    Alternative method using POST request (if your middleware requires POST)
    """
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"faceId": face_id}
            
            async with session.post(f"{MIDDLEWARE_URL}/get-image", json=payload) as response:
                if response.status == 200:
                    with open(save_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    return True
                    
        return False
    except Exception as e:
        print(f"Error with POST method: {e}")
        return False

# Create directories for temporary files and outputs
TEMP_DIR = Path("./temp")
OUTPUT_DIR = Path("./output")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.post("/test-middleware-call")
async def test_middleware_call():
    """
    Test endpoint to demonstrate how your middleware should call /swap-faces
    This simulates what your middleware at localhost:5010 should do
    """
    try:
        import requests
        from pathlib import Path
        
        # Example: Create or use test images
        # In your real middleware, you'd have actual images from Kinect
        
        # For testing, let's create a simple call to our own /swap-faces endpoint
        api_url = "http://localhost:8000/swap-faces"
        
        # You would replace these with actual image files from your Kinect system
        # test_source = "path/to/kinect/face/image.jpg"
        # test_target = "path/to/target/image.jpg"
        
        print("This is how your middleware should call the face swap API:")
        print(f"POST {api_url}")
        print("Files to send:")
        print("- source_image: (your Kinect face image)")
        print("- target_image: (target image to swap into)")
        print("- output_quality: 80 (optional)")
        
        return {
            "message": "Test endpoint - shows how middleware should call /swap-faces",
            "example_call": {
                "method": "POST",
                "url": "http://localhost:8000/swap-faces",
                "files": {
                    "source_image": "your_kinect_face.jpg",
                    "target_image": "target_image.jpg"
                },
                "data": {
                    "output_quality": 80
                }
            },
            "python_example": '''
import requests

# How your middleware should call this API:
files = {
    'source_image': open('kinect_face.jpg', 'rb'),
    'target_image': open('target.jpg', 'rb')
}
data = {'output_quality': 80}

response = requests.post('http://localhost:8000/swap-faces', files=files, data=data)
result = response.json()
print(result)
            '''
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/middleware-example")
async def middleware_example():
    """
    Returns example code for your middleware integration
    """
    return {
        "title": "Middleware Integration Example",
        "description": "How to call FaceFusion API from your localhost:5010 middleware",
        "examples": {
            "python_requests": '''
import requests
import os

def call_facefusion_api(source_image_path, target_image_path, quality=80):
    """
    Call FaceFusion API from your middleware
    """
    url = "http://localhost:8000/swap-faces"
    
    try:
        with open(source_image_path, 'rb') as source_file, \\
             open(target_image_path, 'rb') as target_file:
            
            files = {
                'source_image': source_file,
                'target_image': target_file
            }
            data = {'output_quality': quality}
            
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success! Output saved to: {result['output_file']}")
                return result
            else:
                print(f"Error: {response.status_code}")
                return None
                
    except Exception as e:
        print(f"Error calling FaceFusion API: {e}")
        return None

# Example usage in your middleware:
# result = call_facefusion_api("kinect_face.jpg", "target.jpg", 90)
            ''',
            "curl_example": '''
curl -X POST "http://localhost:8000/swap-faces" \\
  -F "source_image=@kinect_face.jpg" \\
  -F "target_image=@target_image.jpg" \\
  -F "output_quality=80"
            ''',
            "javascript_fetch": '''
async function callFaceFusionAPI(sourceImageFile, targetImageFile, quality = 80) {
    const formData = new FormData();
    formData.append('source_image', sourceImageFile);
    formData.append('target_image', targetImageFile);
    formData.append('output_quality', quality);
    
    try {
        const response = await fetch('http://localhost:8000/swap-faces', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('Success:', result);
            return result;
        } else {
            console.error('Error:', response.status);
            return null;
        }
    } catch (error) {
        console.error('Error calling FaceFusion API:', error);
        return null;
    }
}
            '''
        }
    }
    source_image: UploadFile = File(..., description="Source face image"),
    target_image: UploadFile = File(..., description="Target image"),
    output_quality: Optional[int] = 80
):
    """
    Swap faces between source and target images
    """
    # Generate unique IDs for this request
    request_id = str(uuid.uuid4())
    
    try:
        # Create temporary and output files
        source_path = TEMP_DIR / f"{request_id}_source.jpg"
        target_path = TEMP_DIR / f"{request_id}_target.jpg"
        output_path = OUTPUT_DIR / f"swapped_{request_id}.jpg"  # Save directly to output folder
        
        # Save uploaded files
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_image.file, f)
        
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_image.file, f)
        
        # Call FaceFusion face swapping logic
        # This is a simplified example - you'll need to adapt based on actual FaceFusion API
        success = await process_face_swap(
            str(source_path), 
            str(target_path), 
            str(output_path),
            quality=output_quality
        )
        
        if not success or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Face swapping failed - output file not created")
        
        # Cleanup temporary files only (keep output file)
        cleanup_temp_files([source_path, target_path])
        
        # Return success response with file location
        return {
            "success": True,
            "message": "Face swapping completed successfully!",
            "output_file": str(output_path),
            "filename": f"swapped_{request_id}.jpg"
        }
        
    except Exception as e:
        # Cleanup on error
        cleanup_temp_files([source_path, target_path])
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/swap-faces-video")
async def swap_faces_video(
    source_image: UploadFile = File(..., description="Source face image"),
    target_video: UploadFile = File(..., description="Target video"),
    output_quality: Optional[int] = 80
):
    """
    Swap face in a video
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Save files
        source_path = TEMP_DIR / f"{request_id}_source.jpg"
        target_path = TEMP_DIR / f"{request_id}_target.mp4"
        output_path = OUTPUT_DIR / f"swapped_video_{request_id}.mp4"  # Save directly to output folder
        
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_image.file, f)
        
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_video.file, f)
        
        # Process video (this would be more complex)
        success = await process_video_face_swap(
            str(source_path),
            str(target_path),
            str(output_path),
            quality=output_quality
        )
        
        if not success or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Video face swapping failed - output file not created")
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"swapped_video_{request_id}.mp4",
            background=cleanup_files([source_path, target_path, output_path])
        )
        
    except Exception as e:
        cleanup_temp_files([source_path, target_path])
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

async def process_face_swap(source_path: str, target_path: str, output_path: str, quality: int = 80):
    """
    Wrapper function for FaceFusion face swapping using CLI
    """
    try:
        print(f"Processing face swap: {source_path} -> {target_path} -> {output_path}")
        
        # Try to find the correct FaceFusion path
        facefusion_paths = [
            "../facefusion",  # If API is in a subdirectory
            "./facefusion",   # If API is in same directory as facefusion
            ".",              # If API is inside facefusion directory
            "/home/merlotllm/Documents/facefusion"  # Absolute path based on your error
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
        
        # Build FaceFusion command with proper arguments
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
            "--execution-providers", "cpu",  # Use CPU (change to "cuda" if you have GPU)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {facefusion_dir}")
        
        # Run the command asynchronously
        result = await asyncio.to_thread(
            subprocess.run, 
            cmd, 
            capture_output=True, 
            text=True,
            cwd=facefusion_dir  # Use the detected facefusion directory
        )
        
        # Check if command was successful
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
        
        # Build FaceFusion command for video processing
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
            "--execution-providers", "cpu",  # Use CPU (change to "cuda" for GPU)
        ]
        
        print(f"Running video command: {' '.join(cmd)}")
        
        # Run the command asynchronously (video processing takes longer)
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            cwd="../facefusion",  # Go up one level then into facefusion directory
            timeout=3600  # 1 hour timeout for video processing
        )
        
        # Check if command was successful
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
        await asyncio.sleep(1)  # Give time for processing to complete
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Cleaned up temp file: {path}")
            except Exception as e:
                print(f"Cleanup error: {e}")
    
    # Run cleanup immediately since we're not serving files anymore
    asyncio.create_task(cleanup())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FaceFusion API"}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with upload interface"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>FaceFusion API - Face Swapping Tool</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 2px dashed #dee2e6;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #4facfe;
            background: #f0f8ff;
        }
        
        .upload-section h3 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.4em;
        }
        
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
            margin-bottom: 15px;
        }
        
        .file-input {
            position: absolute;
            left: -9999px;
        }
        
        .file-input-label {
            display: block;
            padding: 15px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            cursor: pointer;
            text-align: center;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
            font-size: 1em;
        }
        
        .file-input-label:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .file-name {
            margin-top: 10px;
            padding: 10px;
            background: #e9ecef;
            border-radius: 5px;
            font-size: 0.9em;
            color: #666;
        }
        
        .quality-section {
            margin: 20px 0;
        }
        
        .quality-section label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: #333;
        }
        
        .quality-slider {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }
        
        .quality-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4facfe;
            cursor: pointer;
        }
        
        .submit-btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
        }
        
        .submit-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .progress {
            display: none;
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            text-align: center;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4facfe;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result {
            display: none;
            margin-top: 20px;
            padding: 20px;
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 10px;
            text-align: center;
        }
        
        .error {
            display: none;
            margin-top: 20px;
            padding: 20px;
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 10px;
            color: #721c24;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 5px;
        }
        
        .tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            background: transparent;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .tab.active {
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            color: #4facfe;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔄 FaceFusion API</h1>
            <p>Advanced Face Swapping Tool</p>
        </div>
        
        <div class="content">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('image')">📷 Image Swap</button>
                <button class="tab" onclick="switchTab('video')">🎥 Video Swap</button>
                <button class="tab" onclick="switchTab('faceid')">🆔 FaceID Swap</button>
            </div>
            
            <!-- Image Face Swap -->
            <div id="image-tab" class="tab-content active">
                <form id="imageForm" enctype="multipart/form-data">
                    <div class="upload-section">
                        <h3>🎭 Source Face</h3>
                        <div class="file-input-wrapper">
                            <input type="file" id="sourceImage" name="source_image" class="file-input" accept="image/*" required>
                            <label for="sourceImage" class="file-input-label">Choose Source Image</label>
                        </div>
                        <div id="sourceImageName" class="file-name" style="display: none;"></div>
                    </div>
                    
                    <div class="upload-section">
                        <h3>🖼️ Target Image</h3>
                        <div class="file-input-wrapper">
                            <input type="file" id="targetImage" name="target_image" class="file-input" accept="image/*" required>
                            <label for="targetImage" class="file-input-label">Choose Target Image</label>
                        </div>
                        <div id="targetImageName" class="file-name" style="display: none;"></div>
                    </div>
                    
                    <div class="quality-section">
                        <label for="imageQuality">Output Quality: <span id="imageQualityValue">80</span>%</label>
                        <input type="range" id="imageQuality" name="output_quality" min="1" max="100" value="80" class="quality-slider">
                    </div>
                    
                    <button type="submit" class="submit-btn">🚀 Swap Faces</button>
                </form>
            </div>
            
            <!-- Video Face Swap -->
            <div id="video-tab" class="tab-content">
                <form id="videoForm" enctype="multipart/form-data">
                    <div class="upload-section">
                        <h3>🎭 Source Face</h3>
                        <div class="file-input-wrapper">
                            <input type="file" id="sourceImageVideo" name="source_image" class="file-input" accept="image/*" required>
                            <label for="sourceImageVideo" class="file-input-label">Choose Source Image</label>
                        </div>
                        <div id="sourceImageVideoName" class="file-name" style="display: none;"></div>
                    </div>
                    
                    <div class="upload-section">
                        <h3>🎥 Target Video</h3>
                        <div class="file-input-wrapper">
                            <input type="file" id="targetVideo" name="target_video" class="file-input" accept="video/*" required>
                            <label for="targetVideo" class="file-input-label">Choose Target Video</label>
                        </div>
                        <div id="targetVideoName" class="file-name" style="display: none;"></div>
                    </div>
                    
                    <div class="quality-section">
                        <label for="videoQuality">Output Quality: <span id="videoQualityValue">80</span>%</label>
                        <input type="range" id="videoQuality" name="output_quality" min="1" max="100" value="80" class="quality-slider">
                    </div>
                    
                    <button type="submit" class="submit-btn">🚀 Swap Faces in Video</button>
                </form>
            </div>
            
            <!-- FaceID Face Swap -->
            <div id="faceid-tab" class="tab-content">
                <form id="faceidForm" enctype="multipart/form-data">
                    <div class="upload-section">
                        <h3>🆔 Face ID from Kinect</h3>
                        <input type="text" id="faceId" name="face_id" placeholder="Enter Face ID from your Kinect system" 
                               style="width: 100%; padding: 15px; border: 2px solid #dee2e6; border-radius: 10px; font-size: 1em;" required>
                        <p style="margin-top: 10px; color: #666; font-size: 0.9em;">
                            This will fetch the face image from your middleware at localhost:5010
                        </p>
                    </div>
                    
                    <div class="upload-section">
                        <h3>🖼️ Target Image</h3>
                        <div class="file-input-wrapper">
                            <input type="file" id="targetImageFaceId" name="target_image" class="file-input" accept="image/*" required>
                            <label for="targetImageFaceId" class="file-input-label">Choose Target Image</label>
                        </div>
                        <div id="targetImageFaceIdName" class="file-name" style="display: none;"></div>
                    </div>
                    
                    <div class="upload-section">
                        <h3>⚙️ Swap Direction</h3>
                        <label style="display: flex; align-items: center; margin-bottom: 10px;">
                            <input type="radio" name="use_as_source" value="true" checked style="margin-right: 10px;">
                            Use FaceID as source (replace face in target image)
                        </label>
                        <label style="display: flex; align-items: center;">
                            <input type="radio" name="use_as_source" value="false" style="margin-right: 10px;">
                            Use FaceID as target (put uploaded face into FaceID image)
                        </label>
                    </div>
                    
                    <div class="quality-section">
                        <label for="faceidQuality">Output Quality: <span id="faceidQualityValue">80</span>%</label>
                        <input type="range" id="faceidQuality" name="output_quality" min="1" max="100" value="80" class="quality-slider">
                    </div>
                    
                    <button type="submit" class="submit-btn">🚀 Swap Faces with Face ID</button>
                </form>
            </div>
            
            <div id="progress" class="progress">
                <div class="spinner"></div>
                <p>Processing your request... This may take a while.</p>
            </div>
            
            <div id="result" class="result">
                <h3>✅ Success!</h3>
                <p>Your face-swapped file has been saved to the output folder.</p>
            </div>
            
            <div id="error" class="error">
                <h3>❌ Error</h3>
                <p id="errorMessage"></p>
            </div>
        </div>
    </div>

    <script>
        // Tab switching
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            document.querySelector(`[onclick="switchTab('${tab}')"]`).classList.add('active');
            document.getElementById(`${tab}-tab`).classList.add('active');
        }
        
        // File input handlers
        document.getElementById('sourceImage').addEventListener('change', function(e) {
            showFileName('sourceImageName', e.target.files[0]);
        });
        
        document.getElementById('targetImage').addEventListener('change', function(e) {
            showFileName('targetImageName', e.target.files[0]);
        });
        
        document.getElementById('sourceImageVideo').addEventListener('change', function(e) {
            showFileName('sourceImageVideoName', e.target.files[0]);
        });
        
        document.getElementById('targetVideo').addEventListener('change', function(e) {
            showFileName('targetVideoName', e.target.files[0]);
        });
        
        document.getElementById('targetImageFaceId').addEventListener('change', function(e) {
            showFileName('targetImageFaceIdName', e.target.files[0]);
        });
        
        // Quality sliders
        document.getElementById('imageQuality').addEventListener('input', function(e) {
            document.getElementById('imageQualityValue').textContent = e.target.value;
        });
        
        document.getElementById('videoQuality').addEventListener('input', function(e) {
            document.getElementById('videoQualityValue').textContent = e.target.value;
        });
        
        document.getElementById('faceidQuality').addEventListener('input', function(e) {
            document.getElementById('faceidQualityValue').textContent = e.target.value;
        });
        
        function showFileName(elementId, file) {
            const element = document.getElementById(elementId);
            if (file) {
                element.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
                element.style.display = 'block';
            }
        }
        
        // Form submission handlers
        document.getElementById('imageForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            await submitForm('/swap-faces', new FormData(this));
        });
        
        document.getElementById('videoForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            await submitForm('/swap-faces-video', new FormData(this));
        });
        
        document.getElementById('faceidForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            await submitForm('/swap-faces-with-faceid', new FormData(this));
        });
        
        async function submitForm(endpoint, formData) {
            const progress = document.getElementById('progress');
            const result = document.getElementById('result');
            const error = document.getElementById('error');
            
            // Hide previous results
            result.style.display = 'none';
            error.style.display = 'none';
            progress.style.display = 'block';
            
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                // Parse JSON response instead of blob
                const jsonResponse = await response.json();
                
                if (jsonResponse.success) {
                    // Update result display
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = `
                        <h3>✅ Success!</h3>
                        <p>${jsonResponse.message}</p>
                        <p><strong>Output saved to:</strong> ${jsonResponse.output_file}</p>
                        <p><strong>Filename:</strong> ${jsonResponse.filename}</p>
                        <div style="margin-top: 15px; padding: 15px; background: #e7f3ff; border-radius: 8px;">
                            <p><strong>📁 Your file is ready!</strong></p>
                            <p>Check your <code>output</code> folder for the processed file.</p>
                        </div>
                    `;
                    
                    progress.style.display = 'none';
                    result.style.display = 'block';
                } else {
                    throw new Error('Processing failed');
                }
                
            } catch (err) {
                progress.style.display = 'none';
                error.style.display = 'block';
                document.getElementById('errorMessage').textContent = 
                    `Processing failed: ${err.message}. Please check your files and try again.`;
            }
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)