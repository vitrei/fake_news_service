from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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
            "--execution-providers", "cpu",  # Use CPU (change to "cuda" if you have GPU)
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
            "--execution-providers", "cpu",  # Use CPU (change to "cuda" for GPU)
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
        
        // Quality sliders
        document.getElementById('imageQuality').addEventListener('input', function(e) {
            document.getElementById('imageQualityValue').textContent = e.target.value;
        });
        
        document.getElementById('videoQuality').addEventListener('input', function(e) {
            document.getElementById('videoQualityValue').textContent = e.target.value;
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