from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from fastapi.middleware.cors import CORSMiddleware

from utils import facefusion_swap

app = FastAPI(title="FaceFusion API", description="Face swapping API using FaceFusion")

user_profile_builder_service = "http://localhost:8010"

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

@app.post("/faceswap")
async def faceswap_api(request: FaceSwapRequest):
    try:
        # Call Service A to get user profile
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
        
        swapped_image_path = facefusion_swap(source_image_path, request.target_face_path)
        return {"swapped_image_path": swapped_image_path}
    
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Failed to call Service A: {str(e)}")
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face swap failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)