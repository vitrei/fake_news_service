from pydantic import BaseModel
from typing import Optional

class FaceSwapRequest(BaseModel):
    user_id: str
    target_face_path: str

class FaceSwapVideoRequest(BaseModel):
    user_id: str
    target_video_path: str
    output_quality: Optional[int] = 80