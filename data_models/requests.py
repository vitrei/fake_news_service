from pydantic import BaseModel
from typing import Optional

class FaceSwapRequest(BaseModel):
    user_id: str|int
    target_face_path: str

class FaceSwapVideoRequest(BaseModel):
    user_id: str|int
    target_video_path: str
    output_quality: Optional[int] = 80