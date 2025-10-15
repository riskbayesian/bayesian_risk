from fastapi import FastAPI, Request, HTTPException
import uvicorn
import numpy as np
import base64
import io
from typing import Dict, List, Any
import sys
import os
from pydantic import BaseModel, Field

# Add the parent directory to the path so we can import get_manip_obj_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from get_manip_obj_data import process_numpy_imgs

app = FastAPI()

# Pydantic models for request/response schemas
class ManipulationDetectionRequest(BaseModel):
    """Request model for manipulation object detection"""
    rgb_frames: str = Field(..., description="Base64 encoded numpy array of RGB frames")
    save: bool = Field(default=False, description="Whether to save debug visualizations")
    batch_size_processing: int = Field(default=3, description="Number of frames to process at once")

class ManipulationDetectionResponse(BaseModel):
    """Response model for manipulation object detection"""
    scores: str = Field(..., description="Base64 encoded list of confidence scores arrays")
    labels: str = Field(..., description="Base64 encoded list of class labels arrays")
    masks: str = Field(..., description="Base64 encoded list of segmentation masks arrays")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="ok", service="hoist_former_manipulation_segmentation")

@app.post("/process_manipulation_objects", response_model=ManipulationDetectionResponse)
async def process_manipulation_objects(request: ManipulationDetectionRequest):
    """
    Process a batch of RGB images to detect manipulation objects.
    
    Args:
        request: ManipulationDetectionRequest containing RGB frames and processing parameters
        
    Returns:
        ManipulationDetectionResponse containing detection results
    """
    try:
        # Decode the RGB frames
        rgb_frames_bytes = base64.b64decode(request.rgb_frames)
        rgb_frames = np.load(io.BytesIO(rgb_frames_bytes))
        
        print(f"[hoist_former] Received RGB frames with shape: {rgb_frames.shape}")
        
        # Process the images using the existing function
        scores, labels, masks = process_numpy_imgs(
            numpy_images=rgb_frames,
            save=request.save,
            batch_size_processing=request.batch_size_processing
        )
        
        # Serialize the results
        def serialize_list_of_arrays(arr_list):
            # Use pickle to handle inhomogeneous arrays
            import pickle
            buf = io.BytesIO()
            pickle.dump(arr_list, buf)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        
        result = ManipulationDetectionResponse(
            scores=serialize_list_of_arrays(scores),
            labels=serialize_list_of_arrays(labels),
            masks=serialize_list_of_arrays(masks)
        )
        
        print(f"[hoist_former] Processed {len(scores)} frames successfully")
        return result
        
    except Exception as e:
        print(f"[hoist_former] Error processing manipulation objects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", service="hoist_former_manipulation_segmentation")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
