from fastapi import FastAPI, Request, HTTPException
import uvicorn
import numpy as np
import base64
import io
import requests
import time
from typing import Dict, List, Any, Tuple
from pydantic import BaseModel, Field

app = FastAPI()

# Configuration for hoist_former service
HOIST_FORMER_SERVICE_URL = "http://hoist_former:5000"

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

class HoistFormerHealthResponse(BaseModel):
    """Response model for hoist_former health check"""
    hoist_former_status: str = Field(..., description="Status of hoist_former service")
    error: str = Field(default=None, description="Error message if any")

def serialize_array(array):
    """Serialize a numpy array to base64 string"""
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def deserialize_array(encoded):
    """Deserialize a base64 string back to numpy array"""
    array_bytes = base64.b64decode(encoded)
    buf = io.BytesIO(array_bytes)
    return np.load(buf)

def deserialize_list_of_arrays(encoded):
    """Deserialize a base64 string back to list of numpy arrays"""
    import pickle
    array_bytes = base64.b64decode(encoded)
    buf = io.BytesIO(array_bytes)
    return pickle.load(buf)

def wait_for_hoist_former_service(timeout: int = 60) -> bool:
    """Wait for the hoist_former service to be ready"""
    start = time.time()
    while True:
        try:
            response = requests.get(f"{HOIST_FORMER_SERVICE_URL}/")
            if response.status_code == 200:
                print(f"[main] Hoist_former service is ready")
                return True
        except Exception as e:
            pass
        
        if time.time() - start > timeout:
            print(f"[main] Timeout waiting for hoist_former service")
            return False
        
        time.sleep(1)

def test_hoist_former_service_availability() -> bool:
    """
    Test if the hoist_former service is available and responding.
    
    Returns:
        bool: True if service is available, False otherwise
    """
    try:
        # Try to connect to the health endpoint (POST method)
        response = requests.post(f"{HOIST_FORMER_SERVICE_URL}/health", timeout=10)
        if response.status_code == 200:
            print(f"[main] Hoist_former service is available and healthy")
            return True
        else:
            print(f"[main] Hoist_former service returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[main] Cannot connect to hoist_former service at {HOIST_FORMER_SERVICE_URL}")
        return False
    except requests.exceptions.Timeout:
        print(f"[main] Timeout connecting to hoist_former service")
        return False
    except Exception as e:
        print(f"[main] Error testing hoist_former service: {str(e)}")
        return False

def call_hoist_former_manipulation_detection(rgb_frames: np.ndarray, 
                                           save: bool = False, 
                                           batch_size_processing: int = 3) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Call the hoist_former service to detect manipulation objects in RGB frames.
    
    Args:
        rgb_frames: RGB frames as numpy array with shape (B, H, W, C)
        save: Whether to save debug visualizations
        batch_size_processing: Number of frames to process at once
        
    Returns:
        Tuple of (scores, labels, masks) - each is a list of numpy arrays
    """
    try:
        # Wait for service to be ready
        if not wait_for_hoist_former_service():
            raise Exception("Hoist_former service not available")
        
        # Prepare payload using Pydantic model
        payload = ManipulationDetectionRequest(
            rgb_frames=serialize_array(rgb_frames),
            save=save,
            batch_size_processing=batch_size_processing
        )
        
        print(f"[main] Sending RGB frames with shape: {rgb_frames.shape} to hoist_former service")
        
        # Make request to hoist_former service
        response = requests.post(
            f"{HOIST_FORMER_SERVICE_URL}/process_manipulation_objects",
            json=payload.dict(),
            timeout=300  # 5 minute timeout for processing
        )
        
        if response.status_code != 200:
            raise Exception(f"Hoist_former service returned status {response.status_code}: {response.text}")
        
        result = ManipulationDetectionResponse(**response.json())
        
        # Deserialize results
        scores = deserialize_list_of_arrays(result.scores)
        labels = deserialize_list_of_arrays(result.labels)
        masks = deserialize_list_of_arrays(result.masks)
        
        print(f"[main] Received results from hoist_former: {len(scores)} frames processed")
        
        return scores, labels, masks
        
    except Exception as e:
        print(f"[main] Error calling hoist_former service: {str(e)}")
        raise e

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="ok", service="main_vision_pipeline")

@app.post("/process_manipulation_objects", response_model=ManipulationDetectionResponse)
async def process_manipulation_objects(request: ManipulationDetectionRequest):
    """
    Process RGB frames to detect manipulation objects using the hoist_former service.
    
    Args:
        request: ManipulationDetectionRequest containing RGB frames and processing parameters
        
    Returns:
        ManipulationDetectionResponse containing detection results
    """
    try:
        # Decode the RGB frames
        rgb_frames_bytes = base64.b64decode(request.rgb_frames)
        rgb_frames = np.load(io.BytesIO(rgb_frames_bytes))
        
        # Call the hoist_former service
        scores, labels, masks = call_hoist_former_manipulation_detection(
            rgb_frames=rgb_frames,
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
        
        return result
        
    except Exception as e:
        print(f"[main] Error processing manipulation objects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", service="main_vision_pipeline")

@app.get("/hoist_former_health", response_model=HoistFormerHealthResponse)
async def hoist_former_health_check():
    """Check health of hoist_former service"""
    try:
        response = requests.get(f"{HOIST_FORMER_SERVICE_URL}/health")
        if response.status_code == 200:
            return HoistFormerHealthResponse(hoist_former_status="healthy")
        else:
            return HoistFormerHealthResponse(hoist_former_status="unhealthy")
    except Exception as e:
        return HoistFormerHealthResponse(hoist_former_status="unavailable", error=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
