from fastapi import FastAPI, Request
import uvicorn
import numpy as np
import base64
import io

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/process")
async def process_array(request: Request):
    payload = await request.json()
    array_bytes = base64.b64decode(payload['data'])
    array = np.load(io.BytesIO(array_bytes))

    print(f"[manip_seg] Received array with shape: {array.shape}")

    # Example processing: multiply by 2
    processed_array = array * 2

    # Serialize the processed array
    buf = io.BytesIO()
    np.save(buf, processed_array)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')

    return {"result": encoded}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
