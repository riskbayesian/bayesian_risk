import requests
import numpy as np
import io
import base64
import time

def serialize_array(array):
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def deserialize_array(encoded):
    array_bytes = base64.b64decode(encoded)
    buf = io.BytesIO(array_bytes)
    return np.load(buf)

def call_manip_seg(array):
    print(f"[main] Sending array with shape: {array.shape}")
    payload = {'data': serialize_array(array)}
    response = requests.post("http://manip_seg:8000/process", json=payload)
    result_encoded = response.json()['result']
    processed_array = deserialize_array(result_encoded)
    return processed_array

def main():
    # Wait for manip_seg to be ready
    url = "http://manip_seg:8000/"
    timeout = 60
    start = time.time()
    while True:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                break
        except Exception:
            pass
        if time.time() - start > timeout:
            raise Exception("Timeout waiting for manip_seg to be ready")
        time.sleep(1)

    large_array = np.random.rand(1000, 1000)  # Example large array
    processed_array = call_manip_seg(large_array)
    print(f"[main] Received processed array with shape: {processed_array.shape}")
    print(f"[main] Sample data: {processed_array[0, :5]}")  # print first 5 elements of first row

if __name__ == "__main__":
    main()
