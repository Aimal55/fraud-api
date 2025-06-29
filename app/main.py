from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import ast
import onnxruntime as ort
from ocr_pipeline.pipeline import run_ocr  # your OCR function

app = FastAPI()

# Load ONNX model
onnx_session = ort.InferenceSession("model/saved_model/model.onnx")
input_name = onnx_session.get_inputs()[0].name

device_usage_counts = {}

# Define request schema with raw fields
class TransactionData(BaseModel):
    amount: float
    bin: int 
    device_id: str
    geo: str  
    

class ScoreRequest(BaseModel):
    transaction: TransactionData
    receipt_path: str

@app.post("/score")
def score_endpoint(request: ScoreRequest):
    try:
        amount = request.transaction.amount
        device_id = request.transaction.device_id
        geo_str = request.transaction.geo
        bin_value = request.transaction.bin  # <-- new line

        # Optional: you can print/log bin_value if needed:
        print(f"Received BIN: {bin_value}")

        # Parse geo tuple
        try:
            geo_lat, geo_lon = ast.literal_eval(geo_str)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid geo format, expected '(lat, lon)' string.")

        # Device count
        count = device_usage_counts.get(device_id, 0) + 1
        device_usage_counts[device_id] = count
        device_tx_count = count

        # Compute Geo distance
        R = 6371
        lat1, lon1 = np.radians(25.0), np.radians(67.0)
        lat2, lon2 = np.radians(geo_lat), np.radians(geo_lon)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        geo_distance = R * c

        # Feature array
        features = np.array([[amount, device_tx_count, geo_distance]], dtype=np.float32)

        # Fraud model prediction
        pred = onnx_session.run(None, {input_name: features})[0]
        fraud_score = float(pred[0])

        # OCR result
        ocr_result = run_ocr(request.receipt_path)

        return {
            "fraud_score": fraud_score,
            "merchant_name": ocr_result["merchant_name"],
            "total": ocr_result["total"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")