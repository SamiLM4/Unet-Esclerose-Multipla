from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import cv2
from inference import predict
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/segment")
async def segment(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")

    mask = predict(image)

    mask = (mask * 255).astype(np.uint8)

    _, buffer = cv2.imencode(".png", mask)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png"
    )