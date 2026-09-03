import io
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

from model_loader import load_model
from inference import read_image_rgb_bytes, predict_all_damages

app = FastAPI()

session = None
config = None


@app.on_event("startup")
def startup():
    global session, config
    session, config = load_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_image_rgb_bytes(image_bytes)

    result = predict_all_damages(image, session, config)

    overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".png", overlay_bgr)

    return StreamingResponse(io.BytesIO(
        encoded.tobytes()), media_type="image/png")
