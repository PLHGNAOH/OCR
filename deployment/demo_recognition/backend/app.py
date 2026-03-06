import tempfile
import json
import os
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import requests
from PIL import Image

from model import predict

app = FastAPI()


@app.get("/ocr")
async def ocr_url(image_url: str):

    try:
        response = requests.get(image_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
            temp.write(response.content)
            temp_path = temp.name

        # 🔥 Bây giờ predict trả về (image đã vẽ, predictions)
        image, predictions = predict(temp_path)

        os.unlink(temp_path)

        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        buffer.seek(0)

        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={"X-Predictions": json.dumps(predictions)}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/upload")
async def ocr_upload(file: UploadFile = File(...)):

    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be image")

        content = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
            temp.write(content)
            temp_path = temp.name

        # 🔥 Predict trả ảnh đã annotate
        image, predictions = predict(temp_path)

        os.unlink(temp_path)

        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        buffer.seek(0)

        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={"X-Predictions": json.dumps(predictions)}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))