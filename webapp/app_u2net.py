import io
import logging
import time
import uuid

from PIL import Image
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse

from webapp.u2net import engine_u2net

app = FastAPI()


@app.post("/remove-bg", response_class=HTMLResponse)
async def remove_bg(request: Request, file: UploadFile = File(...)):
    try:
        start_time = time.time()  # Start time

        new_name = str(uuid.uuid4()).split("-")[0]
        ext = file.filename.split(".")[-1]
        # Read file into an in-memory buffer
        in_memory_file = io.BytesIO(file.file.read())

        # Load image directly from in-memory buffer
        input_image = Image.open(in_memory_file)
        img_pil = engine_u2net.remove_bg_mult(input_image)

        # Create another in-memory buffer to store the processed image
        img_output = io.BytesIO()
        img_pil.save(img_output, format='PNG')
        img_output.seek(0)

        # End time and calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        headers = {"S-Processing-Time": str(processing_time), "S-Processing-Server": "V1"}

        # Determine the MIME type
        mime_type = "image/png"

        return StreamingResponse(img_output, media_type=mime_type, headers=headers)


    except Exception as ex:
        logging.info(ex)
        print(ex)
        return JSONResponse(status_code=400, content={"error": str(ex)})
