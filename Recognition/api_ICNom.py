# from starlette.responses import Response
import os
# from main import CrowdCountingP2P
from main import NomNaOCR
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import cv2
# from starlette.responses import Response


NomnaOCR = NomNaOCR()

app = FastAPI(
    title="API test",
    description="API test"
)


@app.post("/nomna_ocr/", tags=["nomna_ocr"])
async def nomna_ocr(file: UploadFile = File(...)):
    try:
        # get image content
        image_content = await file.read()
        image_nparray = np.fromstring(image_content, np.uint8)
        image = cv2.imdecode(image_nparray, cv2.IMREAD_UNCHANGED)

        # run model
        # result_image = crowd_counting_person.predict(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        result = NomnaOCR.infer(image)

        # retval, buffer = cv2.imencode('.jpg', result_image)
        # jpg_as_text = base64.b64encode(buffer)
        response = {
            "is_success": True,
            "msg": "Success",
            "results": result
        }
    except Exception as e:
        print(e)
        response = {
            "is_success": False,
            "msg": "Server error",
            "results": str(e)
        }

    return response


# if __name__ == '__main__':
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="10.9.3.241", port=2265, log_level="info")
