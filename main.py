from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes
from starlette.responses import HTMLResponse
from starlette.responses import Response
import io
import cv2
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware


model = get_yolov5()

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (you may want to restrict this to specific origins in a production environment)
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow specific HTTP methods
    allow_headers=["*"],  # Allow specific headers
)


@app.get("/app", response_class=HTMLResponse)
async def serve_html():
    # Read the HTML file
    with open("./html/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get('/app/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')


@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/app/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # This should update the images with boxes and labels
    
    # Use results.ims instead of results.imgs
    if results.ims:
        for img in results.ims:
            bytes_io = io.BytesIO()
            img_base64 = Image.fromarray(img)
            img_base64.save(bytes_io, format="JPEG")
            # Since the loop would overwrite bytes_io for each img, 
            # ensure this is the desired behavior or adjust accordingly.
            return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
    else:
        # Handle the case where results.ims is empty or not as expected
        return Response(content="No images found in results", status_code=404)
