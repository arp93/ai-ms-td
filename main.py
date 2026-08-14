from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
import shutil
import os

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your YOLO waste model
model = YOLO("yolov8n.pt")   # <-- your working model


@app.post("/scan-image")
async def scan_image(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO inference
    results = model(temp_path)

    detections = []

    # Parse YOLO results
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            name = model.names[cls]

            detections.append({
                "class": name,
                "confidence": conf,
                "bbox": xyxy
            })

    # Remove temp file
    os.remove(temp_path)

    return {
        "count": len(detections),
        "detections": detections
    }


@app.get("/")
def root():
    return {"message": "YOLO Waste Detection API is running!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
