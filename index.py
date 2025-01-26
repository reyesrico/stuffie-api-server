from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image

app = FastAPI()

# Allow CORS for all origins (you can restrict this to specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model and processor
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

class ImageURL(BaseModel):
    url: str

def add_cors_headers(response: JSONResponse):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.post("/generate-caption", dependencies=[Depends(add_cors_headers)])
async def generate_caption(image_url: ImageURL):
    try:
        image = Image.open(requests.get(image_url.url, stream=True).raw)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"caption": generated_caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server with `uvicorn index:app`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)