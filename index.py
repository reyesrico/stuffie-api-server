from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/test")
async def test():
    return {"message": "Hello World"}

@app.post("/test-post")
async def test_post():
    return {"message": "Hello World"}

@app.post("/generate-caption")
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