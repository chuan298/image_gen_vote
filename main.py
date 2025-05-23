from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from PIL import Image
import base64
import io
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
import os
import torch # For torch.Generator
import random # For generating random seeds

from model_loader import (
    load_models, get_flux_pipeline,
    get_hidream_pipeline, get_hidream_params,
    get_openai_client
)
from validator import ImageValidator

# --- Pydantic Models ---
class GenerationRequest(BaseModel):
    prompt: str
    num_images: int = Field(default=1, ge=1, le=8)
    top_k: int = Field(default=1, ge=1)
    seed: Optional[int] = Field(default=None, description="Client-provided seed. If None, server generates one.") # Optional


class ImageResponse(BaseModel):
    base64_image: str
    format: str

class GenerationResponse(BaseModel):
    refined_prompt: str
    images: List[ImageResponse]
    original_prompt: str
    model_type: str
    used_seed: int # Server will always return the seed that was used

# --- App State ---
app_state: Dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Loading models...")
    load_models()
    validator_instance = ImageValidator()
    validator_instance.validation_engine.load_pipelines()
    app_state["validator"] = validator_instance
    print("Models loaded and validator initialized.")
    yield
    print("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# --- Helper Functions ---
async def refine_prompt_openai(original_prompt: str) -> str:
    client = get_openai_client()
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert prompt engineer. Refine the user's prompt for a text-to-image AI to be more vivid, descriptive, and detailed. Output only the refined prompt."},
                {"role": "user", "content": original_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        refined = completion.choices[0].message.content.strip()
        return refined if refined else original_prompt
    except Exception as e:
        print(f"OpenAI error: {e}. Using original prompt.")
        return original_prompt

async def generate_images_flux_internal(prompt: str, num_images: int, seed: int) -> List[Image.Image]: # Seed is now mandatory
    pipeline = get_flux_pipeline()
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    output = await asyncio.to_thread(
        pipeline,
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=0.0,
        output_type="pil",
        num_images_per_prompt=num_images,
        generator=generator
    )
    
    generated_images = output.images if isinstance(output.images, list) else [output.images]

    if len(generated_images) < num_images:
        print(f"FLUX: Pipeline initially returned {len(generated_images)} images, requested {num_images}. Generating more if needed.")
        # For FLUX, if fewer images are returned than requested (num_images_per_prompt might not always be fully respected for all values)
        # we can loop to generate the remaining, varying the seed slightly for each subsequent image in the batch
        # This approach ensures `num_images` are produced while originating from the `initial_seed`
        current_images_count = len(generated_images)
        for i in range(num_images - current_images_count):
            # Create a new generator with a slightly perturbed seed for subsequent images in the batch
            # to encourage variation if the pipeline doesn't generate all `num_images` in one go.
            loop_seed = seed + current_images_count + i + 1 
            loop_generator = torch.Generator(device=pipeline.device).manual_seed(loop_seed)
            print(f"FLUX: Generating additional image {i+1} with seed {loop_seed}")
            
            img_output = await asyncio.to_thread(
                pipeline, prompt=prompt, num_inference_steps=20, guidance_scale=0.0, output_type="pil", generator=loop_generator
            )
            generated_images.extend(img_output.images)
            if len(generated_images) >= num_images:
                break
    return generated_images[:num_images]


async def generate_images_hidream_internal(prompt: str, num_images: int, seed: int) -> List[Image.Image]: # Seed is now mandatory
    pipeline = get_hidream_pipeline()
    params = get_hidream_params()
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    output = await asyncio.to_thread(
        pipeline,
        prompt=prompt,
        height=params["height"],
        width=params["width"],
        guidance_scale=params["guidance_scale"],
        num_inference_steps=params["num_inference_steps"],
        num_images_per_prompt=num_images,
        generator=generator,
        output_type="pil"
    )
    return output.images


def pil_to_base64(image: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- API Endpoint ---
@app.post("/generate", response_model=GenerationResponse)
async def generate_endpoint(request: GenerationRequest):
    if request.top_k > request.num_images:
        raise HTTPException(status_code=400, detail="top_k cannot be greater than num_images")

    active_model_type = os.getenv("ACTIVE_MODEL_TYPE", "FLUX").upper()
    
    # Determine seed: use client's if provided, else generate one
    actual_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

    try:
        print(f"Received request for prompt: '{request.prompt}', num_images: {request.num_images}, top_k: {request.top_k}, client_seed: {request.seed}, used_seed: {actual_seed}")
        print(f"Active model type: {active_model_type}")

        refined_prompt = await refine_prompt_openai(request.prompt)
        print(f"Refined prompt: '{refined_prompt}'")

        generated_images_pil: List[Image.Image] = []
        torch.set_default_device("cpu")
        if active_model_type == "FLUX":
            generated_images_pil = await generate_images_flux_internal(refined_prompt, request.num_images, actual_seed)
        elif active_model_type == "HIDREAM":
            generated_images_pil = await generate_images_hidream_internal(refined_prompt, request.num_images, actual_seed)
        else:
            raise HTTPException(status_code=500, detail=f"Invalid ACTIVE_MODEL_TYPE configured: {active_model_type}")
        torch.set_default_device("cuda")

        print(f"Generated {len(generated_images_pil)} images with {active_model_type} using seed {actual_seed}.")

        if not generated_images_pil:
            raise HTTPException(status_code=500, detail="Image generation failed or returned no images.")
        if len(generated_images_pil) < request.num_images:
            print(f"Warning: Requested {request.num_images} but generated {len(generated_images_pil)}")
            if len(generated_images_pil) < request.top_k:
                 raise HTTPException(status_code=500, detail=f"Generated only {len(generated_images_pil)} images, less than top_k ({request.top_k}).")

        validator: ImageValidator = app_state["validator"]
        selected_images_pil = await validator.validate_and_select_top_k(
            generated_images_pil, refined_prompt, request.prompt, request.top_k
        )
        print(f"Selected {len(selected_images_pil)} images after validation.")
        
        if not selected_images_pil:
             raise HTTPException(status_code=500, detail="Validation resulted in no images, or top_k selection failed.")

        response_images = []
        for img_pil in selected_images_pil:
            b64_img = await asyncio.to_thread(pil_to_base64, img_pil, "PNG")
            response_images.append(ImageResponse(base64_image=b64_img, format="PNG"))

        return GenerationResponse(
            refined_prompt=refined_prompt,
            images=response_images,
            original_prompt=request.prompt,
            model_type=active_model_type,
            used_seed=actual_seed # Return the seed that was used
        )

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)