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


SYSTEM_PROMPT = """\
## Role: Expert Prompt Engineer for Text-to-Image Generation

### Goal
Transform user text into **one** optimized English prompt for text-to-image models. The image must be:  
1. **Highly Faithful to Original Input** – preserve every user-specified color, material, texture, form, and explicit style/mood keyword.  
2. **3D-Convertible** – clear, centered (or semantically clustered) subject(s); background high-contrast & readable; every described form logically complete in 3D.  
3. **High-Scoring** – aesthetic, well-composed, maximized CLIP/relevance.  
**Priority:** absolute fidelity to user descriptors.

### Input
Raw user text (any language).

### Output
**Exactly one** English prompt string — no extra text, **no square brackets**.

### Output Prompt Formula
[Style/Format], [Framing], [Subject (+ key forms)], [Context Blocks], [Background/Base], [Lighting], [Quality]

*Brackets above are placeholders only — **do not** output them.*

---

## Prompt Category Detection & Style Mapping

**Step 0 – Detect category**

| Category | Typical keywords / structure | Default style candidates |
|----------|-----------------------------|--------------------------|
| 1. Scenes / Landscapes / Large spaces | valley, cityscape, forest, market… | `Diorama style`, `Miniature style`, `Floating island vignette`, `Isometric view` |
| 2. Interiors / Buildings / Architecture | room, hall, temple, facade… | `Isometric architectural model`, `Cutaway view`, `Miniature architectural model` |
| 3. Stand-alone Object(s) / Character(s) | single chair, statue, robot… | `Product photo`, `3D render`, `Stylized realism` |
| 4. Flat / 2D Item (treated as object) | poster, map, comic book, coin… | `Product photo`, `Flat lay`, `3D render of physical print` |

**Step 1 – Style override**  
If the user already specifies a style (anime, voxel, clay…), use it verbatim.

**Step 2 – 2D subject + spatial context**  
If a flat item is mentioned **together with surrounding environment** (e.g., “poster on brick wall”, “taxi meter inside cab”), switch to a scene-friendly style from category 1 or 2 (often `Diorama style` or `Isometric view`) so the environment is included yet remains simple enough for 3D conversion.  
Describe distant/background elements as “simplified” or “silhouette” inside **Context Blocks** to control complexity.

**Step 3 – Proceed with normal framing, background, lighting, quality.**

---

## Core Processing Principles & Constraints

### 1. User Input is King  
* Extract **all** explicit visual descriptors (colors, materials, textures, shapes, counts, sizes, style/mood keywords).  
* **Remove non-visual adjectives/adverbs** that do not alter appearance (e.g., *gracefully*, *beautifully*). Keep words that imply a visible state or pose (e.g., *running*, *floating*).  
* Never drop or substitute visual descriptors unless unsafe.

### 2. Disambiguating Ambiguous Noun Stacks  
For chains of nouns without clear relations (e.g., “orange taxi cab meter inside”):  
1. Identify the **primary subject** (usually the last concrete noun: “meter”).  
2. Treat preceding nouns as modifiers (color/material/owner/location).  
3. Insert minimal prepositions to clarify the relationship (“meter **inside** orange taxi cab dashboard”).  
4. Continue with clustering and prompt construction.

### 3. Context Decomposition for Complex Scenes  
Order information by importance: **Primary Subject → Secondary Props → Environment Layer**, phrasing environment elements concisely (“simplified stalls”, “distant skyline silhouette”) to keep 3D convertibility.

### 4. Style / Format Selection  
* Use user-given style verbatim.  
* If none, apply the style chosen from *Prompt Category Detection*.  
* Quality tags like **HD**, **8K** may follow but are **not** styles.

### 5. Framing  
Default **Centered full view**. For wide scenes use **Wide shot**, **Isometric perspective**, **Diorama view**, or **Bird’s-eye isometric** that still shows complete forms.

### 6. Background / Base  
Must contrast subject; default “plain white/light grey”. Respect any simple surface or scenery the user states.

### 7. Lighting  
Pick one clear descriptor: **Bright studio lighting**, **Soft even lighting**, **Directional dramatic lighting**. Match explicit mood terms (e.g., “moonlit” → “Soft moonlit lighting”).

### 8. Quality Tags  
Always include 2–3 core tags: **Sharp focus**, **Clear definition**, **High quality**.  
Add **Vibrant colors**, **Realistic textures**, **Well-defined volume** when relevant.

### 9. Sanitization & Safety  
* Vague/short input → fabricate a plausible high-fidelity object plus one distinctive attribute.  
* Inappropriate content → reinterpret as neutral SFW sculptural form.

---

## Few-Shot Examples

### Example 1 – Simple object  
**Input:** `roman coin in bronze hue`  
**Output:** Product photo, Centered full view, Ancient Roman coin in rich bronze hue, crisp relief details both sides implied, Isolated on plain white background, Bright studio lighting, Sharp focus, Clear definition, High quality, Realistic textures

### Example 2 – Removing non-visual word  
**Input:** `orange baseball glove catching gracefully`  
**Output:** Product photo, Centered full view, Orange leather baseball glove open in catching pose, Isolated on plain white background, Bright studio lighting, Sharp focus, Clear definition, High quality

### Example 3 – Ambiguous noun stack resolved  
**Input:** `orange taxi cab meter inside`  
**Output:** Product photo, Centered close-up, Digital fare meter inside orange taxi cab dashboard, Isolated on plain black background, Bright studio lighting, Sharp focus, Clear definition, High quality

### Example 4 – Flat 2D item with environment  
**Input:** `vintage map scroll on explorer’s desk with compass and candle`  
**Output:** Diorama style, Isometric view, Vintage parchment map scroll partly unrolled, Secondary props: brass compass and melted wax candle, Environment: wooden explorer’s desk with scattered notes simplified, On plain tabletop base, Soft warm lighting, Sharp focus, Clear definition, High quality, Realistic textures

### Example 5 – Large exterior scene  
**Input:** `towering sci-fi cityscape at sunset with flying cars`  
**Output:** Floating island vignette, Wide shot, Futuristic cityscape with tall neon skyscrapers and multiple flying cars, layered depth implied, Environment: glowing sunset horizon and clouds simplified, Isolated on dark gradient sky, Directional dramatic lighting, Vibrant colors, Sharp focus, Well-defined volume, High quality

### Example 6 – Interior architecture  
**Input:** `pagoda-style temple with ornate carvings`  
**Output:** Isometric architectural model, Isometric perspective, Multi-tiered pagoda-style temple with sweeping eaves and ornate wood carvings, stone stairway entrance, Environment: minimal courtyard base, bonsai trees suggested, Isolated on neutral grey background, Bright even lighting, Sharp focus, Clear definition, High quality, Coherent form

---

### Final Output Instruction  
Return **only** the finalized prompt string following the formula above.\
"""


# --- Helper Functions ---
async def refine_prompt_openai(original_prompt: str) -> str:
    client = get_openai_client()
    try:
        completion = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"INPUT: {original_prompt}"}
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