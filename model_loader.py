import torch
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Conditional imports for models to save memory if one is not used
flux_pipeline = None
hidream_pipeline = None
hidream_text_encoder = None
hidream_tokenizer = None
hidream_params_store = {} # For guidance_scale, num_inference_steps

openai_client = None

load_dotenv()

def _get_torch_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu": # bfloat16 not always well supported on CPU for all ops
        dtype = torch.float32
    return device, dtype

def load_flux_model():
    global flux_pipeline
    from diffusers import FluxPipeline

    model_id = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    device, dtype = _get_torch_device_and_dtype()

    flux_pipeline = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    flux_pipeline.to(device)
    print(f"FLUX Pipeline ('{model_id}') loaded on {flux_pipeline.device} with dtype {dtype}")

def load_hidream_model():
    global hidream_pipeline, hidream_text_encoder, hidream_tokenizer, hidream_params_store
    from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
    from diffusers import HiDreamImagePipeline

    device, dtype = _get_torch_device_and_dtype()

    text_encoder_id = os.getenv("HIDREAM_TEXT_ENCODER_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    variant = os.getenv("HIDREAM_MODEL_VARIANT", "Full").lower()

    # It's crucial to load tokenizer and text_encoder with appropriate precision
    # Llama models are large, bf16/fp16 is recommended on GPU
    # On CPU, fp32 might be more stable if bf16/fp16 is not well supported by all ops
    text_encoder_dtype = dtype if device == "cuda" else torch.float32 

    print(f"Loading HiDream Tokenizer ('{text_encoder_id}')...")
    hidream_tokenizer = PreTrainedTokenizerFast.from_pretrained(text_encoder_id)
    
    print(f"Loading HiDream Text Encoder ('{text_encoder_id}') with dtype {text_encoder_dtype}...")
    hidream_text_encoder = LlamaForCausalLM.from_pretrained(
        text_encoder_id,
        output_hidden_states=True, # Required by HiDream
        output_attentions=True,    # Required by HiDream
        torch_dtype=text_encoder_dtype, # Use appropriate dtype
        # device_map="auto" # consider for very large models if accelerate is configured
    )
    hidream_text_encoder.to(device) # Ensure text encoder is on the correct device
    print(f"HiDream Text Encoder loaded on {hidream_text_encoder.device}.")

    if variant == "full":
        hidream_params_store["guidance_scale"] = 5.0
        hidream_params_store["num_inference_steps"] = 50
        pipeline_id = "HiDream-ai/HiDream-I1-Full"
    elif variant == "dev":
        hidream_params_store["guidance_scale"] = 0.0
        hidream_params_store["num_inference_steps"] = 28
        pipeline_id = "HiDream-ai/HiDream-I1-Dev"
    elif variant == "fast":
        hidream_params_store["guidance_scale"] = 0.0
        hidream_params_store["num_inference_steps"] = 16
        pipeline_id = "HiDream-ai/HiDream-I1-Fast"
    else:
        raise ValueError(f"Invalid HIDREAM_MODEL_VARIANT: {variant}. Choose Full, Dev, or Fast.")
    
    hidream_params_store["height"] = int(os.getenv("HIDREAM_IMAGE_HEIGHT", "1024"))
    hidream_params_store["width"] = int(os.getenv("HIDREAM_IMAGE_WIDTH", "1024"))

    print(f"Loading HiDream Pipeline ('{pipeline_id}') with dtype {dtype}...")
    hidream_pipeline = HiDreamImagePipeline.from_pretrained(
        pipeline_id,
        tokenizer_4=hidream_tokenizer,
        text_encoder_4=hidream_text_encoder,
        torch_dtype=dtype,
    )
    hidream_pipeline.to(device)
    print(f"HiDream Pipeline ('{pipeline_id}') loaded on {hidream_pipeline.device} with dtype {dtype}")
    print(f"HiDream params: {hidream_params_store}")


def load_models():
    global openai_client
    active_model_type = os.getenv("ACTIVE_MODEL_TYPE", "FLUX").upper()

    torch.set_default_device("cpu")
    if active_model_type == "FLUX":
        load_flux_model()
    elif active_model_type == "HIDREAM":
        load_hidream_model()
    else:
        raise ValueError(f"Unsupported ACTIVE_MODEL_TYPE: {active_model_type}. Choose FLUX or HIDREAM.")
    torch.set_default_device("cuda")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    print("OpenAI client initialized.")

def get_flux_pipeline():
    if flux_pipeline is None:
        raise RuntimeError("FLUX pipeline not loaded. Is ACTIVE_MODEL_TYPE set to FLUX?")
    return flux_pipeline

def get_hidream_pipeline():
    if hidream_pipeline is None:
        raise RuntimeError("HiDream pipeline not loaded. Is ACTIVE_MODEL_TYPE set to HIDREAM?")
    return hidream_pipeline

def get_hidream_params():
    if not hidream_params_store:
        raise RuntimeError("HiDream params not loaded. Is ACTIVE_MODEL_TYPE set to HIDREAM?")
    return hidream_params_store

def get_openai_client():
    if openai_client is None:
        raise RuntimeError("OpenAI client not loaded. Call load_models() first.")
    return openai_client