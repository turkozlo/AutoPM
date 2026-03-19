"""
Local LLM Server - OpenAI-compatible API powered by HuggingFace Transformers.

Exposes /v1/chat/completions and /v1/models endpoints.
Designed to run FULLY OFFLINE (no internet access).

Usage:
    python server.py --model models/Qwen_Qwen2.5.14B-Instruct --port 5000
"""

import argparse
import os
import time
import uuid
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------
#  Force OFFLINE mode for HuggingFace
# ---------------------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# ---------------------------------------------------------------
#  Pydantic schemas (OpenAI-compatible)
# ---------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[ChatCompletionChoice] = []
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo] = []


# ---------------------------------------------------------------
#  Global model state
# ---------------------------------------------------------------
_model = None
_tokenizer = None
_model_name = ""
_device = "cpu"


def load_model(model_path: str):
    """Load model and tokenizer from a local directory."""
    global _model, _tokenizer, _model_name, _device

    print(f"[LLM Server] Loading model from: {model_path}")
    print(f"[LLM Server] OFFLINE mode: HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")

    _tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    # Decide device
    if torch.cuda.is_available():
        _device = "cuda"
        print(f"[LLM Server] CUDA available. GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        _device = "cpu"
        print("[LLM Server] No GPU detected, running on CPU (will be slow).")

    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        device_map="auto" if _device == "cuda" else None,
    )

    if _device == "cpu":
        _model = _model.to(_device)

    _model.eval()
    _model_name = model_path
    print(f"[LLM Server] Model loaded successfully on {_device}.")


# ---------------------------------------------------------------
#  FastAPI app
# ---------------------------------------------------------------
app = FastAPI(title="Local LLM Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
def list_models():
    """Return list of available models (OpenAI-compatible)."""
    return ModelListResponse(data=[ModelInfo(id=_model_name)])


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """Generate a chat completion (OpenAI-compatible)."""
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # Build prompt using chat template
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if hasattr(_tokenizer, "apply_chat_template"):
            input_ids = _tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(_model.device)
        else:
            # Fallback: manual prompt construction
            prompt_text = ""
            for m in messages:
                prompt_text += f"<|{m['role']}|>\n{m['content']}\n"
            prompt_text += "<|assistant|>\n"
            input_ids = _tokenizer(prompt_text, return_tensors="pt").input_ids.to(_model.device)

        prompt_len = input_ids.shape[1]

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": max(request.temperature, 0.01),
            "top_p": request.top_p,
            "do_sample": request.temperature > 0.01,
            "pad_token_id": _tokenizer.eos_token_id,
        }

        # Handle stop sequences
        if request.stop:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class StopOnTokens(StoppingCriteria):
                def __init__(self, stop_ids_list):
                    self.stop_ids_list = stop_ids_list

                def __call__(self, input_ids, scores, **kwargs):
                    for stop_ids in self.stop_ids_list:
                        seq_len = len(stop_ids)
                        if input_ids.shape[1] >= seq_len:
                            if input_ids[0, -seq_len:].tolist() == stop_ids:
                                return True
                    return False

            stop_ids_list = []
            for s in request.stop:
                ids = _tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids_list.append(ids)

            if stop_ids_list:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens(stop_ids_list)])

        # Generate
        with torch.no_grad():
            output_ids = _model.generate(input_ids, **gen_kwargs)

        # Decode only the new tokens
        new_tokens = output_ids[0][prompt_len:]
        response_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_tokens = len(new_tokens)

        return ChatCompletionResponse(
            model=_model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                total_tokens=prompt_len + completion_tokens,
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "model": _model_name,
        "device": _device,
        "model_loaded": _model is not None,
    }


# ---------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local LLM Server (OpenAI-compatible)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to local model directory (e.g. models/Qwen_Qwen2.5.14B-Instruct)")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    load_model(args.model)
    print(f"[LLM Server] Starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
