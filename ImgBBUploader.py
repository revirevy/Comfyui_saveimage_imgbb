import requests
import base64
import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import io, time
from torchvision import transforms

import json
import torch
from comfy.model_management import get_torch_device

class LLM_prompt_generator:
    def __init__(self):
        self.type = "output"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LLM_API": ("STRING", {"default": "", "multiline": False}),
                "user_prompt": ("STRING", {"default": "", "multiline": False}),
                "model_name": (["qwen/qwen-2-7b-instruct:free",
                                "google/gemma-2-9b-it:free",
                                "mistralai/mistral-7b-instruct:free",
                                "microsoft/phi-3-mini-128k-instruct:free",
                                "meta-llama/llama-3-8b-instruct:free",
                                "gryphe/mythomist-7b:free",
                                "openchat/openchat-7b:free",
                                "undi95/toppy-m-7b:free",
                                "huggingfaceh4/zephyr-7b-beta:free",
                                "openai/shap-e",
                                "google/gemini-pro-1.5-exp",
                                "meta-llama/llama-3.1-8b-instruct:free",
                                "microsoft/phi-3-medium-128k-instruct:free",
                ], {"default": "microsoft/phi-3-medium-128k-instruct:free"}),
                "custom_model" : ("STRING", {"default": ""}),
                 "max_tokens": ("INT", {"default": 250, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "top_k": ("INT", {"default": 0, "min": 0}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "make_api_request"
    CATEGORY = "advanced/model"
    DESCRIPTION = "LLM model to generate prompt from openrouter."

    def make_api_request(self, LLM_API, user_prompt, model_name,custom_model,
    max_tokens,    temperature,    top_p,    top_k,    frequency_penalty,    presence_penalty,    repetition_penalty):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API}",
                },
                data=json.dumps({
                    "model": model_name if custom_model=="" else custom_model,
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "repetition_penalty": repetition_penalty
                })
            )
            
            if 'choices' in response.json():
                return (response.json().get('choices')[0].get('message').get('content'),)
            else:
                return ("Error: No response from API",)
        except Exception as e:
            return (f"Error: {str(e)}",)



class ImgBBUploader:
    def __init__(self):
        self.type = "output"
        
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
 
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_url", "delete_url")
 
    FUNCTION = "upload_to_imgbb"
 
    CATEGORY = "image/upload"
    DESCRIPTION = "Upload the to your imgBB account."
 
    def upload_to_imgbb(self, image, api_key, prompt=None, extra_pnginfo=None):
        if not api_key:
            return "Error: No API key provided", ""

        def upload_them(xbase64_image):
            url = "https://api.imgbb.com/1/upload"
            payload = {
                "key": api_key,
                "image": xbase64_image,
            }
            max_retries = 4
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, data=payload, timeout=60)
                    # response.raise_for_status()  # Raise an exception for bad status codes
                    result = response.json()
    
                    if result.get("success"):
                        return result["data"]["url"], result["data"]["delete_url"]
                    else:
                        error_message = result.get("error", {}).get("message", "Unknown error")
                        return f"Error: {error_message}", ""
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        return f"Error: Failed to upload after {max_retries} attempts. Last error: {str(e)}", ""
                    else:
                        print(f"Upload attempt {attempt + 1} failed. Retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
    
        
        
        # for img in image:
        for (batch_number, ximage) in enumerate(image):
            i = 255. * ximage.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            # img = transform(ximage)
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                        
            # Save the image with metadata to a byte stream
            img_byte_arr = io.BytesIO()
            
            img.save(img_byte_arr, format='PNG', pnginfo=metadata)
            img_byte_arr = img_byte_arr.getvalue()
            
            base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
            xretour = upload_them(base64_image)

        return xretour
    

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ImgBBUploader": ImgBBUploader,
    "LLM_prompt_generator": LLM_prompt_generator
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImgBBUploader": "Upload to ImgBB",
    "LLM_prompt_generator": "LLM Prompt Generator openrouter"
}
