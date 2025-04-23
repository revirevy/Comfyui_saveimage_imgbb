import requests
import os
import base64
import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import io, time
from torchvision import transforms
from pyuploadcare import Uploadcare
import tempfile
import datetime
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
                                "microsoft/phi-3-medium-128k-instruct:free",
                                
                                "meta-llama/llama-3-8b-instruct:free",
                                 "meta-llama/llama-3.1-8b-instruct:free",
                                "meta-llama/llama-3.1-70b-instruct:free",
                                "meta-llama/llama-3.1-405b-instruct:free",
                                
                                "meta-llama/llama-3.2-1b-instruct:free",
                                "meta-llama/llama-3.2-3b-instruct:free",
                                "meta-llama/llama-3.2-11b-vision-instruct:free",
                                "meta-llama/llama-3.2-90b-vision-instruct:free",
                                
                                
                                "gryphe/mythomist-7b:free",
                                "openchat/openchat-7b:free",
                                "undi95/toppy-m-7b:free",
                                "huggingfaceh4/zephyr-7b-beta:free",
                                "openai/shap-e",
                                
                                "google/gemini-flash-1.5-8b-exp",
                                "google/gemini-pro-1.5-exp",
                                "google/gemini-exp-1121:free",
                                "google/learnlm-1.5-pro-experimental:free",
                                "google/gemini-exp-1114:free",

                                "google/gemini-2.0-flash-thinking-exp:free",
                                "google/gemini-2.0-flash-thinking-exp-1219:free",
                                "google/gemini-2.0-flash-exp:free",
                                "google/gemini-exp-1206:free",
                                                              
                                
                                "nousresearch/hermes-3-llama-3.1-405b:free",
                                "liquid/lfm-40b:free",

                                "sophosympatheia/rogue-rose-103b-v0.2:free"



                                
                                
                ], {"default": "meta-llama/llama-3.2-11b-vision-instruct:free"}),
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
                "platform": (["imgbb","Uploadcare","PhotoPrism"], {"default": "PhotoPrism", "multiline": False}),
                "host_address": ("STRING", {"default": "", "multiline": False}),
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
 
    def upload_to_imgbb(self, image, api_key, platform, host_address, prompt=None, extra_pnginfo=None):
        if not api_key:
            return "Error: No API key provided", ""

        def upload_them(xbase64_image,image_name=""):
            if platform in ["imgbb",""] : 
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
            elif platform ==  "Uploadcare" :
                # url = "https://api.imgbb.com/1/upload"
                pubkey,seckey = api_key.split('|')
                if not pubkey or not seckey:
                    return "Error: No API key provided", ""
                uploadcare = Uploadcare(public_key=pubkey, secret_key=seckey)
                uploaded_file = uploadcare.upload(xbase64_image)
                return uploaded_file.cdn_url,""
                
            elif platform == "PhotoPrism":  # New PhotoPrism upload logic
  
                xfiles = {'file': (f"instantid_{int(time.time())}.png" ,xbase64_image, 'image/png')}
                print(f"instantid_{int(time.time())}.png")
                try:
                    upload_url = f"{host_address}/upload"

                    # Make the POST request to upload the photo
                    # response = requests.post(upload_url, files=xfiles, headers=headers, timeout=60)
                    response = requests.post(upload_url, auth=("", api_key), files=xfiles, verify=False, timeout=10)
                    
                    # Raise an error for bad responses (4xx and 5xx)
                    response.raise_for_status()
                    
                    return 'File uploaded successfully' if response.status_code==200 else f'Erreur importation code:{response.status_code} ',response.status_code 

                except requests.exceptions.RequestException as e:
                    return f"Error uploading to PhotoPrism: {str(e)}", ""  



        
        # for img in image:
        for (batch_number, ximage) in enumerate(image):
            i = 255. * ximage.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            # img = transform(ximage)
            metadata = PngInfo()
            current_time = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
            
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
                metadata.add_text("DateTime", current_time) 

            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                        

            if platform ==  "Uploadcare" or platform ==  "PhotoPrism":
                with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as temp_file:
                    # Save the image with metadata to the temporary file
                    img.save(temp_file, format='PNG', pnginfo=metadata)
                    temp_file_path = temp_file.name
                
                    # Open the temporary file in binary read mode
                    with open(temp_file_path, 'rb') as file_obj:
                        # Now you can use fileno() and os.fstat()
                        # print(os.fstat(file_obj.fileno()))
                        xretour = upload_them(file_obj,temp_file_path)
                        # xretour = (xretour[0] + "\nFILESTAT\n" + os.fstat(file_obj.fileno()).__str__(), xretour[1])

            elif  platform ==  "imgbb"  :
                # Save the image with metadata to a byte stream
                img_byte_arr = io.BytesIO()
                
                img.save(img_byte_arr, format='PNG', pnginfo=metadata)
                # Create a temporary file
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
