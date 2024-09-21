import requests
import base64
import cv2
import numpy as np

class ImgBBUploader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }
 
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_url", "delete_url")
 
    FUNCTION = "upload_to_imgbb"
 
    CATEGORY = "image/upload"
 
    def upload_to_imgbb(self, image, api_key):
        if not api_key:
            return "Error: No API key provided", ""

        # Convert image tensor to base64
        # image_data = image.cpu().numpy()[0].transpose(1, 2, 0)
        # image_data = (image_data * 255).astype('uint8')
        # _, buffer = cv2.imencode('.png', image_data)
        # base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Convert image tensor to numpy array
        image_data = image.cpu().numpy()[0].transpose(1, 2, 0)
        
        # Ensure the image data is in the correct range (0-255) and type (uint8)
        image_data = np.clip(image_data * 255, 0, 255).astype(np.uint8)
        
        # Check image dimensions and channels
        if image_data.shape[2] not in [1, 3, 4]:
            # Convert to RGB if necessary
            image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        
        try:
            _, buffer = cv2.imencode('.png', image_data)
        except cv2.error as e:
            return f"Error encoding image: {str(e)}", ""
        
        base64_image = base64.b64encode(buffer).decode('utf-8')

    
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": api_key,
            "image": base64_image,
        }
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = requests.post(url, data=payload, timeout=60)
                response.raise_for_status()  # Raise an exception for bad status codes
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

        return "Error: Unexpected end of upload function", ""


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ImgBBUploader": ImgBBUploader
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImgBBUploader": "Upload to ImgBB"
}
