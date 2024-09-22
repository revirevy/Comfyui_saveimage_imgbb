import requests
import base64
import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import io
from torchvision import transforms


# class SaveImage:
#     def __init__(self):
#         self.output_dir = folder_paths.get_output_directory()
#         self.type = "output"
#         self.prefix_append = ""
#         self.compress_level = 4

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "images": ("IMAGE", {"tooltip": "The images to save."}),
#                 "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
#             },
#             "hidden": {
#                 "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
#             },
#         }

#     RETURN_TYPES = ()
#     FUNCTION = "save_images"

#     OUTPUT_NODE = True

#     CATEGORY = "image"
#     DESCRIPTION = "Saves the input images to your ComfyUI output directory."

#     def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
#         filename_prefix += self.prefix_append
#         full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
#         results = list()
#         for (batch_number, image) in enumerate(images):
#             i = 255. * image.cpu().numpy()
#             img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
#             metadata = None
#             if not args.disable_metadata:
#                 metadata = PngInfo()
#                 if prompt is not None:
#                     metadata.add_text("prompt", json.dumps(prompt))
#                 if extra_pnginfo is not None:
#                     for x in extra_pnginfo:
#                         metadata.add_text(x, json.dumps(extra_pnginfo[x]))

#             filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
#             file = f"{filename_with_batch_num}_{counter:05}_.png"
#             img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
#             results.append({
#                 "filename": file,
#                 "subfolder": subfolder,
#                 "type": self.type
#             })
#             counter += 1

#         return { "ui": { "images": results } }



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
        }
 
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_url", "delete_url")
 
    FUNCTION = "upload_to_imgbb"
 
    CATEGORY = "image/upload"
    DESCRIPTION = "Upload the to your imgBB account."
 
    def upload_to_imgbb(self, image, api_key):
        if not api_key:
            return "Error: No API key provided", ""

        # Convert image tensor to base64
        # image_data = image.cpu().numpy()[0].transpose(1, 2, 0)
        # image_data = (image_data * 255).astype('uint8')
        # _, buffer = cv2.imencode('.png', image_data)
        # base64_image = base64.b64encode(buffer).decode('utf-8')

        
        # # Convert image tensor to numpy array
        # image_data = image.cpu().numpy()[0].transpose(1, 2, 0)
        
        # # Ensure the image data is in the correct range (0-255) and type (uint8)
        # image_data = np.clip(image_data * 255, 0, 255).astype(np.uint8)
        
        # # # Check image dimensions and channels
        # # if image_data.shape[2] not in [1, 3, 4]:
        # #     # Convert to RGB if necessary
        # #     image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        
        # # try:
        # #     _, buffer = cv2.imencode('.png', image_data)
        # # except cv2.error as e:
        # #     return f"Error encoding image: {str(e)}", ""

        transform = transforms.ToPILImage()
        
        # for img in image:
        for (batch_number, ximage) in enumerate(image):
            i = 255. * ximage.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            # img = transform(ximage)
            
            # Save the image with metadata to a byte stream
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
            upload_them(base64_image)
            
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
    
            return "Error: Unexpected end of upload function", ""


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ImgBBUploader": ImgBBUploader
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImgBBUploader": "Upload to ImgBB"
}
