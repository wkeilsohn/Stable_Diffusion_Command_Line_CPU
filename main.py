# William Keilsohn
# September 1 2025

# Import Packages
import os
import time
import torch
from diffusers import StableDiffusionPipeline

# Import Additional Files
from model_handler import *
from pipe_image import *

# Declare Variables
torch_dtype = torch.float32
device = "cpu" # Assumed there is no GPU present.
## Temporary Image Parameters
width, height = (1024, 1024)

# Declare Functions
def diffusion_wrapper(model, torch_dtype=torch_dtype, allow_symlinks=False):
    # Acts as a wrapper function for the normal stable diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch_dtype, local_dir_use_symlinks=allow_symlinks)
    pipe = pipe.to(device)
    return pipe

# Time Management
def image_namer():
    ctime = time.localtime()
    fstring = "%H_%M_%S"
    ctime = time.strftime(fstring, ctime)
    return ctime + ".png"

# Interface 

def start_program(timesteps, sigmas, guidence_scale, negative_prompt, num_images_per_prompt, eta, generator, latents, prompt_embeds, negative_prompt_embeds, ip_adapter_image, ip_adapter_image_embeds, output_type, return_dict, guidence_rescale, clip_skip, callback_on_step_end, callback_on_step_end_tensor_inputs, image_name="", gen_models=False, List_Models=False, Model="", prompt=prompt, height=height, width=width, num_inference_steps=50):
    # runs the user interface inside main.
    print("Hello World")
    if gen_models == True:
        model_converter()
    if List_Models == True:
        models = getModels()
        if len(models) == 0:
            print("There are no models available. Please download and convert models.")
        else:
            print(models)
    if not Model: #Probably unnecessary to check again, but better safe than sorry.
        pipe = diffusion_wrapper(model=Model)
        image = image_wrapper(pipe=pipe, height=height, width=width, num_inference_steps=num_inference_steps, timesteps=timesteps, sigmas=sigmas, guidence_scale=guidence_scale, negative_prompt=negative_prompt, num_images_per_prompt=num_images_per_prompt, eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, ip_adapter_image=ip_adapter_image, ip_adapter_image_embeds=ip_adapter_image_embeds, output_type=output_type, return_dict=return_dict, guidence_rescale=guidence_rescale, clip_skip=clip_skip, callback_on_step_end=callback_on_step_end, callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs, prompt=prompt)
        if not image_name:
            name = image_namer()
            image.save(name)
    else:
        print("No model selected. Please select a model and re-run application.")
        pass

# Run Application
if __name__ == "__main__":
    start_program()