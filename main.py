# William Keilsohn
# September 1 2025

# Import Packages
import os
import argparse
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

# Central Functionality

def central_program(timesteps, sigmas, guidence_scale, negative_prompt, num_images_per_prompt, eta, generator, latents, prompt_embeds, negative_prompt_embeds, ip_adapter_image, ip_adapter_image_embeds, output_type, return_dict, guidence_rescale, clip_skip, callback_on_step_end, callback_on_step_end_tensor_inputs, image_name="", gen_models=False, List_Models=False, Model="", prompt=prompt, height=height, width=width, num_inference_steps=50):
    if gen_models == True:
        model_converter()
    elif List_Models == True:
        models = getModels()
        if len(models) == 0:
            print("There are no models available. Please download and convert models.")
        else:
            print(models)
    elif not Model: #Probably unnecessary to check again, but better safe than sorry.
        print("No model selected. Please select a model and re-run application.")
    else:
        pipe = diffusion_wrapper(model=Model)
        image = image_wrapper(pipe=pipe, height=height, width=width, num_inference_steps=num_inference_steps, timesteps=timesteps, sigmas=sigmas, guidence_scale=guidence_scale, negative_prompt=negative_prompt, num_images_per_prompt=num_images_per_prompt, eta=eta, generator=generator, latents=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, ip_adapter_image=ip_adapter_image, ip_adapter_image_embeds=ip_adapter_image_embeds, output_type=output_type, return_dict=return_dict, guidence_rescale=guidence_rescale, clip_skip=clip_skip, callback_on_step_end=callback_on_step_end, callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs, prompt=prompt)
        if not image_name:
            name = image_namer()
            image.save(name)
        

# Interface
def parse_maker():
    parser = argparse.ArgumentParser(description='Command Line Stable Diffusion Image Generation for Local Models')
    parser.add_argument("--timesteps", required=False, default=None)
    parser.add_argument("--sigmas", required=False, default=None)
    parser.add_argument("--guidence_scale", required=False, default=None)
    parser.add_argument("--negative_prompt", required=False, default=None)
    parser.add_argument("--num_images_per_prompt", required=False, default=1)
    parser.add_argument("--eta", required=False, default=None)
    parser.add_argument("--generator", required=False, default=None)
    parser.add_argument("--latents", required=False, default=None)
    parser.add_argument("--prompt_embeds", required=False, default=None)
    parser.add_argument("--negative_prompt_embeds", required=False, default=None)
    parser.add_argument("--ip_adapter_image", required=False, default=None)
    parser.add_argument("--ip_adapter_image_embeds", required=False, default=None)
    parser.add_argument("--output_type", required=False, default=None)
    parser.add_argument("--return_dict", required=False, default=None)
    parser.add_argument("--guidence_rescale", required=False, default=None)
    parser.add_argument("--clip_skip", required=False, default=None)
    parser.add_argument("--callback_on_step_end", required=False, default=None)
    parser.add_argument("--callback_on_step_end_tensor_inputs", required=False, default=None)
    parser.add_argument("--image_name", required=False, default="", type=str)
    parser.add_argument("--gen_models", required=False, default=False, type=bool)
    parser.add_argument("--list_models", required=False, default=False, type=bool)
    parser.add_argument("--model", required=False, default="", type=str)
    parser.add_argument("--prompt", required=False, default=prompt, type=str)
    parser.add_argument("--height", required=False, default=height, type=int)
    parser.add_argument("--width", required=False, default=width, type=int)
    parser.add_argument("--num_inference_steps", required=False, default=50, type=int)
    args = parser.parse_args()
    return args

# Run Application
if __name__ == "__main__":
    args = parse_maker()
    print(args.model)
    central_program(timesteps=args.timesteps, sigmas=args.sigmas, guidence_scale=args.guidence_scale, negative_prompt=args.negative_prompt, num_images_per_prompt=args.num_images_per_prompt, eta=args.eta, generator=args.generator, latents=args.latents, prompt_embeds=args.prompt_embeds, negative_prompt_embeds=args.negative_prompt_embeds, ip_adapter_image=args.ip_adapter_image, ip_adapter_image_embeds=args.ip_adapter_image_embeds, output_type=args.output_type, return_dict=args.return_dict, guidence_rescale=args.guidence_rescale, clip_skip=args.clip_skip, callback_on_step_end=args.callback_on_step_end, callback_on_step_end_tensor_inputs=args.callback_on_step_end_tensor_inputs, image_name=args.image_name, gen_models=args.gen_models, List_Models=args.list_models, Model=args.model, prompt=args.prompt, height=args.height, width=args.width, num_inference_steps=args.num_inference_steps)