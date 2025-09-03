# William Keilsohn
# September 3 2025

# Import Packages
import os
from diffusers import StableDiffusionPipeline

# Declare Variables
prompt = "An astronaut ridding a tiger"

# Declare Functions 

# Generate Images

def image_wrapper(pipe, height, width, num_inference_steps, timesteps, sigmas, guidence_scale, negative_prompt, num_images_per_prompt, eta, generator, latents, prompt_embeds, negative_prompt_embeds, ip_adapter_image, ip_adapter_image_embeds, output_type, return_dict, guidence_rescale, clip_skip, callback_on_step_end, callback_on_step_end_tensor_inputs, prompt=prompt):
    # Acts as a wrapper for the image composition function 
    image = pipe(
        prompt = prompt,
        width = width,
        height = height,
        num_inference_steps = num_inference_steps,
        timesteps = timesteps, 
        sigmas = sigmas, 
        guidence_scale = guidence_scale, 
        negative_prompt = negative_prompt, 
        num_images_per_prompt = num_images_per_prompt, 
        eta = eta, 
        generator = generator, 
        latents = latents, 
        prompt_embeds = prompt_embeds, 
        negative_prompt_embeds = negative_prompt_embeds, 
        ip_adapter_image = ip_adapter_image, 
        ip_adapter_image_embeds = ip_adapter_image_embeds, 
        output_type = output_type, 
        return_dict = return_dict, 
        guidence_rescale = guidence_rescale, 
        clip_skip = clip_skip, 
        callback_on_step_end = callback_on_step_end, 
        callback_on_step_end_tensor_inputs = callback_on_step_end_tensor_inputs
    ).images[0]
    return image