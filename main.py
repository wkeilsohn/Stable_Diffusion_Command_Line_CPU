# William Keilsohn
# September 1 2025

# Import Packages
import os
import time
from datetime import timedelta
import torch
from diffusers import StableDiffusionPipeline

# Declare Variables
cpath = os.getcwd()
mfolder = os.path.join(cpath, "models")
dpath = os.path.join(cpath, "diffusers")
torch_dtype = torch.float32
device = "cpu"

convert_str = "python ./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path {} --dump_path {} {}"

# Check for Models
def model_converter():
    for i in os.scandir(mfolder):
        if i.is_file():
            md = os.path.basename(i)
            md_name = md.split(".")[0]
            ending = md.split(".")[1]
            # print(md_name)
            tmp_dir = os.path.join(dpath, md_name)
            if os.path.isdir(tmp_dir):
                pass
            else:
                if ending == "safetensors":
                    end_val = "--from_safetensors"
                else:
                    end_val = ""
                md_local = os.path.join(mfolder, md)
                cstring = convert_str.format(md_local, md_name, end_val)
                os.system(cstring)

# Declare Functions

# Generate Images


# Run Application
if __name__ == "__main__":
    model_converter()