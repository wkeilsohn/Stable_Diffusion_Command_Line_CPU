# William Keilsohn
# September 3rd 2025

# Import Packages
import os
from diffusers import StableDiffusionPipeline

# Declare Variables
cpath = os.getcwd()
mfolder = os.path.join(cpath, "models")
dpath = os.path.join(cpath, "diffusers")

# Define Functions
convert_str = "python ./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path {} --dump_path {} {}"
# Check for Models

def model_folder_parter(): # Syntax needs adjustment.
    try:
        os.system("mkdir ./diffusers/models")
    except:
        pass

def model_converter():
    model_folder = os.path.join(dpath, "models")
    for i in os.scandir(mfolder):
        if i.is_file():
            md = os.path.basename(i)
            md_name = md.split(".")[0]
            ending = md.split(".")[1]
            tmp_dir = os.path.join(model_folder, md_name)
            if os.path.isdir(tmp_dir):
                pass
            else:
                if ending == "safetensors":
                    end_val = "--from_safetensors"
                else:
                    end_val = ""
                md_local = os.path.join(mfolder, md)
                cstring = convert_str.format(md_local, tmp_dir, end_val)
                os.system(cstring)

def getModels():
    dirs = []
    model_folder = os.path.join(dpath, "models")
    for i in os.listdir(model_folder):
      f = os.path.join(model_folder, i)
      if os.path.isdir(f):
            dirs.append(i)
    return dirs