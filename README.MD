# Stable Diffusion Command Line for CPU

A small python-based application aimed at helping individuals run local safetensor checkpoints without the use of a dedicated GPU. Essentially, this is an amalgamation of a few already available packages, but I wanted to put them all together in one place and under one set of commands so that it would be easier for someone to use and get set up.

Additionally, this application was built on a FUJJISU FMVA4100BP which does not have a GPU and instead has integrated memory. Thus, this application was somewhat built out of the necessity to produce images with lower than opitmal level tech.

## Instalation

A requierments.txt file is provided. Please make sure those packages are installed prior to instalation of this package. 

Additionally, you will still need the following git hub repository as well:

- https://github.com/huggingface/diffusers

When installing diffusers from Hugging Face, you either need to download the git repository into this one in order to create a "diffusers" folder, or you need to create your own difussers folder in this repository and copy over the <i> convert_original_stable_diffusion_to_diffusers.py </i> file from the scripts folder. Either way, the final configuration should look as follows:

```
./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py
```

Additionally, you MAY need to download and install conda from the Anaconda website.

## Setup

After you have installed the required packages and supporting applications, please place your .safetensor file in the <b> models </b> folder. Checkpoint files in the .safetensor format can either be downloaded from Hugging Face or from Civitai. 

You can use the more standard .ckpt file format as well.

```
https:\\github.com\haofanwang\Lora-for-Diffusers
```

## Use

As expected, this utility can be invoked from the command line. Generally speaking every invocation uses the standard python main.py run command:

```
python main.py
```

Parameters can be added based on the function one wishes to call:

Lists all models converted into local files from either safetensors or checkpoint files:
```
python main.py --list_models=True 
```
Converts all safetensors and/or checkpoint files into local models:
```
python main.py --gen_models=True
```

Failing to provide a model from the directory will terrminate the program and return the following:
```
No model selected. Please select a model and re-run application.
```

A prompt does NOT need to be provided, but the default prompt is as follows:
```
prompt = "An astronaut ridding a tiger"
```
If you do not want to generate this image, then a new prompt must be provided. 



## Testing

For testing this application, I primarily used the VividOrangeMix. 
You can download the model set from:

```
https://huggingface.co/WarriorMama777/OrangeMixs
```


## Lisence

This package is provided as is.