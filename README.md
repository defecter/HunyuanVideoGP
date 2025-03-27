<!-- ## **HunyuanVideo** -->

[中文阅读](./README_zh.md)


# HunyuanVideoGP: Text2Video and Image2Video Generation for the GPU Poor
<div align="center">
  <a href="https://github.com/Tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo Code&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://aivideo.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green&logo=github-pages"></a> &ensp;
  <a href="https://video.hunyuan.tencent.com"><img src="https://img.shields.io/static/v1?label=Playground&message=Web&color=green&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2412.03603"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv:HunyuanVideo&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo"><img src="https://img.shields.io/static/v1?label=HunyuanVideo&message=HuggingFace&color=yellow"></a> &ensp; &ensp;
  <a href="https://huggingface.co/tencent/HunyuanVideo-PromptRewrite"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-PromptRewrite&message=HuggingFace&color=yellow"></a> &ensp; &ensp;

</div>
<p align="center">


## News

* 03/27/2025 version 6.3: Official RTX 50xx support (see installation instructions below)  + added support for a AccVideo distilled HunyanVideo model. Only needs 5 steps. Not as good as the original model but seems better that Fast HuyuanVideo.  
* 03/15/2025 version 6.2: 
  Lora Fest special edition: very fast loading / unload of loras for those Loras collectors around. You can also now add / remove loras in the Lora folder without restarting the app. You will need to refresh the requirements *pip install -r
* 03/07/2025: Version 6.1: Upgraded HunyanVideo Image to Video with new model released today (that obviously replaces the model of yesterday) \ 
* 03/06/2025: Version 6.0: Support for HunyanVideo Image to Video with Fast generation, Low VRAM (up to 12s video) and Lora support\ 
          You need to do a **pip install -r requirements.txt** if you had already installed the app
* 02/27/2025: Version 5.1: Added Loras Preset to easily store and share combinations of loras and their multipliers 
* 02/25/2025: Version 5.0: **Out Of this World Release by DeepBeepMeep that lands only in HunyuanVideo GP: VRAM laws have been broken as VRAM consumption has been divided by 3 and 20%-50% faster at no quality loss !**

*You can now generate videos that lasts up to 10s of 1280x720 and 16s of 848x480 with 24 GB of VRAM with Loras and no quantization !!!*

Welcome to low VRAM GPUs owners as from now on you can generate multiseconds videos.

Many thanks to RIFLEx (https://github.com/thu-ml/RIFLEx) and their very good released timing, for their positional embeddign breakthrough that allows generating videos longer than up to 10s that doesn't look like still life.

Please note that although there will be still sufficient VRAM left, generating video longer than 10s with Hunyuan current models is useless as the videos starts to get redundant

If you have already installed HunyuanVideoGP, you will need to run *pip install -r requirements.txt*. Upgrading to python 2.6.0 and the corresponding attention libaries is a plus for performance.

* 03/10/2025: Version 4.1: Improved lora presets, they can now  include prompts and comments to guide the user 
* 02/11/2025: Version 4.0 Quality of life features: fast abort video generation, detect automatically attention modes not supported, you can now change video engine parameters without having to restart the app
* 02/11/2025: Version 3.5 optimized lora support (reduced VRAM requirements and faster). You can now generate 1280x720 97 frames with Loras in 3 minutes only in the fastest mode
* 02/10/2025: Version 3.4 New --fast and --fastest switches to automatically get the best performance
* 02/10/2025: Version 3.3 Prefill automatically optimal parameters for Fast Hunyuan
* 02/07/2025: Version 3.2 Added support for Xformers attention and reduce VRAM requirements for sdpa attention
* 01/21/2025: Version 3.1 Ability to define a Loras directory and turn on / off any Lora when running the application
* 01/11/2025: Version 3.0 Multiple prompts / multiple generations per prompt, new progression bar, support for pretrained Loras
* 01/06/2025: Version 2.1 Integrated Tea Cache (https://github.com/ali-vilab/TeaCache) for even faster generations
* 01/04/2025: Version 2.0 Full leverage of mmgp 3.0 (faster and even lower RAM requirements ! + support for compilation on Linux and WSL)
* 12/22/2024: Version 1.0 First release

## Features
*GPU Poor version by **DeepBeepMeep**. This great video generator can now run smoothly on any GPU.*

This version has the following improvements over the original Hunyuan Video model:
- Reduce greatly the RAM requirements and VRAM requirements
- Much faster thanks to compilation and fast loading / unloading
- 5 profiles in order to able to run the model at a decent speed on a low end consumer config (32 GB of RAM and 12 VRAM) and to run it at a very good speed on a high end consumer config (48 GB of RAM and 24 GB of VRAM)
- Autodownloading of the needed model files
- Improved gradio interface with progression bar and more options
- Multiples prompts / multiple generations per prompt
- Support multiple pretrained Loras with 32 GB of RAM or less
- Switch easily between Hunyuan and Fast Hunyuan models and quantized / non quantized models
- Much simpler installation



This fork by DeepBeepMeep is an integration of the mmpg module on the gradio_server.py.

It is an illustration on how one can set up on an existing model some fast and properly working CPU offloading with changing only a few lines of code in the core model.

For more information on how to use the mmpg module, please go to: https://github.com/deepbeepmeep/mmgp

You will find the original Hunyuan Video repository here: https://github.com/Tencent/HunyuanVideo
 


## Installation Guide for Linux and Windows for GPUs up to RTX40xx

**If you are looking for a one click installation, just go to the Pinokio App store : https://pinokio.computer/**

Otherwise you will find the instructions below:

This app has been tested on Python 3.10 / 2.6.0  / Cuda 12.4.

```shell
# 0 Download the source and create a Python 3.10.9 environment using conda or create a venv using python
git clone https://github.com/deepbeepmeep/HunyuanVideoGP
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp

# 1 Install pytorch 2.6.0
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124  

# 2. Install pip dependencies
pip install -r requirements.txt

# 3.1 optional Sage attention support (30% faster)
# Windows only: extra step only needed for windows as triton is included in pytorch with the Linux version of pytorch
pip install triton-windows 
# For both Windows and Linux
pip install sageattention==1.0.6 


# 3.2 optional Sage 2 attention support (40% faster)
# Windows only
pip install triton-windows 
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu126torch2.6.0-cp310-cp310-win_amd64.whl
# Linux only (sorry only manual compilation for the moment, but is straight forward with Linux)
git clone https://github.com/thu-ml/SageAttention
cd SageAttention 
pip install -e .

# 3.3 optional Flash attention support (easy to install on Linux but may be complex on Windows as it will try to compile the cuda kernels)
pip install flash-attn==2.7.2.post1

```

Note pytorch *sdpa attention* is available by default. It is worth installing *Sage attention* (albout not as simple as it sounds) because it offers a 30% speed boost over *sdpa attention* at a small quality cost.
In order to install Sage, you will need to install also Triton. If Triton is installed you can turn on *Pytorch Compilation* which will give you an additional 20% speed boost and reduced VRAM consumption.

## Installation Guide for Linux and Windows for GPUs up to RTX50xx
RTX50XX are only supported by pytorch starting from pytorch 2.7.0 which is still in beta. Therefore this version may be less stable.\
It is important to use Python 3.10 otherwise the pip wheels may not be compatible.
```
# 0 Download the source and create a Python 3.10.9 environment using conda or create a venv using python
git clone https://github.com/deepbeepmeep/HunyuanVideoGP
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp

# 1 Install pytorch 2.7.0:
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128

# 2. Install pip dependencies
pip install -r requirements.txt

# 3.1 optional Sage attention support (30% faster)
# Windows only: extra step only needed for windows as triton is included in pytorch with the Linux version of pytorch
pip install triton-windows 
# For both Windows and Linux
pip install sageattention==1.0.6 


# 3.2 optional Sage 2 attention support (40% faster)
# Windows only
pip install triton-windows 
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu128torch2.7.0-cp310-cp310-win_amd64.whl 

# Linux only (sorry only manual compilation for the moment, but is straight forward with Linux)
git clone https://github.com/thu-ml/SageAttention
cd SageAttention 
pip install -e .
```


## Run the application

### Run a Gradio Server on port 7860 (recommended)
To run the Text to Video application:
```bash
python gradio_server.py
```

To run the Image to Video application:
```bash
python gradio_server.py --i2v
```

Every lora stored in the subfoler 'loras' for t2v and 'loras_i2v' will be automatically loaded. You will be then able to activate / desactive any of them when running the application by selecting them in the area below "Activated Loras" .

For each activated Lora, you may specify a *multiplier* that is one float number that corresponds to its weight (default is 1.0) .The multipliers for each Lora shoud be separated by a space character or a carriage return. For instance:\
*1.2 0.8* means that the first lora will have a 1.2 multiplier and the second one will have 0.8. 

Alternatively for each Lora's multiplier you may specify a list of float numbers multipliers  separated by a "," (no space) that gives the evolution of this Lora's multiplier over the steps. For instance let's assume there are 30 denoising steps and the multiplier is *0.9,0.8,0.7* then for the steps ranges 0-9, 10-19 and 20-29 the Lora multiplier will be respectively 0.9, 0.8 and 0.7. 

If multiple Loras are defined, remember that each multiplier associated to different Loras should be separated by a space or a carriage return, so we can specify the evolution of multipliers for multiple Loras. For instance for two Loras (press Shift Return to force a carriage return):

```
0.9,0.8,0.7 
1.2,1.1,1.0
```
You can edit, save or delete Loras presets (combinations of loras with their corresponding multipliers) directly from the gradio Web interface. These presets will save the *comment* part of the prompt that should contain some instructions how to use the corresponding the loras (for instance by specifying a trigger word or providing an example).A comment in the prompt is a line that starts that a #. It will be ignored by the video generator. For instance:

```
# use they keyword ohnvx to trigger the Lora*
A ohnvx is driving a car
```
Each preset, is a file with ".lset" extension stored in the loras directory and can be shared with other users

Last but not least you can pre activate Loras corresponding and prefill a prompt (comments only or full prompt) by specifying a preset when launching the gradio server:
```bash
python gradio_server.py --lora-preset  mylorapreset.lset # where 'mylorapreset.lset' is a preset stored in the 'loras' folder
```

You will find prebuilt Loras on https://civitai.com/ or you will be able to build them with tools such as kohya or onetrainer.


### Give me Speed (Text 2 Video only for the moment) !
If you are a speed addict and are ready to accept some tradeoff on the quality I have added two switches:
- Fast Hunyuan Video enabled by default + Sage Attention + Teacache (an advanced acceleration algorithm x2 the speed for a small quality cost)
```bash
python gradio_server.py --fast
```

- Fast Hunyuan Video enabled by default + Sage Attention + Teacache (an advanced acceleration algorithm x2 the speed for a small quality cost) + Compilation  
```bash
python gradio_server.py --fastest
```
Please note that the first sampling step of the first video generation will take two minutes to perform the compilation. Consecutive generations will be very fast unless you trigger a new compilation by changing the resolution, duration of the video or add / remove loras.

For these two switches to work you will need to install Triton and Sage attention.

As you can change the prompt without causing a recompilation, theses switches work quite well with th *Multiple prompts* and / or *Multiple Generations* options.

With the *--fastest* switch activated **a 1280x720 97 frames video takes with a Lora takes less than 4 minutes to be generated** !


If you are looking for a good tradeoff between speed and quality I suggest you use the official HunyuanVideo model with Sage attention and pytorch compilation. You may as well turn on Teacache which will degrade less the video quality given there are more processing steps. 
```bash
python gradio_server.py --attention sage --compile
```

### Command line parameters for Gradio Server
--profile no : default (4) : no of profile between 1 and 5\
--quantize-transformer bool: (default True) : enable / disable on the fly transformer quantization\
--lora-dir path : Path of directory that contains Loras in diffusers / safetensor format\
--lora-preset preset : name of preset gile (without the extension) to preload
--verbose level : default (1) : level of information between 0 and 2\
--server-port portno : default (7860) : Gradio port no\
--server-name name : default (0.0.0.0) : Gradio server name\
--open-browser : open automatically Browser when launching Gradio Server\
--fast : start the app by loading Fast Hunyuan Video generator (faster but lower quality) + sage attention + teacache x2 
--lock-config : prevent modifying the video engine configuration from the interface\
--multiple-images : allow the users to choose multiple images as different starting points for new videos\ 
--compile : turn on pytorch compilation\
--fastest : shortcut for --fast + --compile\
--attention mode: force attention mode among, sdpa, flash, sage and xformers\
--vae-mode: 0-5, defalt(0) : VAE tiling to be used for latents decoding
--preload no : number in Megabytes to preload partially the diffusion model in VRAM , may offer slight speed gains especially on older hardware. Works only with profile 2 and 4.

### Profiles (for power users only)
You can choose between 5 profiles, these will try to leverage the most your hardware, but have little impact for HunyuanVideo GP:
- HighRAM_HighVRAM  (1):  the fastest well suited for a RTX 3090 / RTX 4090 but consumes much more VRAM, adapted for fast shorter video
- HighRAM_LowVRAM  (2): a bit slower, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos
- LowRAM_HighVRAM  (3): adapted for RTX 3090 / RTX 4090 with limited RAM  but at the cost of VRAM (shorter videos)
- LowRAM_LowVRAM  (4): if you have little VRAM or want to generate longer videos 
- VerylowRAM_LowVRAM  (5): at least 24 GB of RAM and 10 GB of VRAM : if you don't have much it won't be fast but maybe it will work

Profile 2 (High RAM) and 4 (Low RAM)are the most recommended profiles since they are versatile (support for long videos for a slight performance cost).\
However, a safe approach is to start from profile 5 (default profile) and then go down progressively to profile 4 and then to profile 2 as long as the app remains responsive or doesn't trigger any out of memory error.

### Other Models for the GPU Poor
- Wan2GP: https://github.com/deepbeepmeep/Wan2GP :\
Another great 3D Image to Video and Text to Video generator. It can run on very low config as one its models is only 1.5 B parameters

- Hunyuan3D-2GP: https://github.com/deepbeepmeep/Hunyuan3D-2GP :\
A great image to 3D and text to 3D tool by the Tencent team. Thanks to mmgp it can run with less than 6 GB of VRAM

- FluxFillGP: https://github.com/deepbeepmeep/FluxFillGP :\
One of the best inpainting / outpainting tools based on Flux that can run with less than 12 GB of VRAM.

- Cosmos1GP: https://github.com/deepbeepmeep/Cosmos1GP :\
This application include two models: a text to world generator and a image / video to world (probably the best open source image to video generator).

- OminiControlGP: https://github.com/deepbeepmeep/OminiControlGP :\
A Flux derived application very powerful that can be used to transfer an object of your choice in a prompted scene. With mmgp you can run it with only 6 GB of VRAM.

- YuE GP: https://github.com/deepbeepmeep/YuEGP :\
A great song generator (instruments + singer's voice) based on prompted Lyrics and a genre description. Thanks to mmgp you can run it with less than 10 GB of VRAM without waiting forever.
