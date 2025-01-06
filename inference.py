from tqdm import tqdm
import numpy as np
import torch
from einops import repeat
import cv2
import jsonlines
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Union
import os
import glob
from PIL import Image

import diffuser
from diffuser.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffuser import StableDiffusionInpaintPipeline,UNet2DConditionModel
from diffuser.image_processor import PipelineImageInput, VaeImageProcessor
from transformers import CLIPTextModel, CLIPImageProcessor
from diffuser.utils.torch_utils import randn_tensor
from diffuser.models import AsymmetricAutoencoderKL
import inspect
from scipy.stats import gaussian_kde

rand_seed = 717


def main(classname,brk):
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "checkpoints/inpainting/", 
            torch_dtype=torch.float32,
            local_files_only=True,
            safety_checker = None,
            requires_safety_checker = False
    )
    
    pipeline.load_textual_inversion('output/bs6-lr10-6-mse-only%s_init_emb_%s_%s/'%(classname,classname,brk), 
                                    token="%s_%s"%(classname,brk), 
                                    use_safetensors=True,
                                    text_encoder=pipeline.text_encoder,
                                    tokenizer=pipeline.tokenizer,
                                    weight_name='model.safetensors'
                                    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipeline = pipeline.to(device)


    generator = torch.Generator(device=device)
    generator = generator.manual_seed(rand_seed)
    ds = 20
    gs = 3.5
    w,h = 512, 512
    c = 0
    
    with jsonlines.open('dataset/%s/script/test_%s.jsonl'%(classname,brk), 'r') as reader:
        for line in reader:
            os.makedirs('generate/%s/%s/image/'%(classname,brk), exist_ok=True)

            image = line['good']
            mask = line['mask']
            prompt = classname+'_'+brk
            image = Image.open(image)
            image = image.resize((w, h))
            mask = Image.open(mask)
            mask = mask.resize((w, h))
            
            img_add = Image.blend(image.convert("RGBA"), mask.convert("RGBA"), 0.3)
            image_merge=np.array(img_add.convert('RGB'))

            result = pipeline(
                        prompt=prompt,
                        image=image,
                        mask_image=mask,
                        num_inference_steps=ds,
                        guidance_scale=gs,
                        generator=generator,
                        num_images_per_prompt=1,
                        path= os.path.join('generate/',line['good'][:-7])
                    )
            images = result.images
            attn_maps = result.attn_maps

            image_merge = np.append(image_merge,np.array(images[0]),axis=1)

            att = torch.zeros((1,1,h,w))
            up = torch.nn.Upsample(size=(h, w),mode='bilinear')
            
            for ats in attn_maps[2:-1]: # (7,64,64) 7层layer
                if len(ats) != 0:
                    ats = torch.stack(ats).cpu()
                    att += up(ats.unsqueeze(0))
            
            attn_maps = att/len(attn_maps[2:-1]) #(1,64,64)
            attn_maps = attn_maps[0].repeat(3,1,1).permute(1,2,0)
            attn = attn_maps.cpu().numpy()
            
            normed_mask = attn / attn.max()
            normed_mask = (normed_mask * 255).astype('uint8')

            print('save image to ', os.path.join('generate/%s/%s/image_mask/'%(classname,brk)))            
            images[0].save(os.path.join('generate/%s/%s/image/'%(classname,brk),'%04d.png'%c))




if __name__ == "__main__":
    obs = [
        'bottle',
        'cable',
        'capsule',
        'carpet',
        'grid',
        'hazelnut',
        'leather',
        'metal_nut',
        'pill',
        'screw',
        'tile',
        'toothbrush',
        'transistor',
        'wood',
        'zipper'
    ]
    for ob in obs:
        brks = os.listdir('dataset/%s/ground_truth/'%ob)
        for brk in brks:
            main(ob,brk)

