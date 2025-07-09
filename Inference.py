# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from Decoupled_Cross_Attention_Mechanism.ip_adapter import Resampler

import argparse
import logging
import os
import torch.utils.data as data
import torchvision
import json
import accelerate
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import transformers
import random
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader
from src.unet_hacked_main_UNet import UNet2DConditionModel
from src.unet_hacked_DP_UNet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.DPDEdit_pipeline import StableDiffusionXLInpaintPipeline as EditPipeline


logger = get_logger(__name__, log_level="INFO")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fashion image editing with multimodal inputs.")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--width", type=int, default=768, help="Width of the output image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the output image.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--output_dir", type=str, default="result_pose_n", help="Directory to save the results.")
    parser.add_argument("--unpaired", action="store_true", help="Whether to use unpaired data.")
    parser.add_argument("--data_dir", type=str, default="/home/wxl/data/VITON-HD", help="Directory of the data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--test_batch_size", type=int, default=2, help="Batch size for testing.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale for classifier-free guidance.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Precision type for mixed precision training.")
    return parser.parse_args()


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor."""
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
    return image_tensor


class VitonHDTestDataset(data.Dataset):
    """Dataset for VITON-HD fashion image editing."""

    def __init__(self, dataroot_path: str, phase: Literal["train", "test"], size: Tuple[int, int] = (1024, 512), unpaired: bool = True):
        self.dataroot = dataroot_path
        self.phase = phase
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.to_tensor = transforms.ToTensor()

        # Load annotations
        json_file_path = os.path.join(dataroot_path, phase, "test_garment_texture_caption.json")
        with open(json_file_path, "r") as file:
            data = json.load(file)
        self.annotation_pair = {item["image_id"]: item["caption"] for item in data["annotations"]}

        self.im_names, self.t_names = self.load_filenames()
        self.clip_processor = CLIPImageProcessor()

    def load_filenames(self):
        im_names, t_names = [], []
        filename = os.path.join(self.dataroot, f"{self.phase}_pairs.txt")
        with open(filename, "r") as file:
            for line in file:
                im_name, t_name = line.strip().split()
                if self._files_exist(im_name, t_name):
                    im_names.append(im_name)
                    t_names.append(t_name)
        return im_names, t_names

    def _files_exist(self, im_name, t_name):
        image_path = os.path.join(self.dataroot, self.phase, "image", im_name)
        mask_path = os.path.join(self.dataroot, self.phase, "mask", im_name.replace(".jpg", ".png"))
        texture_paths = self._get_texture_paths(t_name)
        return os.path.exists(image_path) and os.path.exists(mask_path) and any(os.path.exists(path) for path in texture_paths)

    def _get_texture_paths(self, t_name):
        base_name = t_name.replace(".jpg", "")
        return [os.path.join(self.dataroot, self.phase, "texture", f"{base_name}_texture.png")]

    def __getitem__(self, index):
        t_name = self.t_names[index]
        im_name = self.im_names[index]
        texture_annotation = self.annotation_pair.get(im_name, "shirts")
        texture_path = self._select_texture_path(t_name)
        texture = Image.open(texture_path).resize((256, 256))

        image_pil = Image.open(os.path.join(self.dataroot, self.phase, "image", im_name)).resize(self.size)
        image = self.transform(image_pil)

        mask_path = os.path.join(self.dataroot, self.phase, "mask", im_name.replace(".jpg", ".png"))
        mask = Image.open(mask_path).resize(self.size)
        mask = self.to_tensor(mask)[:1]
        mask = 1 - mask
        im_mask = image * mask

        pose_image = Image.open(os.path.join(self.dataroot, self.phase, "image-densepose", im_name))
        pose_image = self.transform(pose_image)

        return {
            "t_name": t_name,
            "im_name": im_name,
            "image": image,
            "texture_pure": self.transform(texture),
            "texture": self.clip_processor(images=texture, return_tensors="pt").pixel_values,
            "inpaint_mask": 1 - mask,
            "im_mask": im_mask,
            "caption": texture_annotation,
            "pose_img": pose_image,
        }

    def _select_texture_path(self, t_name):
        texture_paths = self._get_texture_paths(t_name)
        for path in texture_paths:
            if os.path.exists(path):
                return path
        return texture_paths[-1]

    def __len__(self):
        return len(self.im_names)


def setup_accelerator(args):
    """Set up the accelerator."""
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=accelerator_project_config)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return accelerator


def load_models(args, weight_dtype=torch.float16):
    """Load the required models."""
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet", torch_dtype=weight_dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_path, subfolder="image_encoder", torch_dtype=weight_dtype)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(args.pretrained_model_path, subfolder="unet_encoder", torch_dtype=weight_dtype)
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=weight_dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_path, subfolder="text_encoder_2", torch_dtype=weight_dtype)
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer", revision=None, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer_2", revision=None, use_fast=False)

    return noise_scheduler, vae, unet, image_encoder, unet_encoder, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two


def freeze_models(unet, vae, image_encoder, unet_encoder, text_encoder_one, text_encoder_two):
    """Freeze models for inference."""
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.eval()
    unet_encoder.eval()


def main():
    args = parse_arguments()
    accelerator = setup_accelerator(args)

    weight_dtype = torch.float16

    noise_scheduler, vae, unet, image_encoder, unet_encoder, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two = load_models(args, weight_dtype)

    freeze_models(unet, vae, image_encoder, unet_encoder, text_encoder_one, text_encoder_two)

    test_dataset = VitonHDTestDataset(dataroot_path=args.data_dir, phase="test", unpaired=args.unpaired, size=(args.height, args.width))
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, num_workers=4)

    pipe = EditPipeline.from_pretrained(
        args.pretrained_model_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    pipe.unet_encoder = unet_encoder

    with torch.no_grad():
        for sample in test_dataloader:
            img_emb_list = [sample['texture'][i] for i in range(sample['texture'].shape[0])]
            prompt = sample["caption"]
            num_prompts = sample['texture'].shape[0]
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt
            )

            prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                sample["caption_texture"], num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=negative_prompt
            )

            generator = torch.Generator(pipe.device).manual_seed(args.seed) if args.seed else None
            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                strength=1.0,
                pose_img=sample['pose_img'],
                text_embeds_texture=prompt_embeds_c,
                texture=sample["texture_pure"].to(accelerator.device),
                mask_image=sample['inpaint_mask'],
                image=(sample['image'] + 1.0) / 2.0,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                ip_adapter_image=torch.cat(img_emb_list, dim=0),
            )[0]

            for i, img in enumerate(images):
                x_sample = pil_to_tensor(img)
                torchvision.utils.save_image(x_sample, os.path.join(args.output_dir, sample['im_name'][i]))


if __name__ == "__main__":
    main()


