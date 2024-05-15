# pylint: skip-file
import os

TORCH_GPU_DEVICE_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{TORCH_GPU_DEVICE_ID}"

from torchvision import transforms

import time
import random

import numpy as np
import torch

import diffusers


from instill.helpers.const import DataType, TextToImageInput
from instill.helpers.ray_io import StandardTaskIO

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    construct_infer_response,
    construct_metadata_response,
    Metadata,
)


@instill_deployment
class Sdxl:
    def __init__(self):
        print(f"torch version: {torch.__version__}")
        print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count() : {torch.cuda.device_count()}")
        # print(f"torch.cuda.current_device() : {torch.cuda.current_device()}")
        # print(f"torch.cuda.device(0) : {torch.cuda.device(0)}")
        # print(f"torch.cuda.get_device_name(0) : {torch.cuda.get_device_name(0)}")

        # https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
        # Download through huggingface

        ACCESS_TOKEN = "..."

        self.base = diffusers.DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            device_map="auto",
            token=ACCESS_TOKEN,
        ).to("cuda")

        self.refiner = diffusers.DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
            output_loading_info=True,
            token=ACCESS_TOKEN,
            # max_memory={0: "12GB", 1: "12GB", 2: "12GB", 3: "12GB"},
        ).to("cuda")

    def ModelMetadata(self, req):
        resp = construct_metadata_response(
            req=req,
            inputs=[
                Metadata(
                    name="prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="negative_prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="prompt_image",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="samples",
                    datatype=str(DataType.TYPE_INT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="scheduler",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="steps",
                    datatype=str(DataType.TYPE_INT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="guidance_scale",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[1],
                ),
                Metadata(
                    name="seed",
                    datatype=str(DataType.TYPE_INT64.name),
                    shape=[1],
                ),
                Metadata(
                    name="extra_params",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                Metadata(
                    name="images",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[-1, -1, -1, -1],
                ),
            ],
        )
        return resp

    async def __call__(self, req):
        task_text_to_image_input: TextToImageInput = (
            StandardTaskIO.parse_task_text_to_image_input(request=req)
        )
        print("----------------________")
        print(task_text_to_image_input)
        print("----------------________")

        print("print(task_text_to_image_input.prompt_image)")
        print(type(task_text_to_image_input.prompt_image))
        print("-------\n")

        print("print(task_text_to_image_input.prompt)")
        print(task_text_to_image_input.prompt)
        print("-------\n")

        print("print(task_text_to_image_input.negative_prompt)")
        print(task_text_to_image_input.negative_prompt)
        print("-------\n")

        print("print(task_text_to_image_input.steps)")
        print(task_text_to_image_input.steps)
        print("-------\n")

        print("print(task_text_to_image_input.guidance_scale)")
        print(task_text_to_image_input.guidance_scale)
        print("-------\n")

        print("print(task_text_to_image_input.seed)")
        print(task_text_to_image_input.seed)
        print("-------\n")

        print("print(task_text_to_image_input.samples)")
        print(task_text_to_image_input.samples)
        print("-------\n")

        print("print(task_text_to_image_input.extra_params)")
        print(task_text_to_image_input.extra_params)
        print("-------\n")

        if task_text_to_image_input.seed > 0:
            random.seed(task_text_to_image_input.seed)
            np.random.seed(task_text_to_image_input.seed)

        high_noise_frac = 0.8
        if "high_noise_frac" in task_text_to_image_input.extra_params:
            high_noise_frac = task_text_to_image_input.extra_params["high_noise_frac"]

        num_inference_steps = 40
        if "num_inference_steps" in task_text_to_image_input.extra_params:
            num_inference_steps = task_text_to_image_input.extra_params[
                "num_inference_steps"
            ]

        t0 = time.time()

        image = self.base(
            prompt=task_text_to_image_input.prompt,
            num_inference_steps=num_inference_steps,
            denoising_end=high_noise_frac,
            guidance_scale=task_text_to_image_input.guidance_scale,
            output_type="latent",
        ).images

        image = self.refiner(
            prompt=task_text_to_image_input.prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            guidance_scale=task_text_to_image_input.guidance_scale,
            image=image,
        ).images[0]

        to_tensor_transform = transforms.ToTensor()
        tensor_image = to_tensor_transform(image)
        batch_tensor_image = tensor_image.unsqueeze(0).to("cpu").permute(0, 2, 3, 1)
        torch.cuda.empty_cache()

        print(f"Inference time cost {time.time()-t0}s")

        print(f"image: type({type(batch_tensor_image)}):")
        print(f"image: shape: {batch_tensor_image.shape}")

        # task_output = StandardTaskIO.parse_task_text_generation_output(sequences)
        # task_output = np.asarray(batch_tensor_image).tobytes()
        task_output = batch_tensor_image.numpy().tobytes()

        print("Output:")
        # print(task_output)
        print("type(task_output): ", type(task_output))
        print("batch_tensor_image.numpy().shape:", batch_tensor_image.numpy().shape)
        print("batch_tensor_image.shape: ", batch_tensor_image.shape)

        return construct_infer_response(
            req=req,
            outputs=[
                Metadata(
                    name="images",
                    datatype=str(DataType.TYPE_FP32.name),
                    # shape=[-1, -1, -1, -1],
                    shape=[1, 1024, 1024, 3],
                )
            ],
            raw_outputs=[task_output],
        )


entrypoint = InstillDeployable(Sdxl).get_deployment_handle()
