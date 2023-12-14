# pylint: skip-file
import os
import random

# TORCH_GPU_MEMORY_FRACTION = 0.95  # Target memory ~= 15G on 16G card
TORCH_GPU_MEMORY_FRACTION = 0.4  # Target memory ~= 15G on 40G card
TORCH_GPU_DEVICE_ID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = f"{TORCH_GPU_DEVICE_ID}"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import traceback

from typing import Dict, List, Union

from torchvision import transforms

import io
import time
import json
import base64
from pathlib import Path
from PIL import Image

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(
    TORCH_GPU_MEMORY_FRACTION, 0  # it count of number of device instead of device index
)


import diffusers

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    # Reference: https://docs.nvidia.com/launchpad/data-science/sentiment/latest/sentiment-triton-overview.html

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
        Both keys and values are strings. The dictionary keys and values are:
        * model_config: A JSON string containing the model configuration
        * model_instance_kind: A string containing model instance kind
        * model_instance_device_id: A string containing model instance device ID
        * model_repository: Model repository path
        * model_version: Model version
        * model_name: Model name
        """
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])

        # Load the model

        base_model_path = str(
            Path(__file__).parent.absolute().joinpath("stable-diffusion-xl-base-1.0")
        )
        refiner_model_path = str(
            Path(__file__).parent.absolute().joinpath("stable-diffusion-xl-refiner-1.0")
        )

        self.logger.log_info(f"[DEBUG] load model under path: {base_model_path}")
        self.logger.log_info(f"[DEBUG] load model under path: {refiner_model_path}")

        self.logger.log_info(f"[DEBUG] diffusers version: {diffusers.__version__}")
        self.logger.log_info(f"[DEBUG] torch version: {torch.__version__}")

        self.base = diffusers.DiffusionPipeline.from_pretrained(
            base_model_path,  # "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            device_map="auto",
        ).to("cuda")

        self.refiner = diffusers.DiffusionPipeline.from_pretrained(
            refiner_model_path,  # "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
            output_loading_info=True
            # max_memory={0: "9GB", 1: "9GB"}
            #  max_memory={0: "12GB", 1: "12GB", 2: "12GB", 3: "12GB"},
        ).to("cuda")

        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "IMAGES")
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        # disable_torch_init()
        responses = []

        for request in requests:
            try:
                # binary data typed back to string
                prompt = [
                    t.decode("UTF-8")
                    for t in pb_utils.get_input_tensor_by_name(request, "PROMPT")
                    .as_numpy()
                    .tolist()
                ]

                # prompt = str(pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0].decode("utf-8"))
                self.logger.log_info(
                    f"[DEBUG] input `prompt` type({type(prompt)}): {prompt}"
                )

                negative_prompt = [
                    t.decode("UTF-8")
                    for t in pb_utils.get_input_tensor_by_name(
                        request, "NEGATIVE_PROMPT"
                    )
                    .as_numpy()
                    .tolist()
                ]
                self.logger.log_info(
                    f"[DEBUG] input `negative_prompt` type({type(negative_prompt)}): {negative_prompt}"
                )

                num_images_per_prompt = [
                    t
                    for t in pb_utils.get_input_tensor_by_name(request, "SAMPLES")
                    .as_numpy()
                    .tolist()
                ][0]
                self.logger.log_info(
                    f"[DEBUG] input `num_images_per_prompt` type({type(num_images_per_prompt)}): {num_images_per_prompt}"
                )

                scheduler = [
                    t.decode("UTF-8")
                    for t in pb_utils.get_input_tensor_by_name(request, "SCHEDULER")
                    .as_numpy()
                    .tolist()
                ][0]
                self.logger.log_info(
                    f"[DEBUG] input `scheduler` type({type(scheduler)}): {scheduler}"
                )

                num_inference_steps = [
                    t
                    for t in pb_utils.get_input_tensor_by_name(request, "STEPS")
                    .as_numpy()
                    .tolist()
                ][0]
                self.logger.log_info(
                    f"[DEBUG] input `num_inference_steps` type({type(num_inference_steps)}): {num_inference_steps}"
                )
                guidance_scale = [
                    t
                    for t in pb_utils.get_input_tensor_by_name(
                        request, "GUIDANCE_SCALE"
                    )
                    .as_numpy()
                    .tolist()
                ][0]
                self.logger.log_info(
                    f"[DEBUG] input `guidance_scale` type({type(guidance_scale)}): {guidance_scale}"
                )

                random_seed = [
                    t
                    for t in pb_utils.get_input_tensor_by_name(request, "SEED")
                    .as_numpy()
                    .tolist()
                ][0]
                self.logger.log_info(
                    f"[DEBUG] input `seed` type({type(random_seed)}): {random_seed}"
                )

                # pure text to image in sdxl
                # prompt_image = pb_utils.get_input_tensor_by_name(request, "prompt_image").as_numpy()[0]
                # self.logger.log_info(f'[DEBUG] input `prompt_image` type({type(prompt_image)}): {len(prompt_image)}')

                extra_params_str = ""
                if (
                    pb_utils.get_input_tensor_by_name(request, "extra_params")
                    is not None
                ):
                    extra_params_str = str(
                        pb_utils.get_input_tensor_by_name(request, "extra_params")
                        .as_numpy()[0]
                        .decode("utf-8")
                    )
                self.logger.log_info(
                    f"[DEBUG] input `extra_params` type({type(extra_params_str)}): {extra_params_str}"
                )

                extra_params = {}
                # TODO: Add a function handle penalty
                try:
                    extra_params = json.loads(extra_params_str)
                    if "repetition_penalty" in extra_params:
                        self.logger.log_info(
                            "[DEBUG] WARNING `repetition_penalty` would crash transformerparsing faield!"
                        )
                        del extra_params["repetition_penalty"]
                except json.decoder.JSONDecodeError:
                    self.logger.log_info(
                        "[DEBUG] WARNING `extra_params` parsing faield!"
                    )
                    pass

                # random_seed = 0
                # if pb_utils.get_input_tensor_by_name(request, "random_seed") is not None:
                #     random_seed = int(pb_utils.get_input_tensor_by_name(request, "random_seed").as_numpy()[0])
                # self.logger.log_info(f'[DEBUG] input `random_seed` type({type(random_seed)}): {random_seed}')

                if random_seed > 0:
                    random.seed(random_seed)
                    np.random.seed(random_seed)
                    torch.manual_seed(random_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(random_seed)

                # # Define how many steps and what % of steps to be run on each experts (80/20) here
                # n_steps = 40
                # high_noise_frac = 0.8
                if "high_noise_frac" not in extra_params:
                    extra_params["high_noise_frac"] = 0.8

                t0 = time.time()  # calculate time cost in following function call
                high_noise_frac = 0.8
                # run both experts
                image = self.base(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    denoising_end=high_noise_frac,
                    guidance_scale=guidance_scale,
                    output_type="latent",
                ).images
                image = self.refiner(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    denoising_start=high_noise_frac,
                    guidance_scale=guidance_scale,
                    image=image,
                ).images[0]

                to_tensor_transform = transforms.ToTensor()
                tensor_image = to_tensor_transform(image)
                batch_tensor_image = tensor_image.unsqueeze(0).to("cpu").permute(0, 2, 3, 1)
                torch.cuda.empty_cache()

                print(f"image: type({type(batch_tensor_image)}):")
                print(f"image: shape: {batch_tensor_image.shape}")

                tensor_output = [pb_utils.Tensor("IMAGES", batch_tensor_image.numpy())]
                responses.append(pb_utils.InferenceResponse(tensor_output))

            except Exception as e:
                self.logger.log_info(f"Error generating stream: {e}")
                self.logger.log_info(f"{traceback.format_exc()}")

                # error = pb_utils.TritonError(f"Error generating stream: {e}")
                # triton_output_tensor = pb_utils.Tensor(
                #     "IMAGES", np.asarray(["N/A"], dtype=self.output0_dtype)
                # )
                # response = pb_utils.InferenceResponse(
                #     output_tensors=[triton_output_tensor], error=error
                # )
                # responses.append(response)
                # self.logger.log_info("The model did not receive the expected inputs")
                raise e
            return responses

    def finalize(self):
        self.logger.log_info("Cleaning up ...")
