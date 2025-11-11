import torch
from diffusers import StableDiffusionXLPipeline as HFSDXLPipeline

class SDXLPipeline:
    def __init__(self, pipe, device):
        self.pipe = pipe.to(device)
        self.device = device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, torch_dtype=None, variant=None, use_safetensors=True, model=None, device="cuda:0", **kwargs):
        if model is not None:
            pipe = HFSDXLPipeline(
                vae=model.vae,
                text_encoder=model.text_encoder,
                text_encoder_2=model.text_encoder_2,
                tokenizer=model.tokenizer,
                tokenizer_2=model.tokenizer_2,
                unet=model.unet,
                scheduler=model.noise_scheduler,
                feature_extractor=None,
                image_encoder=None,
            )
            if torch_dtype is not None:
                pipe.to(device, dtype=getattr(torch, str(torch_dtype)))
            else:
                pipe.to(device, dtype=model.weight_dtype)
            return cls(pipe, device)
        pipe = HFSDXLPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=getattr(torch, str(torch_dtype)) if torch_dtype else None,
            variant=variant,
            use_safetensors=use_safetensors,
        )
        return cls(pipe, device)

    @torch.inference_mode()
    def generate(self, prompts, num_images_per_prompt=1, num_inference_steps=30, guidance_scale=5.0, height=1024, width=1024, negative_prompt=None, seed=None):
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(int(seed))
        out = self.pipe(
            prompt=prompts,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=g,
        )
        return out.images
