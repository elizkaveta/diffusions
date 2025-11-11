import torch
from torch import nn
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTokenizer


class DiffusionLora(nn.Module):
    def __init__(self, pretrained_model_name_or_path, rank, lora_modules, init_lora_weights, weight_dtype, device, target_size=1024, prediction_type="epsilon", lora_alpha=None):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.target_size = target_size
        self.device = torch.device(device)
        self.lora_rank = rank
        self.lora_alpha = lora_alpha or rank
        self.lora_modules = set(lora_modules or ["unet"])
        self.init_lora_weights = init_lora_weights
        self.prediction_type = prediction_type
        _m = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
        self.weight_dtype = _m.get(str(weight_dtype).lower(), torch.float16)
        pipe = StableDiffusionXLPipeline.from_pretrained(self.pretrained_model_name_or_path, torch_dtype=self.weight_dtype)
        self.tokenizer: CLIPTokenizer = pipe.tokenizer
        self.tokenizer_2: CLIPTokenizer = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.noise_scheduler = pipe.scheduler
        self.vae = pipe.vae
        self.unet = pipe.unet
        del pipe
        attn_procs = {}
        for name, module in self.unet.named_modules():
            if hasattr(module, "set_processor"):
                proc = LoRAAttnProcessor2_0(hidden_size=module.to_q.in_features, cross_attention_dim=getattr(module, "cross_attention_dim", None), rank=self.lora_rank, network_alpha=self.lora_alpha)
                attn_procs[name] = proc
        self.unet.set_attn_processor(attn_procs)
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)

    def prepare_for_training(self):
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(False)
        for p in self.lora_layers.parameters():
            p.requires_grad_(True)
        self.noise_scheduler.config.prediction_type = self.prediction_type
        self.to(self.device, dtype=self.weight_dtype)

    def get_trainable_params(self, config):
        lr = float(getattr(config.optimizer, "lr", 1e-4))
        wd = float(getattr(config.optimizer, "weight_decay", 0.0))
        return [{"params": list(self.lora_layers.parameters()), "lr": lr, "weight_decay": wd, "name": "lora_unet"}]

    def get_state_dict(self):
        return self.lora_layers.state_dict()

    def load_state_dict_(self, state_dict):
        self.lora_layers.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def _encode_text(self, prompts):
        tok_1 = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(self.device)
        tok_2 = self.tokenizer_2(prompts, padding="max_length", max_length=self.tokenizer_2.model_max_length, truncation=True, return_tensors="pt").to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.weight_dtype):
            enc_1 = self.text_encoder(**tok_1, output_hidden_states=True)
            enc_2 = self.text_encoder_2(**tok_2, output_hidden_states=True)
        prompt_embeds = enc_1.hidden_states[-2]
        text_embeds = enc_2.hidden_states[-2]
        pooled_text_embeds = enc_2.pooler_output
        return prompt_embeds, text_embeds, pooled_text_embeds

    @torch.no_grad()
    def _encode_vae(self, pixel_values):
        with torch.autocast(device_type=self.device.type, dtype=self.weight_dtype):
            scale = getattr(self.vae.config, "scaling_factor", 0.18215)
            latents = self.vae.encode(pixel_values).latent_dist.sample() * scale
        return latents

    def _compute_add_time_ids(self, bsz, device, dtype):
        t = torch.tensor([self.target_size, self.target_size, 0, 0, self.target_size, self.target_size], device=device, dtype=dtype)
        return t[None, :].repeat(bsz, 1)

    def forward(self, pixel_values, prompt, do_cfg=False, noise=None, timesteps=None, *args, **kwargs):
        bsz = pixel_values.size(0)
        pixel_values = pixel_values.to(self.device, dtype=self.weight_dtype, non_blocking=True)
        latents = self._encode_vae(pixel_values)
        if noise is None:
            noise = randn_tensor(latents.shape, dtype=latents.dtype, device=latents.device)
        if timesteps is None:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        prompt_embeds, text_embeds, pooled_text_embeds = self._encode_text(prompt)
        add_time_ids = self._compute_add_time_ids(bsz, latents.device, latents.dtype)
        with torch.autocast(device_type=self.device.type, dtype=self.weight_dtype):
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, text_embeds=text_embeds, time_ids=add_time_ids, pooled_projections=pooled_text_embeds).sample
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError("Unknown prediction_type")
        return {"model_pred": model_pred, "target": target}
