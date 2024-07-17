import os
import pathlib

import torch, torchvision
from torch.nn import functional as F

from .models import VQVAE, build_vae_var


class VarSamplingOptimizer:
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    model_depths = {16, 20, 24, 30}
    vae_ckpt_file_name = 'vae_ch160v4096z32.pth'
    build_vae_var_kwargs = dict(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        num_classes=1000, shared_aln=False,
    )

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    def __init__(self, model_depth:int, device:torch.device, models_dir:str='model_data'):
        self.device = device

        self.model_depth = model_depth
        self.models_dir  = pathlib.Path(models_dir)
        self._download_hf_models_if_not_exists()

        self.vae, self.var = build_vae_var(
            device=device, patch_nums=self.patch_nums,
            depth=model_depth, **self.build_vae_var_kwargs
        )

        self._init_vae_var_weights()

    @property 
    def var_ckpt_file_name(self):
        assert self.model_depth in self.model_depths, f'Only support model_depth in {self.model_depths}'
        return f'var_d{self.model_depth}.pth'

    def _download_hf_models_if_not_exists(self):
        for ckpt_file_name in (self.vae_ckpt_file_name, self.var_ckpt_file_name):
            ckpt_file_path = self.models_dir / ckpt_file_name

            if not ckpt_file_path.is_file():
                if not self.models_dir.is_dir():
                    os.makedirs(self.models_dir)

                ckpt_url = f'{self.hf_home}/{ckpt_file_name}'
                print(f'Downloading {ckpt_url} to {ckpt_file_path}')
                torch.hub.download_url_to_file(ckpt_url, ckpt_file_path.resolve())

    def _init_vae_var_weights(self):
        self.vae.load_state_dict(torch.load(self.models_dir / self.vae_ckpt_file_name, map_location='cpu'), strict=True)
        self.var.load_state_dict(torch.load(self.models_dir / self.var_ckpt_file_name, map_location='cpu'), strict=True)
        
        self.vae.eval()
        self.var.eval()

        for p in self.vae.parameters(): p.requires_grad_(False)
        for p in self.var.parameters(): p.requires_grad_(False)

        for b in self.var.blocks: b.attn.kv_caching(False)


    def downsample_residual(self, residual, patch_size):
        if patch_size == self.patch_nums[-1]:
            return residual # no need to downsample

        return F.interpolate(residual, size=(patch_size, patch_size), mode='area')
    
    def get_optimization_loss(self, f, predicted_map_size_index, label_B, batch_size=8, cfg=4, top_k=900, top_p=0.95, scale_by_inv_area=False, seed=None):
        f_hat_map_size_index = predicted_map_size_index - 1
        predicted_map_size = self.patch_nums[predicted_map_size_index]

        rng = None
        if seed is not None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(seed)

        with torch.no_grad():
            quant_pyramid, f_hat = self.vae.quantize.f_to_quant_pyramid_and_f_hat(f, self.patch_nums, f_hat_map_size_index) 
            predicted_residual = self.var.predict_single_step_from_quant_pyramid(quant_pyramid=quant_pyramid, label_B=label_B, cfg=cfg, batch_size=batch_size, top_k=top_k, top_p=top_p, rng=rng)

        real_residual = f - f_hat

        loss = F.mse_loss(
            self.downsample_residual(predicted_residual, predicted_map_size), 
            self.downsample_residual(real_residual, predicted_map_size)
        )

        if scale_by_inv_area: loss /= predicted_map_size * predicted_map_size

        return loss
