import os
import pathlib

import torch, torchvision

from .models import VQVAE, build_vae_var


class VarSamplingOptimizer:
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    model_depths = {16, 20, 24, 30}
    vae_ckpt = 'vae_ch160v4096z32.pth'

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    def __init__(self, model_depth:int, device:torch.device, models_dir:str=None):
        self.device = device

        vae_path, var_path = self._get_model_paths(model_depth, models_dir)





        self._init_hf_models(model_depth)

        self.vae, self.var = vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=self.patch_nums,
            num_classes=1000, depth=model_depth, shared_aln=False,
        )

        # load checkpoints
        vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
        var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
        vae.eval(), var.eval()
        for p in vae.parameters(): p.requires_grad_(False)
        for p in var.parameters(): p.requires_grad_(False)


    @classmethod
    def _get_model_paths(cls, model_depth:int, models_dir:str):
        assert model_depth in cls.model_depths, f'Only support model_depth in {cls.model_depths}'

        models_dir = pathlib.Path(models_dir) or pathlib.Path(__file__).parent / 'model_data'
        models_dir.mkdir(parents=True, exist_ok=True)

        vae_path = models_dir / cls.vae_ckpt
        var_path = models_dir / f'var_d{model_depth}.pth'

        return vae_path, var_path


    @classmethod
    def _download_hf_models_if_not_exists(cls, model_depth:int):
        for ckpt in (cls.vae_ckpt, f'var_d{model_depth}.pth'):
            if not osp.exists(ckpt): os.system(f'wget {cls.hf_home}/{ckpt}')