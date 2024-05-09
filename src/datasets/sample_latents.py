import os

import torch
import numpy as np
import clip

from src.models.stylegan2.model import Generator
from src.utils import seed_everything
from src.models.delta_edit.utils import encoder_latent, decoder


def generate_codes(
    generator,
    save_dir,
    device,
    mean_latent=None,
    samples=200_000,
    batch_size=16,
    truncation=1
):
    model, preprocess = clip.load("ViT-B/32", device=device)
    avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32)
    upsample = torch.nn.Upsample(scale_factor=7)

    ind = 0
    with torch.no_grad():
        generator.eval()

        # Generate image by sampling input noises
        w_latents_list = []
        s_latents_list = []
        c_latents_list = []
        for start in range(0, samples, batch_size):
            end = min(start + batch_size, samples)
            batch_sz = end - start
            print(f'current_num:{start}')
            sample_z = torch.randn(batch_sz, 512, device=device)

            sample, w_latents = generator([sample_z], truncation=truncation, truncation_latent=mean_latent,return_latents=True)
            style_space, noise = encoder_latent(generator, w_latents)
            s_latents = torch.cat(style_space, dim=1)

            tmp_imgs = decoder(generator, style_space, w_latents, noise)
            # for s in tmp_imgs:
            #     save_image_pytorch(s, f'{save_dir}/{str(ind).zfill(6)}.png')
            #     ind += 1

            img_gen_for_clip = upsample(tmp_imgs)
            img_gen_for_clip = avg_pool(img_gen_for_clip)
            c_latents = model.encode_image(img_gen_for_clip)

            w_latents_list.append(w_latents)
            s_latents_list.append(s_latents)
            c_latents_list.append(c_latents)
        w_all_latents = torch.cat(w_latents_list, dim=0)
        s_all_latents = torch.cat(s_latents_list, dim=0)
        c_all_latents = torch.cat(c_latents_list, dim=0)

        print(w_all_latents.size())
        print(s_all_latents.size())
        print(c_all_latents.size())

        w_all_latents = w_all_latents.cpu().numpy()
        s_all_latents = s_all_latents.cpu().numpy()
        c_all_latents = c_all_latents.cpu().numpy()

        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/wspace_noise_feat.npy", w_all_latents)
        np.save(f"{save_dir}/sspace_noise_feat.npy", s_all_latents)
        np.save(f"{save_dir}/cspace_noise_feat.npy", c_all_latents)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    seed = 42
    seed_everything(42)
    
    generator = Generator(size=1024, style_dim=512, n_mlp=8).to(device)
    stylegan2_ckpt = '/workspace/saved/stylegan2/stylegan2-ffhq-config-f.pt'
    ckpt = torch.load(stylegan2_ckpt)
    generator.load_state_dict(ckpt['g_ema'], strict=False)
    generator.eval()
    generator = generator.to(device)
    
    generate_codes(generator, '/workspace/data/', device, mean_latent=None)
