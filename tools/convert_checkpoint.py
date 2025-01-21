import argparse
import os

import torch

"""
This script is used to convert the key of diffusion scheduler to match the format in this repo.
The conversion is as follows:
    betas                                 -->  diffusion_scheduler.betas
    alphas_cumprod                        -->  diffusion_scheduler.alphas_cumprod
    alphas_cumprod_prev                   -->  diffusion_scheduler.alphas_cumprod_prev
    sqrt_alphas_cumprod                   -->  diffusion_scheduler.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod        -->  diffusion_scheduler.sqrt_one_minus_alphas_cumprod
    log_one_minus_alphas_cumprod         -->  diffusion_scheduler.log_one_minus_alphas_cumprod
    sqrt_recip_alphas_cumprod            -->  diffusion_scheduler.sqrt_recip_alphas_cumprod
    sqrt_recipm1_alphas_cumprod          -->  diffusion_scheduler.sqrt_recipm1_alphas_cumprod
    posterior_variance                   -->  diffusion_scheduler.posterior_variance
    posterior_log_variance_clipped       -->  diffusion_scheduler.posterior_log_variance_clipped
    posterior_mean_coef1                 -->  diffusion_scheduler.posterior_mean_coef1
    posterior_mean_coef2                 -->  diffusion_scheduler.posterior_mean_coef2
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    required=True,
    help="Path to the old checkpoint, e.g., checkpoints/dynamicrafter/i2v_576x1024/model.ckpt",
)
parser.add_argument(
    "--save_key",
    action="store_true",
    help="Save the keys of the old and new checkpoints",
)
args = parser.parse_args()
input_path = args.input_path
save_key = args.save_key

output_dir, filename = (
    os.path.dirname(input_path),
    input_path.split("/")[-1].split(".")[0],
)
output_path = os.path.join(output_dir, f"{filename}_converted.ckpt")

pl_sd = torch.load(input_path, map_location="cpu")
if save_key:
    save_txt = os.path.join(output_dir, f"{filename}-statedict.txt")
    with open(save_txt, "w") as f_open:
        for k in pl_sd["state_dict"].keys():
            f_open.write(k + "\t\n")

for k in list(pl_sd["state_dict"].keys()):
    if "model" not in k and "scale_arr" not in k:
        pl_sd["state_dict"]["diffusion_scheduler." + k] = pl_sd["state_dict"].pop(k)

torch.save(pl_sd, output_path)
if save_key:
    save_txt = os.path.join(output_dir, f"{filename}-statedict-converted.txt")
    with open(save_txt, "w") as f_open:
        for k in pl_sd["state_dict"].keys():
            f_open.write(k + "\t\n")

print(f"New checkpoint saved at {output_path}")
