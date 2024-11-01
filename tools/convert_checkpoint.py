import torch 

# specify your checkpoint path here 
old_ckpt_path = 'VideoTuna/checkpoints/dynamicrafter/i2v_576x1024/model.ckpt'
new_ckpt_path = 'VideoTuna/checkpoints/dynamicrafter/i2v_576x1024/model_converted.ckpt'
pl_sd = torch.load(old_ckpt_path, map_location="cpu")


print(pl_sd['state_dict'].keys())
# with open('dc_i2v_576x1024.txt', 'w') as f_open:
#     for k in pl_sd['state_dict'].keys():
#         f_open.write(k + '\t\n')

for k in list(pl_sd['state_dict'].keys()):
    if 'model' not in k and 'scale_arr' not in k:
        pl_sd['state_dict']['diffusion_scheduler.'+k] = pl_sd['state_dict'].pop(k)

torch.save(pl_sd, new_ckpt_path)
print(f'New checkpoint saved at {new_ckpt_path}')


