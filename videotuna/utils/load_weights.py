import copy
import logging
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from safetensors import safe_open

from videotuna.utils.common_utils import instantiate_from_config

# from lvdm.personalization.lora import net_load_lora

mainlogger = logging.getLogger("mainlogger")


def expand_conv_kernel(pretrained_dict):
    """expand 2d conv parameters from 4D -> 5D"""
    for k, v in pretrained_dict.items():
        if v.dim() == 4 and not k.startswith("first_stage_model"):
            v = v.unsqueeze(2)
            pretrained_dict[k] = v
    return pretrained_dict


def load_from_pretrainedSD_checkpoint(
    model, pretained_ckpt, expand_to_3d=True, adapt_keyname=False
):
    mainlogger.info(
        "------------------- Load pretrained SD weights -------------------"
    )
    sd_state_dict = torch.load(pretained_ckpt, map_location="cpu")
    if "state_dict" in list(sd_state_dict.keys()):
        sd_state_dict = sd_state_dict["state_dict"]
    model_state_dict = model.state_dict()
    ## delete ema_weights just for <precise param counting>
    for k in list(sd_state_dict.keys()):
        if k.startswith("model_ema"):
            del sd_state_dict[k]
    mainlogger.info(
        f"Num of parameters of target model: {len(model_state_dict.keys())}"
    )
    mainlogger.info(f"Num of parameters of source model: {len(sd_state_dict.keys())}")

    if adapt_keyname:
        ## adapting to standard 2d network: modify the key name because of the add of temporal-attention
        mapping_dict = {
            "middle_block.2": "middle_block.3",
            "output_blocks.5.2": "output_blocks.5.3",
            "output_blocks.8.2": "output_blocks.8.3",
        }
        cnt = 0
        for k in list(sd_state_dict.keys()):
            for src_word, dst_word in mapping_dict.items():
                if src_word in k:
                    new_key = k.replace(src_word, dst_word)
                    sd_state_dict[new_key] = sd_state_dict[k]
                    del sd_state_dict[k]
                    cnt += 1
        mainlogger.info(f"[renamed {cnt} source keys to match target model]")

    pretrained_dict = {
        k: v for k, v in sd_state_dict.items() if k in model_state_dict
    }  # drop extra keys
    empty_paras = [
        k for k, v in model_state_dict.items() if k not in pretrained_dict
    ]  # log no pretrained keys
    mainlogger.info(f"Pretrained parameters: {len(pretrained_dict.keys())}")
    mainlogger.info(f"Empty parameters: {len(empty_paras)} ")
    assert len(empty_paras) + len(pretrained_dict.keys()) == len(
        model_state_dict.keys()
    )

    if expand_to_3d:
        ## adapting to 2d inflated network
        pretrained_dict = expand_conv_kernel(pretrained_dict)

    # overwrite entries in the existing state dict
    model_state_dict.update(pretrained_dict)

    # load the new state dict
    try:
        model.load_state_dict(model_state_dict)
    except Exception:
        state_dict = torch.load(model_state_dict, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        model_state_dict = model.state_dict()
        ## for layer with channel changed (e.g. GEN 1's conditon-concatenating setting)
        for n, p in model_state_dict.items():
            if p.shape != state_dict[n].shape:
                mainlogger.info(f"Skipped parameter [{n}] from pretrained! ")
                state_dict.pop(n)
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)

    mainlogger.info("---------------------------- Finish! ----------------------------")
    return model, empty_paras


## Below: written by Yingqing --------------------------------------------------------


def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        mainlogger.info("missing keys:")
        mainlogger.info(m)
    if len(u) > 0 and verbose:
        mainlogger.info("unexpected keys:")
        mainlogger.info(u)

    model.eval()
    return model


def init_and_load_ldm_model(config_path, ckpt_path, device=None):
    assert config_path.endswith(".yaml"), f"config_path = {config_path}"
    assert ckpt_path.endswith(".ckpt"), f"ckpt_path = {ckpt_path}"
    config = OmegaConf.load(
        config_path
    )  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, ckpt_path)  # TODO: check path
    if device is not None:
        model = model.to(device)
    return model


def load_img_model_to_video_model(
    model,
    device=None,
    expand_to_3d=True,
    adapt_keyname=False,
    config_path="configs/latent-diffusion/txt2img-1p4B-eval.yaml",
    ckpt_path="models/ldm/text2img-large/model.ckpt",
):
    pretrained_ldm = init_and_load_ldm_model(config_path, ckpt_path, device)
    model, empty_paras = load_partial_weights(
        model,
        pretrained_ldm.state_dict(),
        expand_to_3d=expand_to_3d,
        adapt_keyname=adapt_keyname,
    )
    return model, empty_paras


def load_partial_weights(
    model, pretrained_dict, expand_to_3d=True, adapt_keyname=False
):
    model2 = copy.deepcopy(model)
    model_dict = model.state_dict()
    model_dict_ori = copy.deepcopy(model_dict)

    mainlogger.info(
        "-------------- Load pretrained LDM weights --------------------------"
    )
    mainlogger.info(f"Num of parameters of target model: {len(model_dict.keys())}")
    mainlogger.info(f"Num of parameters of source model: {len(pretrained_dict.keys())}")

    if adapt_keyname:
        ## adapting to menghan's standard 2d network: modify the key name because of the add of temporal-attention
        mapping_dict = {
            "middle_block.2": "middle_block.3",
            "output_blocks.5.2": "output_blocks.5.3",
            "output_blocks.8.2": "output_blocks.8.3",
        }
        cnt = 0
        newpretrained_dict = copy.deepcopy(pretrained_dict)
        for k, v in newpretrained_dict.items():
            for src_word, dst_word in mapping_dict.items():
                if src_word in k:
                    new_key = k.replace(src_word, dst_word)
                    pretrained_dict[new_key] = v
                    pretrained_dict.pop(k)
                    cnt += 1
        mainlogger.info(f"--renamed {cnt} source keys to match target model.")
    """
    print('==================not matched parames:')
    for k, v in pretrained_dict.items():
        if k not in model_dict:
            print(k, v.shape)
    """
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in model_dict
    }  # drop extra keys
    empty_paras = [
        k for k, v in model_dict.items() if k not in pretrained_dict
    ]  # log no pretrained keys
    mainlogger.info(f"Pretrained parameters: {len(pretrained_dict.keys())}")
    mainlogger.info(f"Empty parameters: {len(empty_paras)} ")
    # disable info
    # mainlogger.info(f'Empty parameters: {empty_paras} ')
    assert len(empty_paras) + len(pretrained_dict.keys()) == len(model_dict.keys())

    if expand_to_3d:
        ## adapting to yingqing's 2d inflation network
        pretrained_dict = expand_conv_kernel(pretrained_dict)

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # load the new state dict
    try:
        model2.load_state_dict(model_dict)
    except Exception:
        # if parameter size mismatch, skip them
        skipped = []
        for n, p in model_dict.items():
            if p.shape != model_dict_ori[n].shape:
                model_dict[n] = model_dict_ori[n]  # skip by using original empty paras
                mainlogger.info(
                    f"Skip para: {n}, size={pretrained_dict[n].shape} in pretrained, \
                {model_dict[n].shape} in current model"
                )
                skipped.append(n)
        mainlogger.info(
            f"[INFO] Skip {len(skipped)} parameters becasuse of size mismatch!"
        )
        # mainlogger.info(f"[INFO] Skipped parameters: {skipped}")
        model2.load_state_dict(model_dict)
        empty_paras += skipped
        mainlogger.info(f"Empty parameters: {len(empty_paras)} ")
        # import pdb;pdb.set_trace()

    mainlogger.info("-------------- Finish! --------------------------")
    return model2, empty_paras


def load_autoencoder(model, config_path=None, ckpt_path=None, device=None):
    if config_path is None:
        config_path = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    if ckpt_path is None:
        ckpt_path = "models/ldm/text2img-large/model.ckpt"
    # if device is None:
    #     device = torch.device(f"cuda:{dist.get_rank()}") if torch.cuda.is_available() else torch.device("cpu")

    pretrained_ldm = init_and_load_ldm_model(config_path, ckpt_path, device)
    autoencoder_dict = {}
    # import pdb;pdb.set_trace()
    # mainlogger.info([n for n in pretrained_ldm.state_dict().keys()])
    # mainlogger.info([n for n in model.state_dict().keys()])
    for n, p in pretrained_ldm.state_dict().items():
        if n.startswith("first_stage_model"):
            autoencoder_dict[n] = p
    model_dict = model.state_dict()
    model_dict.update(autoencoder_dict)
    mainlogger.info(f"Load [{len(autoencoder_dict)}] autoencoder parameters!")

    model.load_state_dict(model_dict)

    return model


# ----------------------------------------------------------------
def load_safetensors(path):
    assert path.endswith(".safetensors")
    state_dict = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


# ----------------------------------------------------------------
def lora_lora_safetensors(path):
    assert path.endswith(".safetensors")
    state_dict = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    is_lora = all("lora" in k for k in state_dict.keys())
    print(f"is_lora = {is_lora}")

    return state_dict


def convert_lora(
    model,
    state_dict,
    LORA_PREFIX_UNET="lora_unet",
    LORA_PREFIX_TEXT_ENCODER="lora_te",
    alpha=0.6,
):
    # load base model
    # pipeline = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float32)

    # load LoRA weight from .safetensors
    # state_dict = load_file(checkpoint_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "embedding_manager" in key:
            continue
            # currently not supoprt embedding_manager
            # TO SOLVE

        print(f"key={key}")
        if "text" in key and LORA_PREFIX_TEXT_ENCODER in key:
            # layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            layer_infos = ["cond_stage_model", "transformer"] + key.split(".")[0].split(
                LORA_PREFIX_TEXT_ENCODER + "_"
            )[-1].split("_")
            curr_layer = model
            # if isinstance(model.cond_stage_model, FrozenOpenCLIPEmbedder):
            #     curr_layer = model.cond_stage_model.model
        elif LORA_PREFIX_UNET in key:
            # else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = model.diffusion_model
        else:
            # normal sd layers
            layer_infos = key.split(".")[:-1]
            curr_layer = model
        print(f"layer_infos={layer_infos}")

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
        # print(f'curr_layer={curr_layer.__name__}')
        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        elif "lora_up" in key:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))
        else:
            continue

        # update weight
        print(f"lora pair_keys={pair_keys}")
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = (
                state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            )
            curr_layer.weight.data += alpha * torch.mm(
                weight_up, weight_down
            ).unsqueeze(2).unsqueeze(3).to(curr_layer.weight.data.device)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).to(
                curr_layer.weight.data.device
            )

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return model


def check_diffuser_ckpt(sd):
    for k, v in sd.items():
        if "cond_stage_model.model.transformer.resblocks" in k:
            return True
    return False


def change_sd_weight(
    model,
    sd_sd,
    time_varied_inversion_v2=False,
):
    model_sd = model.state_dict()
    device = model_sd[list(model_sd.keys())[0]].device
    is_diffuser = check_diffuser_ckpt(sd_sd)
    for k, v in sd_sd.items():
        if "cond_stage_model.model.transformer.text_model.embeddings" in k:
            continue
            # TODO
        if k == "model_ema.decay" or k == "model_ema.num_updates":  # unused
            continue
        if "diffusion_model.middle_block.2." in k:
            k = k.replace(
                "middle_block.2", "middle_block.3"
            )  # 2 is temporal transformer, 3 is resblock
        if "diffusion_model.output_blocks.5.2.conv" in k:
            k = k.replace("output_blocks.5.2.conv", "output_blocks.5.3.conv")
        if "diffusion_model.output_blocks.8.2.conv" in k:
            k = k.replace("output_blocks.8.2.conv", "output_blocks.8.3.conv")

        if k not in model_sd:
            import pdb

            pdb.set_trace()

        # merge new token
        if (
            time_varied_inversion_v2
            and k == "cond_stage_model.model.token_embedding.weight"
        ):
            n_token = sd_sd["cond_stage_model.model.token_embedding.weight"].shape[0]
            v = torch.cat([sd_sd[k].to(device), model_sd[k][n_token:]], dim=0)
            # v = model_sd[k]
        model_sd[k] = v
    model.load_state_dict(model_sd)
    return model


def load_sd_state_dict(sd_ckpt):
    if sd_ckpt.endswith("safetensors"):
        sd_sd = lora_lora_safetensors(sd_ckpt)
    else:
        sd_sd = torch.load(sd_ckpt, map_location="cpu")
        if "state_dict" in sd_sd:
            sd_sd = sd_sd["state_dict"]
        elif "module" in sd_sd:
            # deepspeed sd ckpt
            new_pl_sd = OrderedDict()
            for key in sd_sd["module"].keys():
                new_pl_sd[key[16:]] = sd_sd["module"][key]
            sd_sd = new_pl_sd
    return sd_sd


def load_model_checkpoint_t2v(
    model,
    ckpt,
    lora_path=None,
    lora_scale=1.0,
    lora_path_2=None,
    lora_scale_2=1.0,
    time_varied_inversion_v2=False,
    strict=True,
    sd_ckpt=None,
):
    """is also compatible with t2i. e.g. sd2-1"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    is_deepspeed = False
    if "state_dict" in pl_sd:
        pl_sd = pl_sd["state_dict"]
    elif "module" in pl_sd:
        pl_sd = pl_sd["module"]
        is_deepspeed = True

    if "model_ema.decay" in pl_sd:  # unused keys in sd
        pl_sd.pop("model_ema.decay")
    if "model_ema.num_updates" in pl_sd:  # unused keys in sd
        pl_sd.pop("model_ema.num_updates")

    if time_varied_inversion_v2:
        merge_new_token = True
        load_lora_text_encoder = True
    else:
        merge_new_token = False
        load_lora_text_encoder = False
    load_lora_text_encoder = False

    if is_deepspeed:
        new_pl_sd = OrderedDict()
        for key in pl_sd.keys():
            new_pl_sd[key[16:]] = pl_sd[key]
        pl_sd = new_pl_sd

    # merge_new_token
    if merge_new_token:
        if lora_path is not None:
            lora_sd = torch.load(lora_path, map_location="cpu")
            sd_lora_and_adapter_lora = False
            if (
                sd_lora_and_adapter_lora
            ):  # tmp code.  combine a sd lora with a adapter lora
                token_emb = None
            else:
                token_emb = lora_sd["cond_stage_model.model.token_embedding.weight"]
            pl_sd["cond_stage_model.token_table.weight"] = lora_sd[
                "cond_stage_model.token_table.weight"
            ]
            del lora_sd
        elif sd_ckpt is not None:
            sd_sd = load_sd_state_dict(sd_ckpt)
            token_emb = sd_sd["cond_stage_model.model.token_embedding.weight"][49408:]
        else:
            import pdb

            pdb.set_trace()
        if token_emb is not None:
            pl_sd["cond_stage_model.model.token_embedding.weight"] = torch.cat(
                [pl_sd["cond_stage_model.model.token_embedding.weight"], token_emb],
                dim=0,
            )

    if "scale_arr" in pl_sd:
        print("scale_arr", pl_sd["scale_arr"])
        pl_sd.pop("scale_arr")
    if "cond_stage_model.token_embedding.weight" in pl_sd:  # TODO
        pl_sd.pop("cond_stage_model.token_embedding.weight")

    model.load_state_dict(pl_sd, strict=False)
    print(">>> model checkpoint loaded.")

    # replace base sd weight
    if sd_ckpt is not None:
        if not merge_new_token:
            sd_sd = load_sd_state_dict(sd_ckpt)

        model = change_sd_weight(
            model,
            sd_sd,
            time_varied_inversion_v2=time_varied_inversion_v2,
        )
        del sd_sd

    # insert lora weght
    if lora_path is not None:
        net_load_lora(
            model,
            lora_path,
            alpha=lora_scale,
            load_text_encoder=load_lora_text_encoder,
        )
        print("@lora checkpoint loaded.")
        if lora_path_2 is not None:
            net_load_lora(
                model,
                lora_path_2,
                alpha=lora_scale_2,
                load_text_encoder=load_lora_text_encoder,
            )
            print("@lora 2 checkpoint loaded.")

    return model


def load_model_checkpoint_t2v_adapter(model, ckpt, adapter_ckpt=None, **kwargs):
    strict = False if adapter_ckpt else True
    model = load_model_checkpoint_t2v(model, ckpt, strict=strict, **kwargs)

    if adapter_ckpt:
        sd = torch.load(adapter_ckpt, map_location="cpu")
        sd = sd["state_dict"] if "state_dict" in sd else sd
        model.adapter.load_state_dict(sd, strict=True)
        print(">>> adapter checkpoint loaded.")

    return model
