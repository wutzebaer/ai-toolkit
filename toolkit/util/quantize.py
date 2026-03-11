import hashlib
import os
from fnmatch import fnmatch
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from optimum.quanto import freeze
from optimum.quanto.quantize import _quantize_submodule
from optimum.quanto.tensor import Optimizer, qtype, qtypes
from safetensors.torch import load_file
from torchao.quantization.quant_api import (
    Float8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
)
from torchao.quantization.quant_api import (
    quantize_ as torchao_quantize_,
)
from tqdm import tqdm

from toolkit.paths import TOOLKIT_ROOT
from toolkit.print import print_acc

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel

# the quantize function in quanto had a bug where it was using exclude instead of include

Q_MODULES = [
    "QLinear",
    "QConv2d",
    "QEmbedding",
    "QBatchNorm2d",
    "QLayerNorm",
    "QConvTranspose2d",
    "QEmbeddingBag",
]

torchao_qtypes = {
    # "int4": Int4WeightOnlyConfig(),
    "uint2": UIntXWeightOnlyConfig(torch.uint2),
    "uint3": UIntXWeightOnlyConfig(torch.uint3),
    "uint4": UIntXWeightOnlyConfig(torch.uint4),
    "uint5": UIntXWeightOnlyConfig(torch.uint5),
    "uint6": UIntXWeightOnlyConfig(torch.uint6),
    "uint7": UIntXWeightOnlyConfig(torch.uint7),
    "uint8": UIntXWeightOnlyConfig(torch.uint8),
    "float8": Float8WeightOnlyConfig(),
}


class aotype:
    def __init__(self, name: str):
        self.name = name
        self.config = torchao_qtypes[name]


def get_qtype(qtype: Union[str, qtype]) -> qtype:
    if qtype in torchao_qtypes:
        return aotype(qtype)
    if isinstance(qtype, str):
        return qtypes[qtype]
    else:
        return qtype


def quantize(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype, aotype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
):
    """Quantize the specified model submodules

    Recursively quantize the submodules of the specified parent model.

    Only modules that have quantized counterparts will be quantized.

    If include patterns are specified, the submodule name must match one of them.

    If exclude patterns are specified, the submodule must not match one of them.

    Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Note: quantization happens in-place and modifies the original model and its descendants.

    Args:
        model (`torch.nn.Module`): the model whose submodules will be quantized.
        weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
        activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
        include (`Optional[Union[str, List[str]]]`):
            Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
        exclude (`Optional[Union[str, List[str]]]`):
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.
    """
    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
    for name, m in model.named_modules():
        if include is not None and not any(
            fnmatch(name, pattern) for pattern in include
        ):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        try:
            # check if m is QLinear or QConv2d
            if m.__class__.__name__ in Q_MODULES:
                continue
            else:
                if isinstance(weights, aotype):
                    torchao_quantize_(m, weights.config)
                else:
                    _quantize_submodule(
                        model,
                        name,
                        m,
                        weights=weights,
                        activations=activations,
                        optimizer=optimizer,
                    )
        except Exception as e:
            print(f"Failed to quantize {name}: {e}")
            # raise e


def _load_quantized_cache(
    model: torch.nn.Module,
    cached_sd: dict,
    quantization_type,
):
    from accelerate import init_empty_weights
    from optimum.quanto.nn import QModuleMixin
    from optimum.quanto.tensor import QBytesTensor

    with init_empty_weights():
        quantize(model, weights=quantization_type)

    for name, module in model.named_modules():
        if not isinstance(module, QModuleMixin):
            continue
        data_key = f"{name}.weight._data"
        scale_key = f"{name}.weight._scale"
        if data_key not in cached_sd or scale_key not in cached_sd:
            continue
        data = cached_sd[data_key]
        scale = cached_sd[scale_key]
        qtensor = QBytesTensor(
            qtype=quantization_type,
            axis=0,
            size=module.weight.shape,
            stride=module.weight.stride(),
            data=data,
            scale=scale,
        )
        module.weight = torch.nn.Parameter(qtensor, requires_grad=False)

    # Load non-quantized parameters (norms, biases, embeddings, etc.)
    non_quant_sd = {
        k: v for k, v in cached_sd.items()
        if '._data' not in k and '._scale' not in k
    }
    model.load_state_dict(non_quant_sd, strict=False, assign=True)


def _get_quantize_cache_path(model_path: str, qtype_name: str, subfolder: Optional[str] = None) -> str:
    key = f"{model_path}|{subfolder or ''}|{qtype_name}"
    hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
    cache_dir = os.path.join(TOOLKIT_ROOT, ".cache", "quantized")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"quantized_{hash_key}.pt")


def try_load_quantized_transformer(
    model_class,
    model_path: str,
    subfolder: Optional[str],
    model_config,
) -> tuple:
    """Try to load a quantized transformer from cache.

    Returns (model, True) on cache hit, (None, False) on cache miss.
    On cache hit the returned model is fully quantized and ready to use.
    """

    if not model_config.quantize:
        return None, False
    if model_config.accuracy_recovery_adapter is not None:
        return None, False

    cache_path = _get_quantize_cache_path(
        model_path, model_config.qtype, subfolder
    )
    if not os.path.exists(cache_path):
        return None, False

    from accelerate import init_empty_weights

    from toolkit.dequantize import patch_dequantization_on_save

    try:
        print_acc("Loading quantized transformer from cache (skipping from_pretrained)")
        config = model_class.load_config(model_path, subfolder=subfolder)
        with init_empty_weights():
            model = model_class.from_config(config)
        cached_sd = torch.load(cache_path, map_location='cpu', weights_only=True)
        quantization_type = get_qtype(model_config.qtype)
        _load_quantized_cache(model, cached_sd, quantization_type)
        del cached_sd
        patch_dequantization_on_save(model)
        return model, True
    except Exception as e:
        print_acc(f"Failed to load quantized cache, falling back to from_pretrained: {e}")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return None, False


def quantize_model(
    base_model: "BaseModel",
    model_to_quantize: torch.nn.Module,
    model_path: Optional[str] = None,
    subfolder: Optional[str] = None,
):
    from toolkit.dequantize import patch_dequantization_on_save

    if not hasattr(base_model, "get_transformer_block_names"):
        raise ValueError(
            "The model to quantize must have a method `get_transformer_block_names`."
        )

    # patch the state dict method
    patch_dequantization_on_save(model_to_quantize)

    if base_model.model_config.accuracy_recovery_adapter is not None:
        from toolkit.config_modules import NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork

        # we need to load and quantize with an accuracy recovery adapter
        # todo handle hf repos
        load_lora_path = base_model.model_config.accuracy_recovery_adapter

        if not os.path.exists(load_lora_path):
            # not local file, grab from the hub

            path_split = load_lora_path.split("/")
            if len(path_split) > 3:
                raise ValueError(
                    "The accuracy recovery adapter path must be a local path or for a hf repo, 'username/repo_name/filename.safetensors'."
                )
            repo_id = f"{path_split[0]}/{path_split[1]}"
            print_acc(f"Grabbing lora from the hub: {load_lora_path}")
            new_lora_path = hf_hub_download(
                repo_id,
                filename=path_split[-1],
            )
            # replace the path
            load_lora_path = new_lora_path

        # build the lora config based on the lora weights
        lora_state_dict = load_file(load_lora_path)
        
        if hasattr(base_model, "convert_lora_weights_before_load"):
            lora_state_dict = base_model.convert_lora_weights_before_load(lora_state_dict)
        
        network_config = {
            "type": "lora",
            "network_kwargs": {"only_if_contains": []},
            "transformer_only": False,
        }
        first_key = list(lora_state_dict.keys())[0]
        first_weight = lora_state_dict[first_key]
        # if it starts with lycoris and includes lokr
        if first_key.startswith("lycoris") and any(
            "lokr" in key for key in lora_state_dict.keys()
        ):
            network_config["type"] = "lokr"
        
        network_kwargs = {}

        # find firse loraA weight
        if network_config["type"] == "lora":
            linear_dim = None
            for key, value in lora_state_dict.items():
                if "lora_A" in key:
                    linear_dim = int(value.shape[0])
                    break
            linear_alpha = linear_dim
            network_config["linear"] = linear_dim
            network_config["linear_alpha"] = linear_alpha

            # we build the keys to match every key
            only_if_contains = []
            for key in lora_state_dict.keys():
                contains_key = key.split(".lora_")[0]
                if contains_key not in only_if_contains:
                    only_if_contains.append(contains_key)

            network_kwargs["only_if_contains"] = only_if_contains
        elif network_config["type"] == "lokr":
            # find the factor
            largest_factor = 0
            for key, value in lora_state_dict.items():
                if "lokr_w1" in key:
                    factor = int(value.shape[0])
                    if factor > largest_factor:
                        largest_factor = factor
            network_config["lokr_full_rank"] = True
            network_config["lokr_factor"] = largest_factor

            only_if_contains = []
            for key in lora_state_dict.keys():
                if "lokr_w1" in key:
                    contains_key = key.split(".lokr_w1")[0]
                    contains_key = contains_key.replace("lycoris_", "")
                    if contains_key not in only_if_contains:
                        only_if_contains.append(contains_key)
            network_kwargs["only_if_contains"] = only_if_contains
        
        if hasattr(base_model, 'target_lora_modules'):
            network_kwargs['target_lin_modules'] = base_model.target_lora_modules

        # todo auto grab these
        # get dim and scale
        network_config = NetworkConfig(**network_config)

        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=model_to_quantize,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            # conv_lora_dim=self.network_config.conv,
            # conv_alpha=self.network_config.conv_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=base_model.is_transformer,
            base_model=base_model,
            is_ara=True,
            **network_kwargs
        )
        network.apply_to(
            None, model_to_quantize, apply_text_encoder=False, apply_unet=True
        )
        network.force_to(base_model.device_torch, dtype=base_model.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)
        network.eval()
        network.is_active = True
        network.can_merge_in = False
        base_model.accuracy_recovery_adapter = network

        # quantize it
        lora_exclude_modules = []
        quantization_type = get_qtype(base_model.model_config.qtype)
        for lora_module in tqdm(network.unet_loras, desc="Attaching quantization"):
            # the lora has already hijacked the original module
            orig_module = lora_module.org_module[0]
            orig_module.to(base_model.torch_dtype)
            # make the params not require gradients
            for param in orig_module.parameters():
                param.requires_grad = False
            quantize(orig_module, weights=quantization_type)
            freeze(orig_module)
            module_name = lora_module.lora_name.replace('$$', '.').replace('transformer.', '')
            lora_exclude_modules.append(module_name)
            if base_model.model_config.low_vram:
                # move it back to cpu
                orig_module.to("cpu")
        pass
        # quantize additional layers
        print_acc(" - quantizing additional layers")
        quantization_type = get_qtype('uint8')
        quantize(
            model_to_quantize,
            weights=quantization_type,
            exclude=lora_exclude_modules
        )
    else:
        # quantize model the original way without an accuracy recovery adapter
        cache_path = _get_quantize_cache_path(
            model_path or base_model.model_config.name_or_path_original,
            base_model.model_config.qtype,
            subfolder,
        )
        # move and quantize only certain pieces at a time.
        quantization_type = get_qtype(base_model.model_config.qtype)
        # all_blocks = list(model_to_quantize.transformer_blocks)
        all_blocks: List[torch.nn.Module] = []
        transformer_block_names = base_model.get_transformer_block_names()
        for name in transformer_block_names:
            block_list = getattr(model_to_quantize, name, None)
            if block_list is not None:
                all_blocks += list(block_list)
        base_model.print_and_status_update(
            f" - quantizing {len(all_blocks)} transformer blocks"
        )
        for block in tqdm(all_blocks):
            block.to(base_model.device_torch, dtype=base_model.torch_dtype, non_blocking=True)
            quantize(block, weights=quantization_type)
            freeze(block)
            block.to("cpu", non_blocking=True)

        # todo, on extras find a universal way to quantize them on device and move them back to their original
        # device without having to move the transformer blocks to the device first
        base_model.print_and_status_update(" - quantizing extras")
        # model_to_quantize.to(base_model.device_torch, dtype=base_model.torch_dtype)
        quantize(model_to_quantize, weights=quantization_type)
        freeze(model_to_quantize)

        base_model.print_and_status_update(" - caching quantized model for future runs")
        raw_sd = model_to_quantize.orig_state_dict()
        torch.save(raw_sd, cache_path)
        del raw_sd
