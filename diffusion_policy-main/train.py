"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import torch
import os
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary())
    print(device)
    print(torch.cuda.get_device_name())
    print(torch.__version__)
    print(torch.version.cuda)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
    # x = torch.randn(1).cuda()
    # print(x)
    # print(torch.cuda.memory_summary())
    # torch.cuda.empty_cache()
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()
    

if __name__ == "__main__":
    main()
