# Environment decisions

Checked on 2026-09-08 (Asia/Taipei).

- Official nnUNet master/PyPI: 2.8.1, Python >=3.10, torch >=2.1.2 excluding 2.9.*.
- Selected Python 3.11 for established Windows scientific-wheel support.
- Selected official torch 2.8.0 + torchvision 0.23.0 CUDA 12.8 wheels, compatible with the RTX 5080.
- NVIDIA driver 591.86 reports CUDA compatibility 13.1; installed PyTorch runtime is 12.8.
- CUDA tensor allocation and reduction on RTX 5080 passed; this is not a training benchmark.
- nnUNet Linux is the primary target, with Windows also officially supported.

Sources:

- https://github.com/MIC-DKFZ/nnUNet/blob/master/pyproject.toml
- https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/getting-started/installation-and-setup.md
- https://pypi.org/project/nnunetv2/2.8.1/
- https://pytorch.org/get-started/previous-versions/#v2-8-0

Official upstream metadata and installation instructions are saved alongside this file.
