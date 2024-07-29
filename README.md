

# Local Latent Space Bayesian Optimization (LOLBO) implementation for JTVAE and VEGFR-2 inhibitors (JTVAE-LOLBO-VEGFR2)

Discovery of VEGFR-2 Inhibitors employing Junction Tree Variational Encoder with Local Latent Space Bayesian Optimization and Gradient Ascent Exploration

![GitHub issues](https://img.shields.io/github/issues/buchijw/JTVAE-lolbo-VEGFR2?style=for-the-badge)
![GitHub pull requests](https://img.shields.io/github/issues-pr/buchijw/JTVAE-lolbo-VEGFR2?style=for-the-badge)
![License](https://img.shields.io/github/license/buchijw/JTVAE-lolbo-VEGFR2?style=for-the-badge)
![Git LFS](https://img.shields.io/badge/GIT%20LFS-8A2BE2?style=for-the-badge)

<!-- TABLE OF CONTENTS -->

<details open>
  <summary><h3>Table of Contents</h3></summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This repository contains library files, data, and scripts for conducting local latent space Bayesian optimization (LOLBO) on Junction Tree Variational Autoencoder using predicted $pIC_{50}$ and $dual$ function values to find new compounds that potentially express Vascular Endothelial Growth Factor Receptor 2 (VEGFR-2) inhibiting activity.

The model is a part of the paper **"Discovery of VEGFR-2 Inhibitors employing Junction Tree Variational Encoder with Local Latent Space Bayesian Optimization and Gradient Ascent Exploration"**

Official Junction Tree Variational Autoencoder belongs to Wengong Jin (wengong@csail.mit.edu), Regina Barzilay, Tommi Jaakkola [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364).

Official Local Latent Space Bayesian Optimization belongs to Natalie Maus (nmaus@seas.upenn.edu), Haydn T. Jones, Juston S. Moore, Matt J. Kusner, John Bradshaw, Jacob R. Gardner [https://arxiv.org/abs/2201.11872](https://arxiv.org/abs/2201.11872).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- REQUIREMENTS -->

## Requirements

This git repo requires Git LFS installed for large files and submodules to provide necessary library for JTVAE and QSAR models. To clone this repo, please run:

```bash
git lfs install
git clone https://github.com/buchijw/JTVAE-lolbo-VEGFR2.git --recursive
```

We tested on Ubuntu 22.04 LTS with ROCm 6.0, therefore the code supposes to work on CUDA/HIP based on type of PyTorch installed.

Conda environment file for ROCm platforms is included as [conda-environment.yaml](conda-environment.yaml). However, due to packages related to QSAR may need to be manually built, we suggest preparing the environment using the provided scripts in [QSAR repo](https://github.com/buchijw/QSAR-VEGFR2).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- QUICK START -->

## Quick Start

The following directories and files contains the implementations of the model:

* [`data/`](data/) contains SMILES, latent vectors, calculated penalized logP, $pIC_{50}$ and $dual$ values of the initial datasets.
* [`JTVAE-GA/`](JTVAE-GA/) is a clone of [JTVAE-GA repo](https://github.com/buchijw/JTVAE-GA) as a submodule to provide library files and model checkpoint of JTVAE.
* [`QSAR-VEGFR2/`](QSAR-VEGFR2/) is a clone of [QSAR repo](https://github.com/buchijw/QSAR-VEGFR2) as a submodule to provide library files and pipeline of QSAR.
* [`lolbo/`](lolbo/) contains `lolbo` library files.
* [`scripts/`](scripts/) contains codes for conducting Bayesian optimization.
* [`scripts/run_optimize.sh`](scripts/run_optimize.sh) is the Bayesian optimization run file, which need to be modified accordingly.

### Task IDs

In addition to the original LOLBO task IDs, we added two new task IDs for our target functions:

| task_id | Full Task Name                  |
|---------|---------------------------------|
|  pic50  | Predicted $pIC_{50}$ using QSAR |
|  dual   | The $dual$ values               |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contributing

- [Gia-Bao Truong](https://github.com/buchijw/)
- Thanh-An Pham
- Van-Thinh To
- Hoang-Son Lai Le
- Phuoc-Chung Van Nguyen
- [The-Chuong Trinh](https://trinhthechuong.github.io)
- [Tieu-Long Phan](https://tieulongphan.github.io/)
- Tuyen Ngoc Truong<sup>*</sup>

<sup>*</sup>Corresponding author

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

This work has received support from the Korea International Cooperation Agency (KOICA) under the project entitled "Education and Research Capacity Building Project at University of Medicine and Pharmacy at Ho Chi Minh City," conducted from 2024 to 2025 (Project No. 2021-00020-3).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
