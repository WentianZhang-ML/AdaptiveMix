## Dataset Preparation

*   **Animal Faces-HQ Cat (AFHQ-Cat)** can be downloaded from https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq
*   **Flickr-Faces-HQ (FFHQ)** can be downloaded from https://github.com/NVlabs/ffhq-dataset

Same as [stylegan2-ada-pytorch Preparing datasets](https://github.com/NVlabs/stylegan2-ada-pytorch#preparing-datasets), We use [dataset_tool.py](./dataset_tool.py) to prepare the downloaded datasets (run `python dataset_tool.py --help` for more information). The datasets will be stored as ZIP archives containing uncompressed PNG files. 

For example, the ZIP archive of *AFHQ-Cat* can be created by:

```
python dataset_tool.py --source=~/downloads/afhq/train/cat --dest=~/datasets/afhqcat.zip
```

## Configuration

The examples given in this repository can be further customized with additional command line options:

*   --aug (Default: noaug) disables ADA.
*   --adaptivemix (Default: false) controls whether to apply AdaptiveMix for GAN's training. 
*   --noise_std (Default: 0.05) indicates the standard deviations of noise introduced in the mix results.

## Acknowledgments

We develope our AdaptiveMix on StyleGAN-V2 based on [stylegan2-ada-pytorch Preparing datasets](https://github.com/NVlabs/stylegan2-ada-pytorch#preparing-datasets) and [DeceiveD](https://github.com/EndlessSora/DeceiveD). Thanks for their significent contribution on this community. 
