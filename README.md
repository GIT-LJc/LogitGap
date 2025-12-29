# Revisiting Logit Distributions for Reliable Out-of-Distribution Detection

This is an implementation of [LogitGap](https://openreview.net/forum?id=FLdLPUqnsP) in the Annual Conference on Neural Information Processing Systems (NeurIPS 2025).

## Abstract
Out-of-distribution (OOD) detection is critical for ensuring the reliability of deep learning models in open-world applications. While post-hoc methods are favored for their efficiency and ease of deployment, existing approaches often underexploit the rich information embedded in the model’s logits space. In this paper, we propose LogitGap, a novel post-hoc OOD detection method that explicitly exploits the relationship between the maximum logit and the remaining logits to enhance the separability between in-distribution (ID) and OOD samples. To further improve its effectiveness, we refine LogitGap by focusing on a more compact and informative subset of the logit space. Specifically, we introduce a training-free strategy that automatically identifies the most informative logits for scoring. We provide both theoretical analysis and empirical evidence to validate the effectiveness of our approach. Extensive experiments on both vision-language and vision-only models demonstrate that LogitGap consistently achieves state-of-the-art performance across diverse OOD detection scenarios and benchmarks.


## Environments

Our experiments are conducted on NVIDIA GeForce RTX-4090Ti GPUs with Python 3.8 and Pytorch 2.3.1+cu121. Besides, the following commonly used packages are required to be installed:
- [transformers](https://huggingface.co/docs/transformers/installation), scipy, scikit-learn, matplotlib, seaborn, pandas, tqdm. 

## Data Preparation
We suggest putting all datasets under the same folder to ease management and following the instructions below to organize datasets to avoid modifying the source code. The default dataset location is `./datasets`. The file structure looks like:

```
LogitGap
|-- datasets
    |-- ImageNet
    |-- ImageNet10
    |-- ImageNet20
    |-- ImageNet100
    |-- ImageNet_OOD_dataset
        |-- ImageNetOOD
        |-- NINCO
        |-- ImageNet-A
        |-- iNaturalist
    ...
```

## In-distribution Datasets
We consider the following ID datasets:
ImageNet-1k, ImageNet-10, ImageNet-20, and ImageNet-100
### ImageNet-1k (ILSVRC-2012)
- Download ImageNet-1k from [here](https://image-net.org/challenges/LSVRC/2012/index.php#).
- Extract the dataset to `./datasets/ImageNet`. 

### ImageNet Subsets 
ImageNet-10, ImageNet-20, and ImageNet-100 can be generated given the class names and IDs provided in `data/ImageNet10/ImageNet-10-classlist.csv` , `data/ImageNet20/ImageNet-20-classlist.csv`, and `data/ImageNet100/class_list.txt` respectively.

To create ImageNet-10, 20, and 100, the following script can be used:

```python
cd utils
# ImageNet-10 
python create_imagenet_subset.py --in_dataset ImageNet10
# ImageNet-20
python create_imagenet_subset.py --in_dataset ImageNet20
# ImageNet-100
python create_imagenet_subset.py --in_dataset ImageNet100
```

## Out-of-Distribution Datasets

### Semantic Shift
This dataset consists of three parts: NINCO, ImageNetOOD, and ImageNet-O. The file structure looks like
```
|–– ImageNet_OOD_dataset
    |–– NINCO
        |–– NINCO_OOD_classes
    |–– ImageNetOOD
        |–– images 
    |–– ImageNet-O
        |–– n01443537
        |–– ...
```

#### NINCO
- Download the dataset from https://zenodo.org/record/8013288/files/NINCO_all.tar.gz?download=1.
- Extract the dataset to `./datasets/ImageNet_OOD_dataset/NINCO`.

#### ImageNetOOD
- Download the dataset from https://image-net.org/data/imagenetood.tar.gz (refer to https://github.com/princetonvisualai/imagenetood). Extract it to `./datasets/ImageNet_OOD_dataset/ImageNetOOD`.

#### ImageNet-O
- Download the dataset from https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar and extract it to `./datasets/ImageNet_OOD_dataset/ImageNet-O`.

### Covariate Shift
This dataset consists of three parts: ImageNet-Sketch, ImageNet-A, and ImageNet-R. The file structure looks like

```
|–– ImageNet_OOD_dataset
    |–– ImageNet-A
        |–– n01498041
        |–– ...
    |–– ImageNet-R
        |–– n01443537
        |–– ...
    |–– ImageNet-Sketch
        |–– n01440764
        |–– ...
```
#### ImageNet-Sketch
- Download the dataset from https://github.com/HaohanWang/ImageNet-Sketch.
- Extract the dataset to `./datasets/ImageNet_OOD_dataset/ImageNet-Sketch`.

#### ImageNet-A
- Download the dataset from https://github.com/hendrycks/natural-adv-examples and extract it to `./datasets/ImageNet_OOD_dataset/ImageNet-A`.

#### ImageNet-R
- Download the dataset from https://github.com/hendrycks/imagenet-r and extract it to `./datasets/ImageNet_OOD_dataset/ImageNet-R`.

## How to Run

We provide the running scripts in `scripts`, which allow you to reproduce the results on the NeurIPS'25 paper.

For example, to evaluate the performance of LogitGap score on ImageNet-1k, with the following :

```
bash scripts/eval_imagenet.sh
```




## Citation
If you use this code in your research, please kindly cite the following paper:

```
@inproceedings{liang2025revisiting,
    title={Revisiting Logit Distributions for Reliable Out-of-Distribution Detection},
    author={Liang, Jiachen and Hou, RuiBing and Hu, Minyang and Chang, Hong and Shan, Shiguang and Chen, Xilin},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2025},
}
```
