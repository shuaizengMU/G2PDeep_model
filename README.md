## Introduction

Genomic selection uses single-nucleotide polymorphisms (SNPs) to predict quantitative phenotypes for enhancing traits in breeding populations and has been widely used to increase breeding efficiency for plants and animals. Existing statistical methods rely on a prior distribution assumption of imputed genotype effects, which may not fit experimental datasets. Emerging deep learning technology could serve as a powerful machine learning tool to predict quantitative phenotypes without imputation and also to discover potential associated genotype markers efficiently. We propose a deep-learning framework using convolutional neural networks (CNNs) to predict the quantitative traits from SNPs and also to investigate genotype contributions to the trait using saliency maps. 

* G2PDeep Model: https://github.com/shuaizengMU/G2PDeep_model
* G2PDeep Website: https://g2pdeep.org/

## Install

Python 3.6.8
```
pip install -r requirement.txt
```

## Running the program

```
# For SoyNAM data with height quantative phenotype.
python train.py --data_dir ./data/SoyNAM --result_dir ./results --dataset_type height

# For SoyNAM data with oil quantative phenotype.
python train.py --data_dir ./data/SoyNAM --result_dir ./results --dataset_type oil

# For SoyNAM data with moisture quantative phenotype.
python train.py --data_dir ./data/SoyNAM --result_dir ./results --dataset_type moisture

# For SoyNAM data with protein quantative phenotype.
python train.py --data_dir ./data/SoyNAM --result_dir ./results --dataset_type protein

# For SoyNAM data with yield quantative phenotype.
python train.py --data_dir ./data/SoyNAM --result_dir ./results --dataset_type yield
```

## Authors

* **Shuai Zeng** - *University of Missouri, Columbia MO, USA*
* **Email** - *zengs@umsystem.edu* 


## Citation

[1] Zeng, Shuai, Ziting Mao, Yijie Ren, Duolin Wang, Dong Xu, and Trupti Joshi. "G2PDeep: a web-based deep-learning framework for quantitative phenotype prediction and discovery of genomic markers." Nucleic Acids Research 49, no. W1 (2021): W228-W236.

[2] Liu, Yang, Duolin Wang, Fei He, Juexin Wang, Trupti Joshi, and Dong Xu. "Phenotype prediction and genome-wide association study using deep convolutional neural network of soybean." Frontiers in genetics (2019): 1091.

## License

[Apache License 2.0](LICENSE)
