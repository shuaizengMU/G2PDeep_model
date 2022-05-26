## Introduction
Genomic selection uses single-nucleotide polymorphisms (SNPs) to predict quantitative phenotypes for enhancing traits in breeding populations and has been widely used to increase breeding efficiency for plants and animals. Existing statistical methods rely on a prior distribution assumption of imputed genotype effects, which may not fit experimental datasets. Emerging deep learning technology could serve as a powerful machine learning tool to predict quantitative phenotypes without imputation and also to discover potential associated genotype markers efficiently. We propose a deep-learning framework using convolutional neural networks (CNNs) to predict the quantitative traits from SNPs and also to investigate genotype contributions to the trait using saliency maps. 

* G2PDeep Model: https://github.com/shuaizengMU/G2PDeep_model
* G2PDeep Website: https://g2pdeep.org/

## Download dataset
Run following shell script to download public dataset we used to train models.
```
sh run_download_public_data.sh
```