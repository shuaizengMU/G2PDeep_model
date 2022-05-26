#!/bin/bash
DATASET_ROOT="./public_data"
SOYNAM_DATASET_DIR=$DATASET_ROOT"/SoyNAM"
SOY50K_DATASET_DIR=$DATASET_ROOT"/Soy50K"

echo "Dataset root: "$DATASET_ROOT


####### SoyNAM dataset
echo "SoyNAM data: "$SOYNAM_DATASET_DIR
# height
wget -N -P $SOYNAM_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/SoyNAM/height.test.csv
wget -N -P $SOYNAM_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/SoyNAM/height.train.csv

# moisture
wget -N -P $SOYNAM_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/SoyNAM/moisture.test.csv
wget -N -P $SOYNAM_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/SoyNAM/moisture.train.csv

# oil
wget -N -P $SOYNAM_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/SoyNAM/oil.test.csv
wget -N -P $SOYNAM_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/SoyNAM/oil.train.csv

# protein
wget -N -P $SOYNAM_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/SoyNAM/protein.test.csv
wget -N -P $SOYNAM_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/SoyNAM/protein.train.csv

# yield
wget -N -P $SOYNAM_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/SoyNAM/yield.test.csv
wget -N -P $SOYNAM_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/SoyNAM/yield.train.csv


####### Soy50K dataset
echo "Soy50K data: "$SOY50K_DATASET_DIR
# oil_50k
wget -N -P $SOY50K_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/Soy50K/oil_50k.test.csv
wget -N -P $SOY50K_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/Soy50K/oil_50k.train.csv

# protein_50k
wget -N -P $SOY50K_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/Soy50K/protein_50k.test.csv
wget -N -P $SOY50K_DATASET_DIR https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v1.data/Soy50K/protein_50k.train.csv


####### TCGA dataset 

# kfold (zip file)
wget -N -P $DATASET_ROOT https://data.cyverse.org/dav-anon/iplant/home/soykb/G2PDeep/v2.data/TCGA.tar.gz

