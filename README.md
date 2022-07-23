# Privacy Preservation for Re-ID

Learnable Privacy-Preserving Anonymization for Pedestrian Images  [PDF](https://doi.org/10.1145/3503161.3548766)
Junwu Zhang, Mang Ye, Yao Yang
ACM MM, 2022

## Highlights

1. A reversible visual anonymization framework for Re-ID on hybrid images composed of raw and protected images.
2. A progressive training strategy, namely supervision upgradation, which improves Re-ID performance while retaining enough unrecognizability.

## Qualitative Results

![image-20220723162644991](C:\Users\junwu\AppData\Roaming\Typora\typora-user-images\image-20220723162644991.png)

![image-20220723162508012](C:\Users\junwu\AppData\Roaming\Typora\typora-user-images\image-20220723162508012.png)

## Quantitative Results & Pretrained Models

Pretrained models will be published later.

#### Market-1501 Dataset



#### MSMT17  Dataset



#### CUHK03  Dataset



## Quick Start

### 1. Setup

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.7.10**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/whuzjw/privacy-reid.git
cd privacy-reid
pip install -r requirements.txt
```

</details>

<details open>
<summary>Prepare Datasets</summary>

- Download Market1501, MSMT17 and CUHK03 datasets from http://www.liangzheng.org/Project/project_reid.html to <path_to_root_of_datasets>.
- Extract dataset and rename to `market1501`. The data structure would like:

```
<path_to_root_of_datasets/market1501>
    market1501 
        bounding_box_test/
        bounding_box_train/
        query/
```

- Split Market1501 training set into a smaller training set (80%) and a validation set (20%). Furthermore, the validation set is split into a validation gallery set (80%) and a validation query set (20%).

```bash
python3 data/train_split.py --dataset_root_dir '<path_to_root_of_datasets>' --dataset_name 'market1501'
```

MSMT17 can be split by changing 'market1501' to  'msmt17', CUHK03 can be split using Matlab.

The data structure would like:

```
<path_to_root_of_datasets/market1501_val>
    market1501 
        bounding_box_test/
        bounding_box_train/
        query/
        bounding_box_val_gallery/
        bounding_box_val_query/
```

</details>

### 2. Train

<details open>
<summary>Baseline</summary>

```bash
python3 tools/main.py --config_file='configs/AGW_baseline.yml' DATASETS.ROOT_DIR "<path_to_root_of_datasets>" DATASETS.NAMES "('market1501')" INPUT.TYPE "mosaic" INPUT.RADIUS "24.0" OUTPUT_DIR "('./log/market1501/market1501-base-mosaic')" MODEL.MODE "C"
```

Replace <path_to_root_of_datasets> by your path to the root of datasets, e.g., /data/Dataset.

For other datasets, replace 'market1501' by 'msmt17' or 'cuhk03'.

For other desensitization methods,  INPUT.TYPE and INPUT.RADIUS can be changed from mosaic 24.0×24.0 to blur 12.0×12.0 or noise 0.5.

</details>

<details open>
<summary>Our Model w/o Supervision Upgradation</summary>

```bash
python3 tools/main.py --config_file='configs/AGW_baseline.yml' DATASETS.ROOT_DIR "<path_to_root_of_datasets>" DATASETS.NAMES "('market1501')" INPUT.TYPE "mosaic" INPUT.RADIUS "24.0" OUTPUT_DIR "('./log/market1501/market1501-wosu-mosaic')"
```

</details>

<details open>
<summary>Our Full Model</summary>

```bash
python3 tools/main.py --config_file='configs/AGW_baseline.yml' DATASETS.ROOT_DIR "<path_to_root_of_datasets>" DATASETS.NAMES "('market1501_val')" INPUT.TYPE "mosaic" INPUT.RADIUS "24.0" OUTPUT_DIR "('./log/market1501/market1501-full-mosaic')" MODEL.VAL_R1 "<R1_wosu-mosaic>"
```

Replace <R1_mosaic-wo> by the rank-1 accuracy of a trained market1501-wosu-mosaic model under the setting of raw images as query and mosaic images as gallery.

</details>

### 3. Test

The testing command is just the incremental version of the training command, which is prefixed with the training command and suffixed with TEST.EVALUATE_ONLY, MODEL.PRETRAIN_CHOICE, and MODEL.PRETRAIN_DIR. 

<details open>
<summary>Baseline</summary>

```bash
python3 tools/main.py --config_file='configs/AGW_baseline.yml' DATASETS.ROOT_DIR "<path_to_root_of_datasets>" DATASETS.NAMES "('market1501')" INPUT.TYPE "mosaic" INPUT.RADIUS "24.0" OUTPUT_DIR "('./log/market1501/test/market1501-base-mosaic')" MODEL.MODE "C" TEST.EVALUATE_ONLY "('on')" MODEL.PRETRAIN_CHOICE "('self')" MODEL.PRETRAIN_DIR "./log/market1501/market1501-base-mosaic"
```

</details>

<details open>
<summary>Our Model w/o Supervision Upgradation</summary>

```bash
python3 tools/main.py --config_file='configs/AGW_baseline.yml' DATASETS.ROOT_DIR "<path_to_root_of_datasets>" DATASETS.NAMES "('market1501')" INPUT.TYPE "mosaic" INPUT.RADIUS "24.0" OUTPUT_DIR "('./log/market1501/test/market1501-wosu-mosaic')" TEST.EVALUATE_ONLY "('on')" MODEL.PRETRAIN_CHOICE "('self')" MODEL.PRETRAIN_DIR "./log/market1501/market1501-wosu-mosaic"
```

</details>

<details open>
<summary>Our Model w/o Supervision Upgradation</summary>

```bash
python3 tools/main.py --config_file='configs/AGW_baseline.yml' DATASETS.ROOT_DIR "<path_to_root_of_datasets>" DATASETS.NAMES "('market1501')" INPUT.TYPE "mosaic" INPUT.RADIUS "24.0" OUTPUT_DIR "('./log/market1501/test/market1501-full-mosaic')" TEST.EVALUATE_ONLY "('on')" MODEL.PRETRAIN_CHOICE "('self')" MODEL.PRETRAIN_DIR "./log/market1501/market1501-full-mosaic"
```

</details>

## Citation

Please kindly cite this paper in your publications if it helps your research:

```
@article{acmmm21reidprivacy,
  title={Learnable Privacy-Preserving Anonymization for Pedestrian Images},
  author={Junwu, Zhang and Mang, Ye and Yao, Yang},
  journal={ACM MM},
  year={2022},
}
```

Contact: [whuzjw1@gmail.com](mailto:whuzjw1@gmail.com)
