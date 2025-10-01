# interpretable_evaluation_metrics

To run the experiments, use following command line
```
$ nohup sh run.sh &
```

## Datasets Preparation

- **FFHQ**

Head to [FFHQ Dataset Repository](https://github.com/NVlabs/ffhq-dataset), follow the instruction to download 
1024x1024 face images. To note is that `ffhq-dataset-v2.json` needs to be downloaded and put in the same 
directory manually. The original dataset will be downloaded in subfolders, use script `ffhq_50k.py` to create 
a folder containing 50k images for evaluation. 


- **CelebA**

Official website for [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to download the dataset directly from 
[Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg). Remember to delete the first row of `list_attr_celeba.txt`, so it contains only the table of attributes.
Rename the image folder after downloading:
```
$ mv img_align_celeba Img
```

- **ImageNet**

Use the download links from [official website](https://www.image-net.org/index.php) or bash below:

```shell
# 1. Download training set (~138 GB)
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

# Extract outer tar (contains 1000 class tarballs)
mkdir train && tar -xvf ILSVRC2012_img_train.tar -C train

# Extract each class tarball into its own folder
cd train
for f in *.tar; do
    d=$(basename "$f" .tar)
    mkdir "$d"
    tar -xvf "$f" -C "$d"
    rm "$f"   # optional: remove tarball after extraction
done

cd ..
# 2. Download validation set (~6.3 GB)
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
mkdir val && tar -xvf ILSVRC2012_img_val.tar -C val

# 3. Download test set (~7 GB, labels not public)
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
mkdir test && tar -xvf ILSVRC2012_img_test_v10102019.tar -C test
```

- **COCO**

Download directly from [COCO website](https://cocodataset.org/#download) or use the following commands:

```
# Images
$ wget http://images.cocodataset.org/zips/train2014.zip
$ wget http://images.cocodataset.org/zips/val2014.zip
$ wget http://images.cocodataset.org/zips/test2014.zip

# Annotations
$ wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
$ wget http://images.cocodataset.org/annotations/image_info_test2014.zip

# Extract zip files
$ unzip train2014.zip
$ unzip val2014.zip
$ unzip test2014.zip
$ unzip annotations_trainval2014.zip
$ unzip image_info_test2014.zip
```
- **Furniture15K**
The furniture dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/udaysankarmukherjee/furniture-image-dataset?resource=download).


## Experiments

### 1) Experiments Repetition

#### Multiclass Classification on COCO

#### Injection Experiments
- Create `FFHQ_50k` and `FFHQ_1k` datasets with `./dataset_process/ffhq_50k.py`
- Translation FFHQ dataset with [DiffuseIT](https://github.com/cyclomon/DiffuseIT). 
Remember to download checkpoints [imagenet 256x256](https://drive.google.com/file/d/1kfCPMZLaAcpoIcvzTHwVVJ_qDetH-Rns/view) 
and [FFHQ 256x256](https://drive.google.com/file/d/1-oY7JjRtET4QP3PIWg3ilxAo4VfjCa3J/view), put them in `./checkpoints`. 
Download [pre-trained ArcFace](https://drive.google.com/file/d/1SJa5qVNM6jGZdmsnUsGNhjtrssGYuJfT/view), put it in `./id_model`.
Copy `translate_ffhq_diffuseIT.sh` to DiffuseIT folder. Translate with following code:
```
$ sh translate_ffhq_diffuseIT.sh "Make Up"
```
- Create mixed datasets with script `./dataset_process/prepare_injection_datasets.py`

#### Dataset Comparison