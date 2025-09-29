# interpretable_evaluation_metrics

To run the experiments, use following command line
```
$ nohup sh run.sh &
```

## Experiments

### Datasets

- **FFHQ**

Head to [FFHQ Dataset Repository](https://github.com/NVlabs/ffhq-dataset), follow the instruction to download 
1024x1024 face images. To note is that `ffhq-dataset-v2.json` needs to be downloaded and put in the same 
directory manually. The original dataset will be downloaded in subfolders, use script `ffhq_50k.py` to create 
a folder containing 50k images for evaluation. 


- **CelebA**

Official website for [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to download the dataset directly from 
[google drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg).
Rename the folders after downloading:
```
$ mv img_align_celeba Img
```

- **ImageNet**

Use the download links from [official website](https://www.image-net.org/index.php) or command lines below:
```
$ wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
$ tar -xvf ILSVRC2012_img_train.tar
$ wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
$ tar -xvf ILSVRC2012_img_val.tar
$ wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
$ tar -xvf ILSVRC2012_img_test_v10102019.tar
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