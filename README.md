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
The furniture dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/udaysankarmukherjee/furniture-image-dataset?resource=download). Preprocess the dataset with `./dataset_process/furniture_process.py`.



## Experiments

### 1) Experiments Repetition

#### Multiclass Classification on COCO
- Create `CelebA50K` dataset with script `./dataset_process/celeba_50k.py`
- Run experiments with command:
```
$ python main_celeba_classification.py --img_dir /PATH/TO/CelebA50k --ref_img_dir /PATH/TO/CelebA50k --anno_file /PATH/TO/CelebA50k/Anno --score_type EvalCS_local_vit
```

#### Injection Experiments
- Create `FFHQ_50k` and `FFHQ_1k` datasets with `./dataset_process/ffhq_50k.py`
- Translation FFHQ dataset with [DiffuseIT](https://github.com/cyclomon/DiffuseIT). 
Remember to download checkpoints [imagenet 256x256](https://drive.google.com/file/d/1kfCPMZLaAcpoIcvzTHwVVJ_qDetH-Rns/view) 
and [FFHQ 256x256](https://drive.google.com/file/d/1-oY7JjRtET4QP3PIWg3ilxAo4VfjCa3J/view), put them in `./checkpoints`. 
Download [pre-trained ArcFace](https://drive.google.com/file/d/1SJa5qVNM6jGZdmsnUsGNhjtrssGYuJfT/view), put it in `./id_model`.
Copy `translate_ffhq_diffuseIT.sh` to DiffuseIT folder. Translate with following code, which takes 16h for 1000 figures with A100:
```
$ sh translate_ffhq_diffuseIT.sh "Make Up"
```
- Create mixed datasets with script `./dataset_process/prepare_injection_datasets.py`
```
$ python prepare_injection_datasets.py --trans_root /PATH/TO/DIFFUSEIT/GENERATED/FOLDER
```
- Compute `RA-ClipScore` with script
```
$ python main_injection_exp.py --real_dir /PATH/TO/REAL/DATASET --gen_dir /PATH/TO/MIXED/DATASET
```


#### Dataset Comparison
- Create datasets `A`, `B` and `C` with script `./dataset_process/prepare_datasets_compare.py`
- Compute `RA-ClipScore` with script
```
$ python main_dataset_compare.py --real_dir /PATH/TO/A --gen_dir /PATH/TO/B
```


### 2) Ablation Study
- Create dataset with `./dataset_process/prepare_datasets_ablation.py`
- Plotting figures with `main_ablation_study.py`, more details about the parameters, please read the script.
```
$ python main_ablation_study.py
```

### 3) Single Image (Rethinking HCS)

- Attributes highly influence the evaluation results, to run experiment to obtain scores for the man image, use the following command:
```
$ cd sadpad
$ python main_single.py --img_path man.png
```
- When using two attributes, `HCS` will provide scores that are opposite values even if semantically it has no correlations:
```
$ cd sadpad
$ python main_single.py --img_path woman.png
```

### 4) Attribute-wise Evaluation of SOTA Generative Models (FFHQ)
- Generate `50k` images using SOTA generative models, e.g.
```
# Diffusion Models
$ git clone https://github.com/CompVis/latent-diffusion.git
$ cd latent-diffusion
$ conda env create -f environment.yaml
$ conda activate ldm
$ CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py -r models/ldm/ffhq256/model.ckpt -l ./logs/ffhq_step200 -n 50000 --batch_size 16 -c 200 -e 0.0
$ CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py -r models/ldm/ffhq256/model.ckpt -l ./logs/ffhq_step50 -n 50000 --batch_size 16 -c 50 -e 0.0

# StyleGAN2-ada
$ git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
$ cd stylegan2-ada-pytorch
$ python generate.py --outdir=out --trunc=1 --seeds=0-49999 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

# StyleGAN3
$ git clone https://github.com/NVlabs/stylegan3.git
$ cd stylegan3
$ python gen_images.py --outdir=out --trunc=1 --seeds=0-49999 --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl
```
- Run evaluation with script `calc_metrics.py`


### 5) Spatial Evaluation of SOTA Generative Models (ImageNet)


Rebuttal
-------
CLIP
python profile_inference.py \
  --model "ViT-B/32" \
  --image-batch-size 1 \
  --text-batch-size 1


CLIP-FlanT5
python scripts/profile_inference.py \
  --model-path zhiqiulin/clip-flant5-xl \
  --image-file images/example1.jpg \
  --query "Describe this image." \
  --decoder-length 32 \
  --show-table

BLIP-VQA
python profile_blip_vqa.py --image merlion.png --question "Which city is this photo taken?" --device cuda

MiniGPT4
python eval_scripts/profile_vqa_flops.py --cfg-path eval_configs/minigpt4_eval.yaml --image merlion.png --question "What is shown in this image?" --show-table

mPLUG
python profile_vqa_flops.py   --config configs/vqa_mplug_base.yaml   --device cuda   --batch-size 1   --question-length 25   --answer-length 10   --beam-size 5

ShareGPT4V
python scripts/profile_vqa_flops.py   --model-path ./ShareGPT4V-7B   --image examples/photo.png   --question "What is shown in this image?"
----------
In CLIP folder locally:
 python prompt_direction_similarity.py ./figs_age/old.jpg ./figs_age/old1.jpg ./figs_age/old2.jpg ./figs_age/old3.jpg --positive 'young' --negative 'not young'
 ----------
Stanford40
Classification: Running/Jumping

