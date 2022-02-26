<div align="center">
  <h1>SEGA: Semantic Guided Attention on Visual Prototype for Few-Shot Learning <br> (WACV 2022)</h1>
</div>

<div align="center">
  <h3><a href=https://martayang.github.io/>Fengyuan Yang</a>, <a href=https://vipl.ict.ac.cn/homepage/rpwang/index.htm>Ruiping Wang</a>, <a href=http://people.ucas.ac.cn/~xlchen?language=en>Xilin Chen</a></h3>
</div>

<div align="center">
  <h4> <a href=https://openaccess.thecvf.com/content/WACV2022/papers/Yang_SEGA_Semantic_Guided_Attention_on_Visual_Prototype_for_Few-Shot_Learning_WACV_2022_paper.pdf>[Paper link]</a>, <a href=https://openaccess.thecvf.com/content/WACV2022/supplemental/Yang_SEGA_Semantic_Guided_WACV_2022_supplemental.pdf>[Supp link]</a></h4>
</div>

## 1. Requirements
* Python 3.7
* CUDA 11.2
* PyTorch 1.9.0


## 2. Datasets

* miniImagenet [[Google Drive](https://drive.google.com/file/d/17ZsIOmuZmQkdzwnPVw5f6AJ18jd-8X1S/view?usp=sharing)]
    * Download and extract it in a certain folder, let's say  `/data/FSLDatasets/miniImagenet`, then set `_MINI_IMAGENET_DATASET_DIR` of _data/mini_imagenet.py_ to this folder.
* tieredImageNet [[Google Drive](https://drive.google.com/file/d/1_n2YMRzq7AAaUkjEjCRF17PJWD3tyHdg/view?usp=sharing)]
    * Download and extract it in a certain folder, let's say  `/data/FSLDatasets/tieredImageNet`, then set `_TIERED_IMAGENET_DATASET_DIR` of _data/tiered_imagenet.py_ to this folder.
* CIFAR-FS [[Google Drive](https://drive.google.com/file/d/1ZTT9EjGoYG0bTtt4W3fcWlw4NjwkaXHb/view?usp=sharing)]
    * Download and extract it in a certain folder, let's say  `/data/FSLDatasets/CIFAR-FS`, then set `_CIFAR_FS_DATASET_DIR` of _data/CIFAR_FS.py_ to this folder.
* CUB-FS [[Google Drive](https://drive.google.com/file/d/1hbXAVEXqdE7vTvDJpHMUmh5Ok3UU3EG4/view?usp=sharing)]
    * Download and extract it in a certain folder, let's say  `/data/FSLDatasets/cub`, then set `_CUB_FS_DATASET_DIR` of _data/CUB_FS.py_ to this folder.

Note: the above datasets are the same as previous works (e.g.  [FewShotWithoutForgetting](https://github.com/gidariss/FewShotWithoutForgetting), [DeepEMD](https://github.com/icoz69/DeepEMD)) EXCEPT that we include **additional semantic embeddings** ([GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings for the first 3 datasets and attributes embeddings for CUB-FS). Thus, remember to change the _argparse arguments_ `semantic_path` in training and testing scripts. 

## 3. Usage

Our training and testing scripts are all at `scripts/` and are all in the form of jupyter notebook, where both the _argparse arguments_ and output logs can be easily found.

Let's take training and testing paradigm on miniimagenet for example.
For the 1st stage training, run all cells in `scripts/01_miniimagenet_stage1.ipynb`. And for the 2nd stage training and final testing, run all cells in `scripts/01_miniimagenet_stage2_SEGA_5W1S.ipynb`.

## 4. Results

The 1-shot and 5-shot classification results can be found in the corresponding jupyter notebooks.

## 5. Pre-trained Models

The pre-trained models for all 4 datasets after our first training stage can be downloaded from [here](https://drive.google.com/file/d/1eS49e5Wt5gXnMM7TLso1eS2B04kX8oLh/view?usp=sharing).

## Citation

If you find our paper or codes useful, please consider citing our paper:

```bibtex
@inproceedings{yang2022sega,
  title={SEGA: Semantic Guided Attention on Visual Prototype for Few-Shot Learning},
  author={Yang, Fengyuan and Wang, Ruiping and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1056--1066},
  year={2022}
}
```

## Acknowledgments

Our codes are based on [Dynamic Few-Shot Visual Learning without Forgetting](https://github.com/icoz69/DeepEMD) and [MetaOptNet](https://github.com/kjunelee/MetaOptNet), and we really appreciate it. 

## Further

If you have any question, feel free to contact me. My email is _fengyuan.yang@vipl.ict.ac.cn_