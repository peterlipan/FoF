# FoF
[IEEE BIBM 2024] This is the official repository of our paper [**Focus on Focus: Focus-oriented Representation Learning and Multi-view Cross-modal Alignment for Glioma Grading**](https://peterlipan.github.io/data/BIBM24_FoF.pdf)

![framework](/assets/framework.png)

*We playfully titled our paper 'Focus on Focus' to showcase the dual meanings of 'focus'—the act of paying attention and the central site of disease development in pathology. We hope this adds a smile to your reading experience! :) A special shoutout to Yupei for this brilliant little wordplay!*

## Pipeline
1. Create a conda environment using the requirements file.
```bash
conda env create -n env_name -f environment.yaml
conda activate env_name
```
2. Download the TCGA-GBMLGG imaging dataset shared by [PathomicFusion](https://github.com/mahmoodlab/PathomicFusion) from [Google Drive](https://drive.google.com/drive/folders/1swiMrz84V3iuzk8x99vGIBd5FCVncOlf)

3. Modify the parameters in [config/configs.yaml](config/configs.yaml).

4. Execute the training and validation process
```bash
python3 main.py
```

## Illustrations
![fig1](/assets/fig1.png)
![fig2](/assets/fig2.png)
![fig3](/assets/fig3.png)
![fig4](/assets/fig4.png)
The trained FoF framework specifically focuses on regions of critical diagnostic significance, such as microvascular proliferation and pseudopalisading necrosis—hallmark features of Grade IV glioblastoma.

## Citation
```
@article{pan2024focus,
  title={Focus on Focus: Focus-oriented Representation Learning and Multi-view Cross-modal Alignment for Glioma Grading},
  author={Pan, Li and Zhang, Yupei and Yang, Qiushi and Li, Tan and Xing, Xiaohan and Yeung, Maximus CF and Chen, Zhen},
  journal={arXiv preprint arXiv:2408.08527},
  year={2024}
}
```
