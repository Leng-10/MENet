# [Multimodal Cross Enhanced Fusion Network for Diagnosis of Alzheimer's Disease and Subjective Memory Complaints](https://doi.org/10.36227/techrxiv.21354870.v2)
---

![[Y0$H]{1]_~3GCG8TFW]@{%V.png]]

![[B(COS(5{3LL2X(7}Y_GTP~4.png]]

![[Pasted image 20230313091059.png|400]]



## Requirement
### Environment
torch | torchvision | monai | pytorch_metric_learning | numpy | xlwt | time | tqdm

### Dataprepare: [[ADNI]] with the following folder structure:
```
│ADNI/  
├──MRI/  
│  ├── AD  
│  │   ├── train  
│  │   │   ├── wADNI_002_S_0619_brainmask.nii  
│  │   │   ├── wADNI_002_S_0816_brainmask.nii
│  │   │   ├── ......  
│  │   ├── test  
│  │   │   ├── ......  
│  ├── NC  
│  │   ├── train 
│  │   ├── test  
│  ├── ......  
├──PET/  
│  │   ├── train  
│  │   │   ├── wmeanADNI_002_S_0619_brainmask.nii  
│  │   │   ├── wmeanADNI_002_S_0816_brainmask.nii
│  │   │   ├── ......  
│  │   ├── test  
│  │   │   ├── ......  
│  ├── NC  
│  │   ├── train 
│  │   ├── test  
│  ├── ......  
```
## Model
```bash
__all__ = [  
    'ConvMix_1',  # 单模convmix dim1024 patch7 depth5
    'Net_v2',  # 双模convmix dim1024 patch7 depth5
    'ConvMix_MRF_1',  # 单模convmix中的模块换成多尺度模块MRF
    'ConvMix_MRF_CAG_1', # 单模convmix中的模块换成多尺度注意力模块MRF+CAG
    'ConvMix_SWE_CWE_2',  # 双模convmix后接SWE+CWE
    'ConvMix_MRF_CAG_SWE_CWE_2',  # 双模convmix模块替换成MRF+CAG后接SWE+CWE
]

net=ConvMix_MRF_CAG_SWE_CWE_2(patch=5,depth=5)
```
## Train & Test 

train.py
test when test_acc>80






## BibTeX  
  
    @article{lengmultimodal,
    title={Multimodal Cross Enhanced Fusion Network with Multiscale Long-range Reception for Diagnosis of Subjective Memory Complaints},
    author={Leng, Yilin and Cui, Wenju and Peng, Yunsong and Yan, Caiying and Cao, Yuzhu and Yan, Zhuangzhi and Chen, Shuangqing and Jiang, Xi and Zheng, Jian and others}
    year={2023}
    }