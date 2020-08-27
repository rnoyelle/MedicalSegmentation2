# MedicalSegmentation

PyTorch Implementation of [MedicalSegmentation](https://github.com/rnoyelle/MedicalSegmentation)
. A project realised during a 6 months internship at IUCT Oncopole, France.

This provides some deep Learning tools for automatic detection and segmentation of medical images (PET & CT scan).

### Model
Models used during this project are deep learning model like [U-Net](https://arxiv.org/abs/1505.04597). 

Implemented model :

- [x] [3D U-Net](https://arxiv.org/abs/1606.06650)
- [x] [V-Net](https://arxiv.org/abs/1606.04797)
- [x] [DenseX-Net](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8946601).


### Segmentation of tumour on PET/CT scan

**Results :**

<p align="center">
<img style="display: block; margin: auto;" alt="photo" src="images/GIF_example_segmentation.gif">
</p>

## Training a Model 

### Training 3D model (V-Net)
```
python3 training_3d_cnn.py config/default_config.json
```

**Expected results:**
- Dice score :

### Training 2D model DenseX-Net :
```
python3 training_2d_cnn.py config/default_config_2d.json 
```

**Expected results:**
- Dice score :

### Explore data
- To generate 2D MIP :
> `python3 generate_pdf_raw_data.py`




