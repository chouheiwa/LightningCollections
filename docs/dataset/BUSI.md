# BUSI Dataset (Breast Ultrasound Images Dataset)

The original dataset is in
the [BUSI dataset](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset) and you can download
it from the link. After downloading the dataset, you need to unzip the dataset. The dataset structure is as follows:

## Dataset Structure

```
├── BUSI_all
│   ├── train
│   │   ├── images
│   │   └── masks
│   └── val
│       ├── images
│       └── masks
├── BUSI_bad
│   ├── train
│   │   ├── images
│   │   └── masks
│   └── val
│       ├── images
│       └── masks
├── BUSI_benign
│   ├── train
│   │   ├── images
│   │   └── masks
│   └── val
│       ├── images
│       └── masks
└── BUSI_maligant
│   ├── train
│   │   ├── images
│   │   └── masks
│   └── val
│       ├── images
│       └── masks
└── BUSI_normal
    ├── train
    │   ├── images
    │   └── masks
    └── val
        ├── images
        └── masks
```

The BUSI_all is the original dataset, others is the dataset that we split the original dataset into 3 classes with `bad`, `benign`, `maligant`.

The subfolder `train` and `val` are the training and validation dataset. The `images` folder contains the images, and the `masks` folder contains the masks.

You can manually split the dataset by any-fold. We also provide our split logic [script file](preprocess_images.py).

And you can run the script by the following command:

```bash
python docs/dataset/preprocess_images.py --input_path /path/to/BUSI_original_input --output_path /path/to/output
```

And then you can find the dataset with the target structure in the output path.