# UNet

| Code Repository |                  Paper                   | Pretrained Model |
|:---------------:|:----------------------------------------:|:----------------:|
|       N/A       | [Link](https://arxiv.org/abs/1505.04597) |       N/A        |

**UNet** is an old network for image segmentation. It is widely used in medical image segmentation.

## Example Usage

You can check the example script for [train](../../example_script/unet_train.sh)
and [test](../../example_script/unet_test.sh) in the `example_script` folder.

## Network Metrics

### BUSI

|    Dataset     |       AUROC       |     Accuracy      |  AveragePrecision  |      F1Score      |   JaccardIndex    |     Precision     |      Recall       |    Specificity    |       Dice        |                                      Best Model Link                                       |
|:--------------:|:-----------------:|:-----------------:|:------------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:------------------------------------------------------------------------------------------:|
|    BUSI_all    | 84.60761308670044 | 96.4303195476532  | 61.68476343154907  | 76.65501236915588 | 62.14683651924133 | 84.09532308578491 | 70.42424082756042 | 98.79097938537598 | 76.65501236915588 | [Link](https://drive.google.com/file/d/1267OEJU8SImaEg5a7QAh9rW_IQUZck9I/view?usp=sharing) |
|    BUSI_bad    | 84.90326404571533 | 95.67922353744508 | 61.367279291152954 | 76.3655424118042  | 61.76719665527344 | 81.92418217658997 | 71.51328921318054 | 98.29322695732115 | 76.3655424118042  | [Link](https://drive.google.com/file/d/1QxD9ZNyfB8Y7i0jvsbqa4MseaeKCH3Za/view?usp=sharing) |
|  BUSI_benign   | 82.0100724697113  | 97.22399711608888 | 59.81850028038025  | 74.88324642181396 | 59.85069274902344 | 89.13832306861877 | 64.5589292049408  | 99.4612157344818  | 74.88324642181396 | [Link](https://drive.google.com/file/d/1zAakcYiY7BF5B23cYNHVey-pcKUi9auX/view?usp=sharing) |
| BUSI_malignant | 86.00512742996216 | 93.10449957847597 | 65.53518772125244  | 78.34042310714722 | 64.39313888549805 | 81.52351975440979 | 75.39654970169067 | 96.61369323730467 | 78.34042310714722 | [Link](https://drive.google.com/file/d/1ooAdETIRANmflKNJWKsmLJj7tU0Z3VuF/view?usp=sharing) |

### ISIC

|  Dataset  |       AUROC       |     Accuracy      | AveragePrecision  |      F1Score      |   JaccardIndex    |     Precision     |      Recall       |    Specificity    |       Dice        |                                       Best Model Link                                        |
|:---------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------------------------------------------------------------------------------:|
| ISIC-2016 | 93.52524280548096 | 94.76287961006165 | 84.9144995212555  | 90.71258902549744 | 83.0036997795105  | 90.7389521598816  | 90.6862497329712  | 96.3642418384552  | 90.71258902549744 |  [Link](https://drive.google.com/file/d/1wKWYf1Rd8qae6E-TaWKnw6rJshPyw9k-/view?usp=sharing)  |
| ISIC-2017 | 91.37052893638612 | 95.2273964881897  | 75.90051889419556 | 85.72306632995605 | 75.01344680786133 | 85.8748733997345  | 85.57179570198059 | 97.16926217079164 | 85.72306632995605 |  [Link](https://drive.google.com/file/d/1oFalIsXrnWGtpXoCe3awQoSP7bg4uB3T/view?usp=sharing)  |
| ISIC-2018 | 91.76375865936281 | 93.63263845443726 | 78.73244881629944 | 87.07996010780334 | 77.11648941040039 | 86.06358170509338 | 88.12063932418823 | 95.40687203407288 | 87.0799720287323  | [Link](https://drive.google.com/drive/folders/1JhsfiFyWcSPVq2UHccKPltb759YdHuMf?usp=sharing) |
