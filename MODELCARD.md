---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
---

# Model Card for DeepLabV3-Acai-Segmenter

<!-- Provide a quick summary of what the model is/does. -->

This model is a customized DeepLabV3 semantic segmentation network specialized in the binary identification of Açaí trees from geospatial imagery. It predicts binary masks, categorizing pixels as either background or 'Açaí'.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model leverages the DeepLabV3 architecture, operating with configurable backbone encoders, prominently ResNet-50 or MobileNet-V3-Large. It has been fine-tuned from PyTorch standard pre-trained checkpoints (COCO with VOC labels). The network’s classifier head was modified to perform single-class binary segmentation via a 1x1 2D convolutional layer followed by a sigmoid activation function using BCEWithLogitsLoss. It trains upon localized imagery cropped to 256x256 pixels employing overlapping edge-buffers to minimize loss of tree context. During native inference operations, owing to spatial invariance within the fully convolutional footprint, the framework scales holistically over imagery independent of sizing boundaries, wholly omitting any subsequent tile-recombination procedures.

- **Developed by:** Dr. Rodolfo G. Lotte
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** Dr. Rodolfo G. Lotte
- **Model type:** Semantic Segmentation Convolutional Neural Network (CNN)
- **Language(s) (NLP):** N/A (Vision Model)
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** `DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1` or `DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1`

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** Private GitHub Repository (`ss-acai-tree`)
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Direct inference on geographic dataset tiles (e.g., `.tif`, `.jpg`, `.png`) for localized identification of Açaí tree prevalence. Typical workflows involve running the underlying `modules/initialize.py` inference script to generate localized prediction mask files that delineate identified Açaí regions.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The model can be integrated into remote sensing pipelines assessing biodiversity, forest inventory mappings, or economic evaluation engines for agricultural yields strictly focusing on Açaí palm distributions across varied topographies.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

Deploying this model on distinct tree species mapping (e.g. Eucalyptus, Pine) is out-of-scope without comprehensive fine-tuning due to structural feature variances. It is not designed for urban remote sensing mapping tasks or spatial resolution environments heavily differentiated from its trained ground sample distance (GSD).

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The performance of the predictive segmenter is bounded by the quality and resolution of the source imagery. Disparities in atmospheric noise, shadow projections from surrounding dominant tree canopies, or seasonal variations in Açaí phenomenology could degrade identification recall metrics. While tile-based boundaries fundamentally risk contextual loss for targets on edges, the inclusion of an 80-pixel dataset buffer neutralizes omission rates during dataset generation. Furthermore, the inherent translation invariance mapping spatial dimensions during active inference effectively safeguards against boundary artifact generation implicitly, alleviating any need for mechanical recombination.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users extracting derived geospatial aggregations should be statistically aware of potential false positives inherent in binary dense prediction, especially over highly heterogeneous Amazonian and tropical floras. Pre-inferencing illumination standardization is recommended.

## How to Get Started with the Model

Use the code below to get started with the model.

```bash
# Prepare the settings.py file specifying the imagery and annotations source.
# Ensure the model checkpoints (e.g. deeplabv3-resnet50-*.pth) are loaded.
python main.py -augment False -train False -validate False -predict True -verbose True
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Custom optical geospatial imagery annotated specifically for Açaí tree classes. Images and corresponding categorical truth masks are tiled to dimensions of 256x256 from expansive spatial scenes (e.g. 2048x2048 tiles). Background samples exhibiting low information potential (white ratio < 15%) are systematically omitted. The global dataset adheres to a split ratio allocation algorithm preserving 10% for hold-out validation.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

Raw data formats undergo normalization against ImageNet moments (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`). A stochastic augmentation regimen incorporates rotation, blur variants, multiplicative color modulations, and scale jittering (`resize-1`, `resize-2`) strictly on the training partition to fortify morphological invariance.

#### Training Hyperparameters

- **Training regime:** Standard FP32 precision
- **Optimizer:** Adam
- **Learning Rate:** 1e-5
- **Loss Function:** Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`)
- **Early Stopping:** Triggered contingent on `val_iou` plateauing for 20 continuous computational epochs.
- **Max Epochs:** 100
- **Batch Size:** 8 (Training), 1 (Prediction)

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

Testing was synthesized natively by preserving a configurable fraction (default configuration bypasses strict testing segmentation, relying primarily on `VALIDATION_SPLIT = 10%` to evaluate cross-epoch generalization).

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Evaluation is currently measured globally across all validation tiles without domain-specific stratification.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Performance metrics isolate primarily foreground phenomena against the environmental background, avoiding artificial metric inflation from predominant background representation.
- **Intersection over Union (IoU):** Principal metric dictating spatial overlap convergence.
- **Precision and Recall:** Disaggregated to analyze the model's tendency for omission versus commission errors.
- **F1-Score:** Harmonic evaluation of prediction compactness.
- **Accuracy:** General metric of overall correctness.

### Results

[More Information Needed]

#### Summary

Validation scores are actively serialized to `artefacts/plots/deeplabv3-metrics-*.txt` to maintain experiment traceability, accompanied by compiled Precision-Recall topological curves and Confusion Matrices.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Nvidia Architecture GPU Arrays
- **Hours used:** [More Information Needed]
- **Cloud Provider:** Local Computing Environment
- **Compute Region:** Local Computing Environment
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

A Fully Convolutional deep neural network utilizing Atrous Spatial Pyramid Pooling (ASPP) natively inherited from the DeepLabV3 architecture topology, paired with ResNet-50 or MobileNet-V3 encoder components, engineered for geospatial pixel categorization mapping in earth observation workloads.

### Compute Infrastructure

PyTorch (`v2.6+`) CUDA-enabled environment execution framework utilizing natively available compatible Graphics Processing Units.

#### Hardware

- Requires hardware-accelerated GPU with minimum 8GB VRAM availability suitable for computational graphing given 8-batch iterations of 256x256 matrices.

#### Software

- Framework dependency on heavily numerical and vision operations frameworks utilizing `torchvision` `0.21`, `opencv-python`, and explicit dependency on `numpy<2.0.0` or compatibilities enabling deterministic behavior coupled by metric reporting operations handled universally via `scikit-learn` ecosystems.

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

- **IoU:** Intersection-Over-Union, a definitive measure for semantic segmentation assessing the degree of overlap between predicted regions and truth bounding regions.
- **ASPP:** Atrous Spatial Pyramid Pooling; a DeepLab component sampling multi-scale contextual convolutional features parallelly across dilated convolutions.
- **BCEWithLogitsLoss:** Combines a Sigmoid layer and the BCELoss in one single coherent computational class offering supreme numerical stability.

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

Rodolfo Lotte

## Model Card Contact

[More Information Needed]
