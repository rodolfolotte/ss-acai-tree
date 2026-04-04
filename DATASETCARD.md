---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
---

# Dataset Card for ss-acai-tree Geospatial Dataset

<!-- Provide a quick summary of the dataset. -->

This dataset comprises high-resolution optical geospatial imagery meticulously annotated for the binary semantic segmentation of Açaí trees natively arrayed across spatial environments. It acts as the principal training and evaluative foundation for the `ss-acai-tree` DeepLabV3 segmentation pipeline.

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

The dataset encompasses geographic representations (predominantly `.TIF` arrays alongside supplementary `.jpg`/`.png` extracts) depicting natural topologies interspersed with Açaí palm distributions. High-resolution orthomosaic bounds (e.g. 2048x2048 bounds) are parsed computationally into discrete 256x256 tiles to satisfy computational VRAM constraints natively encountered during Deep convolutions. Semantic masks designate pixels logically as `0` for contextual background or `1` for affirmative 'Açaí' (visually mapped to RGB vector `[102, 153, 0]`). A dynamic preprocessing engine rigorously excludes sparse data tiles preserving less than 15% visible ground representation (white masking criteria) to intrinsically boost topological signal-to-noise ratio during training. 

- **Curated by:** Dr. Rodolfo G. Lotte
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** Dr. Rodolfo G. Lotte
- **Language(s) (NLP):** N/A (Geospatial Vision Data)
- **License:** [More Information Needed]

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** Private GitHub Repository (`ss-acai-tree`)
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

The dataset natively satisfies direct utilization for supervised deep learning architectures handling binary semantic semantic segmentation natively mapped for isolated geospatial flora identification constraints.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

Due to narrow categorical tagging (`background` vs `acai`), the dataset is statistically insufficient for holistic tropical multispecies mapping schemas or multi-class semantic segmentation. Applying inference directly onto urban topographies or distinctly variant geographic biomes without significant domain-transfer normalization will yield volatile precision degradations.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

The root organizational hierarchy natively separates source information into `image` and `label` (semantic mask) branches.
- **Images:** Contains raw visual tiles (256x256).
- **Labels:** Possesses identically named counterpart masks depicting the ground truth.
- **Splits:** Contains stochastic algorithmic distribution wherein a reserved 10% data allocation serves strictly as hold-out validation arrays, guaranteeing test/train disjoint integrity.

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

Geospatial cataloging of specific biological species presents extreme scale challenges manually. This dataset serves to bridge traditional geographic information architectures with modern deep learning semantic categorizers. 

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

Original multispectral/optical images were geometrically cropped into 256x256 coordinate grids, accompanied by parallel categorical rasterization mapping true locations of Açaí tree groupings. An 80-pixel buffer is explicitly utilized during this tiling process strictly to capture adjoining neighborhood context for tree targets split across crop boundaries. Because the target fully convolutional inference mechanics are naturally invariant to structural dimensionality, no subsequent spatial recombination procedures are structured or utilized after data generation. Finally, the data undergoes offline morphological filtering, rejecting patches possessing insufficient terrain data points (white ratio < 0.15).

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

[More Information Needed]

### Annotations [optional]

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

Geospatial tracing procedures were utilized to translate geographic coordinate bounds explicitly delineating tree canopies into strictly normalized pixel representations forming boolean `1` mappings mapped specifically to the categorical RGB value of `[102, 153, 0]`.

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

[More Information Needed]

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

Given the high-altitude remote sensing derivation native to overhead floral mappings, explicit Privacy Indicating Information (PII) matrices logically fall out of scope unless coordinates coincide explicitly with restricted civil infrastructures. 

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Class imbalance poses a significant algorithmic inhibitor given forest backgrounds universally dominate spatial coverage against strictly isolated Açaí groupings. The dataset leverages synthetic dataset augmentation operations (`rotation`, `blur`, `resize`) natively addressing inherent variances mapping structural representations.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users extracting derived geospatial aggregations should be statistically aware of potential false positives inherent in binary dense prediction, especially over highly heterogeneous Amazonian and tropical floras. Ensure spatial distributions used conceptually map proportionally against the categorical densities present across these source sets.

## Citation [optional]

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

- **Semantic Mask:** Rasterized 2-dimensional bounding overlay directly indicating pixel-by-pixel spatial classification properties.
- **Tiling Buffer:** A boundary overlap methodology (80px default) applied specifically during dataset creation ensuring localized contextual continuity of tree targets near the edges, avoiding post-inference spatial recombination constraints.

## More Information [optional]

[More Information Needed]

## Dataset Card Authors [optional]

Dr. Rodolfo G. Lotte

## Dataset Card Contact

[More Information Needed]
