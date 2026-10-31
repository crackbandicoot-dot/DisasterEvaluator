---

### State of the Art Report: Automated Assessment of Building Damage Using Pre and Post-Disaster Satellite Imagery

**1. Introduction and Problem Relevance**

The rapid and accurate assessment of building damage after a natural disaster is a critical task for emergency response and recovery. Traditional field inspections are slow and costly, while manual interpretation of satellite images is labor-intensive and prone to errors. In this context, deep learning applied to remote sensing has emerged as a scalable and automated solution. Recent literature confirms that the use of artificial intelligence (AI) has significantly advanced in disaster damage assessment, enabling faster and more accurate analysis than traditional inspections.

The proposed project aligns with the frontier of this area by addressing the **segmentation and classification of building damage** from pairs of satellite images (pre/post-disaster), a formulation that the scientific community considers the standard for this problem.

**2. Reference Datasets: Industry Standard**

The state of the art in this domain is intrinsically linked to the availability of large-scale public datasets annotated rigorously.

*   **xBD (xView2):** This is the predominant benchmark and de facto standard for building damage assessment. The xBD dataset, released as part of the xView2 challenge, provides very high-resolution (VHR) satellite images of multiple disaster events (hurricanes, earthquakes, floods, etc.) along with building-level annotations for four damage levels: no damage, minor, major, and destroyed. The existence of this benchmark has accelerated research in automated damage detection.

*   **Complementary Datasets for Floods:** For the optional scope of generalization to other types of disasters, the literature offers alternatives. **Sen1Floods11** is a geo-referenced dataset designed for training and testing algorithms for flood detection using Synthetic Aperture Radar (SAR) data from Sentinel-1. On the other hand, **SpaceNet 8** focuses on the detection of flooded roads and buildings, providing pre and post-flood event images.

**3. Dominant Architectures and Methodological Approaches**

Recent literature shows an evolution from purely convolutional models to more sophisticated architectures that better capture temporal and spatial information.

*   **Siamese Networks:** These are the fundamental approach for processing image pairs. A siamese network uses two identical branches (with shared weights) to extract features from the pre and post-disaster images separately. These features are then compared or fused to detect changes. The combination of a **U-Net** architecture (effective for segmentation) with a siamese design is a widely validated strategy. For example, a recent study proposed a UNet-based siamese network that achieved a general accuracy of 95.5% in the assessment of damage from the 2023 Morocco earthquake, using the xBD dataset.

*   **Transformer-Based Architectures:** More recently, transformers have been integrated to overcome the limitations of CNNs in capturing long-range dependencies. A notable work proposes a **hierarchical UNet with transformers** that uses the difference between the features encoded by transformers of the pre and post-disaster images. This method, which achieves state-of-the-art results on the xBD dataset, demonstrates the effectiveness of transformers for modeling complex changes. Other proposals, such as **DDNet**, employ a joint attention module to better extract temporal features, achieving a total F1-score of 79.56% on xBD.

*   **Hybrid Approaches (Spatial + Frequency):** Research also explores complementary representations. A recent comparative study evaluated models that operate in the spatial domain (like CNNs), in the frequency domain, and in a dual domain. The results show that dual-domain models, which combine both representations, offer measurable improvements over single-domain approaches, although all models struggle to detect subtle damage due to class imbalance.

**4. Central Challenges of the State of the Art: Class Imbalance and Evaluation**

The literature identifies **severe class imbalance** as the main technical obstacle. The vast majority of buildings in a scene do not present damage, which biases the models towards the majority class. A naive model can achieve high overall accuracy (accuracy) by simply predicting "no damage" for almost all buildings, which is useless in practice. Therefore, the community has developed several strategies:

*   **Specialized Loss Functions:** The use of **weighted loss functions** is fundamental. The combination of **Focal Loss and Dice Loss** has proven particularly effective in addressing class imbalance and improving building boundary precision. Customized losses such as the **Ordinal Class Distance Penalty Loss (OCDPL)** have also been proposed, which penalize classification errors according to the ordinal distance between damage classes, improving the F1-scores of minority classes.

*   **Appropriate Evaluation Metrics:** To reflect real performance on minority classes, the **F1-score per class (or macro-F1)** is the standard metric, rather than overall accuracy. Studies report both the F1-score for building localization and damage classification, with the latter being the most challenging.

**5. Limitations and Future Directions: Generalization and Label Noise**

The state of the art recognizes two important limitations that define future research lines and are relevant to your project's realism:

*   **Generalization to Unseen Events:** Many complex-architecture models achieve excellent performance on the original xBD data split but **see degraded performance when evaluated on completely new disaster events** (e.g., a hurricane not seen during training). This has motivated "data-centric" approaches. An example is **DisasterAdaptiveNet**, which incorporates information about the type of disaster as an input variable, significantly improving its ability to generalize to unseen events compared to purely model-centric methods.

*   **Inherent Noise in Labels:** Damage labels are assigned by human analysts and involve a degree of subjectivity, especially in distinguishing between "minor damage" and "major damage." This noise in labels sets a practical ceiling on model precision. A study on transformer-based methods for patch classification confirms that both label noise and severe class imbalance remain open challenges.

**Conclusion: Project Positioning**

The proposed project is not an isolated exercise but an implementation of medium complexity that sits directly on top of the current practices and challenges of the state of the art. The choice of the xBD dataset, the use of a segmentation architecture (U-Net or similar) with a siamese approach for processing the image pair, and evaluation using F1-score per class are methodological decisions that the literature supports as the right path.

The project aligns with the knowledge frontier by explicitly acknowledging the risks (class imbalance, computation) and proposing standard industry mitigations (class weights, specialized loss functions). Additionally, the idea of a **"cell-based impact map"** as the final deliverable, although a post-processing development, connects with the practical need to translate model predictions into actionable information for disaster response. The post-processing stage is, therefore, an area where the team can add value and justify its own design decisions.

In summary, the state of the art validates the proposal: the datasets, architectures, and metrics you plan to use are the ones the scientific community employs to advance in this field. The project's success will depend on rigorous execution to handle known challenges, especially class imbalance and computational resource limitations, as your planning anticipates.
