# CZII CryoET Object Identification – Kaggle Competition Portfolio Project

https://www.kaggle.com/competitions/czii-cryo-et-object-identification

---

## 기술 스택 (Tech Stack)

* **Python**
* **PyTorch**: Base deep learning framework for building and training models.
* **MONAI**: Medical imaging toolkit (provided U-Net implementation and many 3D transforms).
* **TorchIO**: Toolkit for 3D data augmentation and patch sampling.
* **MedicalNet**: Provided a pretrained 3D ResNet backbone for feature extraction.
* **Optuna**: Hyperparameter optimization.
* **MLflow**: Experiment tracking and logging.
* **Pandas/NumPy/Zarr**: Data manipulation and loading (the cryoET data was stored in Zarr format for efficient IO of large 3D arrays).
* **Matplotlib/Seaborn**: Used for plotting training curves and any data exploration charts.
* **Linux (Ubuntu Desktop 24.04 LTS)**: OS.
* **VSCode, Jupyter Notebook**: IDE.

By combining these tools and techniques, the project covers a wide range of modern data science skills, from deep learning model development to experiment management, all applied in a complex 3D computer vision context.

---

## Project Overview
The project is based on the **CZII Cryo-ET Object Identification** Kaggle competition, which challenged participants to build machine learning models for identifying small biological structures in large 3D volumes. Specifically, the task was to automatically annotate five classes of protein complexes (particles) within 3D cryo-electron tomography (cryoET) images
linkedin.com
. These cryoET tomograms are volumetric datasets (think of them as 3D images) where the goal is to locate the centers of specific protein molecules (such as apo-ferritin, beta-galactosidase, ribosome, thyroglobulin, and a virus-like particle) and classify them into the correct category. This is a **computer vision and detection** problem in a dense 3D space, making it very challenging due to the crowded cellular environment and limited training examples.

**Problem Definition**: Given a 3D electron tomography volume, the model must find coordinates of protein particles of interest and identify their class. In essence, it’s like finding needles in a 3D haystack – the proteins are the needles and the cell volume is the haystack. The competition provided a training set of labeled 3D volumes with annotated particle centers for five target protein classes (plus one additional class that was present but not scored in the competition). The evaluation metric was based on how well the predicted particle positions and classes matched the ground truth (a form of F-beta score for detection).

**Why This Matters**: Automating cryoET annotations has real-world impact in biology and drug discovery, as it could vastly accelerate understanding of cellular machinery. From a data science perspective, this project was an opportunity to apply advanced computer vision techniques to 3D data and tackle a novel domain problem.

Techniques and Tools Used
This project explored multiple cutting-edge techniques and models in an attempt to solve the 3D object detection problem. The following approaches were implemented and evaluated:
3D YOLO-Based Detection Model: We adapted the YOLO (You Only Look Once) object detection paradigm for 3D data. The custom model, YOLO3D_MedicalNet, uses a 3D ResNet-18 backbone (from MedicalNet pretrained on medical imaging data) to extract volume features, followed by a detection head that predicts bounding spheres for particles. Each forward pass outputs a set of predictions per volume patch, with each prediction encoding the 3D coordinates, radius (size) of the particle, and class probabilities. This single-stage detector is analogous to YOLO in 2D, but operates on 3D patches. By using a pretrained 3D ResNet backbone, we leveraged transfer learning to handle the limited data. We defined multiple anchors (proto-typical particle sizes) and the network predicts adjustments to these anchors. The detection model required designing a custom target encoding for 3D centroids and radii of particles.
3D U-Net Segmentation Model (MONAI): In addition to direct detection, we experimented with a segmentation approach. Using the MONAI framework (a medical imaging toolkit built on PyTorch), we built a 3D U-Net model to segment out voxels belonging to any particle. The idea was to create binary or multi-class segmentation volumes where each protein particle is “painted” in the volume. From these segmentations, we could then derive particle centers by connected-component analysis. The U-Net architecture from MONAI was applied, which is well-suited for volumetric data. We trained the U-Net to output separate channels for each protein class (plus background), effectively treating it as a multi-class segmentation. This approach turned the problem into one of detecting shapes in the volume via segmentation masks. After training, we would locate the highest responses in the predicted mask to recover particle coordinates. The U-Net was trained with appropriate loss functions (like combined Dice + Cross-Entropy, etc.) to handle class imbalance.
Custom Loss Function for Detection: For the YOLO-style model, a specialized combined loss function was implemented to guide the training:
Focal Loss for classification: to address class imbalance and focus on hard-to-detect particles, we used a focal loss component (modulating the cross-entropy based on prediction confidence).
Soft F-beta Loss: We introduced a custom “Soft F$\beta$” loss (with $\beta=4$ to emphasize recall) that directly approximates the F$\beta$ detection score. This loss is computed on the predicted vs. true class labels in a differentiable manner, encouraging the model to improve the competition’s target metric.
Regression Loss: For the particle coordinates and size (radius), we used a Smooth L1 loss to regress the model’s predicted center positions to the ground truth coordinates (and radius). This is similar to the localization loss in typical object detectors.
These components were combined into one Composite Loss function that the model optimized. Balancing these losses was key – we weighted the focal, F-beta, and regression terms (tuning hyperparameters $\lambda$ for each) to stabilize training. This custom loss was critical for the 3D detector to learn effectively, given that a standard loss (like just binary cross-entropy on class labels) was insufficient for good performance in this context.
Data Augmentation (3D Augmentations): To improve generalization on the limited data, extensive data augmentation was applied to the 3D volumes:
Spatial transformations: Random flips along each of the 3 axes (x, y, z) were used to simulate different orientations of the sample. We also applied random rotations (e.g. 90° rotations in the XY plane, small degree rotations around arbitrary axes) to augment the data with rotated versions of the tomograms.
Scaling and Elastic deformations: Slight random scaling (zoom in/out by a few percent) and elastic grid distortions were introduced to make the model robust to small size variations and deformations of structures.
Intensity augmentations: We varied the intensity distribution of volumes through shifts in mean and standard deviation, simulating different imaging conditions. Small Gaussian noise was also added in some cases to mimic imaging noise.
We leveraged TorchIO and MONAI’s transformation APIs to implement these augmentations on-the-fly during training. For example, transforms like RandomFlip, RandomAffine, and RandomNoise were composed to provide a different augmented volume each epoch. The augmentations were crucial given the limited number of unique tomograms – effectively multiplying the training examples and making the model more robust to variations.
Training Procedure and Tuning: Training 3D models is computationally intensive. We used mixed precision (AMP) to speed up training and reduce memory usage. Due to memory limits, we trained on smaller sub-volumes (patches) extracted from the full 3D images. Patches of size 64×128×128 voxels were used, sliding through the volume to cover all regions. The code is optimized to generate these patches and corresponding labels (anchoring any particles falling inside a patch with adjusted coordinates). We also employed Optuna for hyperparameter tuning in the U-Net experiments – an automated search to find optimal learning rates, augmentation parameters, and other hyperparams by maximizing validation F-beta score. Experiment tracking was done with MLflow, keeping logs of training/validation loss curves and metrics for each run. This helped in comparing the various experiments (YOLO vs U-Net, different augmentation settings, etc.).
Visualization and Debugging Tools: Throughout development, we used Napari, a 3D image viewer, to visually inspect tomogram slices and the model’s predictions. This helped in debugging whether the model’s predicted coordinates aligned with actual particles. We also plotted intermediate outputs and heatmaps to understand how the networks were responding to features in the data.
