# CZII CryoET Object Identification – Kaggle Competition Portfolio Project

https://www.kaggle.com/competitions/czii-cryo-et-object-identification

---

## Project Overview
The project is based on the **CZII Cryo-ET Object Identification** Kaggle competition, which challenged participants to build machine learning models for identifying small biological structures in large 3D volumes. Specifically, the task was to automatically annotate five classes of protein complexes (particles) within 3D cryo-electron tomography (cryoET) images
linkedin.com
. These cryoET tomograms are volumetric datasets (think of them as 3D images) where the goal is to locate the centers of specific protein molecules (such as apo-ferritin, beta-galactosidase, ribosome, thyroglobulin, and a virus-like particle) and classify them into the correct category. This is a **computer vision and detection** problem in a dense 3D space, making it very challenging due to the crowded cellular environment and limited training examples.

**Problem Definition**: Given a 3D electron tomography volume, the model must find coordinates of protein particles of interest and identify their class. In essence, it’s like finding needles in a 3D haystack – the proteins are the needles and the cell volume is the haystack. The competition provided a training set of labeled 3D volumes with annotated particle centers for five target protein classes (plus one additional class that was present but not scored in the competition). The evaluation metric was based on how well the predicted particle positions and classes matched the ground truth (a form of F-beta score for detection).

**Why This Matters**: Automating cryoET annotations has real-world impact in biology and drug discovery, as it could vastly accelerate understanding of cellular machinery. From a data science perspective, this project was an opportunity to apply advanced computer vision techniques to 3D data and tackle a novel domain problem.
