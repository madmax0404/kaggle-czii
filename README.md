# CZII - CryoET Object Identification (Kaggle 대회)
https://www.kaggle.com/competitions/czii-cryo-et-object-identification

---

## 기술 스택

* **언어**: Python
* **딥러닝 프레임워크**: PyTorch, MONAI, TorchIO
* **ML/DL 모델**: MedicalNet, 3D U-Net, YOLO2D
* **Loss Functions**: Weighted Tversky Loss, Cross-Entropy Loss, F-beta (β=4) based custom loss
* **하이퍼파라미터 튜닝**: Optuna
* **실험 결과 로깅**: MLFlow
* **데이터 처리 및 분석**: Pandas, Numpy, Zarr
* **시각화**: Napari, Matplotlib, Seaborn
* **OS**: Ubuntu Desktop 24.04 LTS, Windows 11
* **IDE**: VSCode, Jupyter Notebook

---

## 프로젝트 개요

이 프로젝트는 **Kaggle CZIi Cryo-ET Object Identification** 대회에 참가하여 진행된 것입니다. 목표는 Cryo-Electron Tomography (Cryo-ET)로 촬영된 3차원 토모그램에서 특정 생물학적 입자를 식별하는 모델을 만드는 것입니다.

Cryo-ET 데이터는 전자현미경 기반의 3D 볼륨 데이터로, **저대비·고노이즈·고차원**이라는 특성을 가지고 있습니다. 이는 기존의 2D 이미지 처리 기법으로는 한계가 있으며, **3차원 기반의 딥러닝 모델**을 적용해야만 유의미한 성능을 낼 수 있었습니다.

본 프로젝트는 단순히 대회 참가를 넘어, **고차원 생물학 데이터를 머신러닝으로 다루는 경험과, 클래스 불균형 및 Recall 중심 평가 환경**에서 성능을 개선하기 위한 다양한 실험을 목표로 하였습니다.

---

## 문제

대회에서 다루는 문제는 단순한 이미지 분류나 객체 검출이 아니라, **Cryo-ET 3D 데이터 내 특정 입자를 좌표 단위로 식별하는 작업**이었습니다.

- 목표 클래스: ribosome, virus-like particles, apo-ferritin, thyroglobulin, β-galactosidase
- 클래스 가중치: “hard” 클래스(thyroglobulin, β-galactosidase)는 weight=2, 나머지는 weight=1
- 평가지표: **F-beta score (β=4)**
  - Recall에 높은 비중 → 놓친 객체가 많을 경우 큰 패널티
  - False Positive는 상대적으로 덜 penalize
  - 즉, **“많이 찾는 것”이 가장 중요한 문제**

이러한 특성은 학습 과정에서 precision/recall trade-off를 관리하는 것이 핵심임을 의미했습니다.

---

## 데이터셋

데이터셋은 3D Cryo-ET 볼륨 데이터와 함께 각 입자의 좌표와 반지름 정보가 제공되었습니다.
- **Train set**: 입자 위치 및 반지름 라벨 제공
- **Test set**: 레이블 비공개 (제출 시 자동 평가)
- **Class 특이점**: beta-amylase는 학습 데이터에 포함되지만 weight=0으로 평가에는 반영되지 않음

데이터의 구조적 특성상, 단순 이미지 전처리 대신 **voxel 단위 patch 추출, scaling(normalization), augmentation** 이 중요했습니다.

추가적으로 제공된 class weight:
- Easy classes (ribosome, virus-like, apo-ferritin) → weight=1
- Hard classes (thyroglobulin, β-galactosidase) → weight=2
- Beta-amylase → weight=0 (학습 가능하나 평가에 반영 X)

![ribosomes](<assets/figs/ribosome_identification.png>)
*Figure 1. Ribosomes*

![ribosome_close_up](<assets/figs/ribosome_close_up.png>)
*Figure 2. Ribosome close-up*

---

## 방법론 및 접근 방식

이 프로젝트에서는 단일 접근 방식에 의존하지 않고, 다양한 아이디어를 병렬적으로 실험했습니다.

1. **EDA & 전처리**
    - Napari를 활용한 Cryo-ET 3D 이미지 시각화
    - voxel 크기 단위 변환 (angstrom 기반 normalization 실험)
    - 데이터 augmentation: 회전, 뒤집기, random patch 추출 등
2. **모델링**
    - **3D U-Net**: voxel 단위 segmentation 접근(Figure 3) → 안정적이나 Recall 최적화에 어려움
    - **YOLO2D**: bounding box 기반 object detection
    - **MedicalNet Transfer Learning**: 기존 의료 영상 모델 가중치 활용
    - **Custom UNet**: Segmentation 접근이 아닌, 입자 중심의 heatmap(Figure 4) 및 radius map(Figure 5) 방식으로 접근
3. **Loss Functions**
    - Cross Entropy Loss (baseline)
    - Weighted Tversky Loss (class imbalance 개선)
    - F-beta(β=4)에 근접한 custom loss 설계
4. **Hyperparameter Tuning**
    - Optuna를 활용하여 learning rate, UNet 모델 채널 사이즈, residual unit 갯수, 최적의 loss function 등 탐색
    - 수십 개의 실험 기록 (날짜별 노트북)
5. **실험 전략**
    - class weight 적용/비적용 비교
    - voxel 크기 변화 실험 (10 vs angstrom 단위)
    - custom loss (FBetaLoss, TverskyCE 조합) vs baseline loss 비교
    - ensemble 미도입 (추후 계획)

![painted_segmentation](<assets/figs/painted_segmentation.png>)
*Figure 3. 3D U-Net Painted Segmentation*

![YOLO2D](<assets/figs/train_batch0.jpg>)
*Figure 4. YOLO2D Object Detection*

![heatmap_approach](<assets/figs/heatmap_approach.png>)
*Figure 5. Heatmap Approach*

![radius_map_approach](<assets/figs/radius_map_approach.png>)
*Figure 6. Radius Map Approach*

---

## 결과 및 주요 관찰

- **3D U-Net**: 안정적인 학습은 가능했으나 Recall 최적화가 어렵고 overfitting 발생 (Figure 8, 9)
- **YOLO3D**: detection 성능 초기 구현은 어려웠으나 bounding box 기반 recall 개선 가능성 확인
- **Custom Loss (Tversky + CE)**: imbalance 상황에서 baseline CE 대비 recall이 향상됨
- **Optuna 튜닝**: 특정 조합에서 validation 성능 소폭 개선, 그러나 leaderboard 반영은 제한적
- 핵심 관찰:
    - **Recall 중심의 평가 환경**에서는 precision 손실을 감수하더라도 recall을 끌어올리는 전략이 필수
    - class weight를 올바르게 반영하지 않으면 성능이 급격히 하락
    - voxel scaling, augmentation 방식에 따라 결과가 크게 달라짐

![Segmentation First Epoch Patches](<assets/figs/first_epoch_patches.png>)
*Figure 7. Segmentation Approach First Epoch Patches*

![Segmentation 52nd Epoch Patches](<assets/figs/52nd_epoch_patches.png>)
*Figure 8. Segmentation Approach 52nd Epoch Patches*

![Segmentation 52nd Epoch Tomogram and Predictions](<assets/figs/52nd_epoch_tomogram_labels_and_predictions.png>)
*Figure 9. Segmentation Approach 52nd Epoch Tomogram, Labels, and Predictions*

---

## 결론 및 향후 과제

본 프로젝트는 단순한 대회 참가를 넘어서, **Cryo-ET와 같은 도전적인 데이터셋을 다루는 과정**에서 다양한 인사이트를 얻을 수 있었습니다.

- Recall 중심의 환경에서 precision/recall trade-off를 조율하는 것이 핵심
- Custom loss 설계와 데이터 증강이 불균형 문제 해결에 중요한 역할

향후 개선 방향:
1. YOLO3D 개선 및 앙상블 도입
2. Semi-supervised/self-supervised 학습으로 데이터 효율 개선
3. Active learning 기반 minority class 보강
4. Uncertainty estimation과 inference-time augmentation 도입

---

## 프로젝트 실행 방법

분석 및 모델 훈련 재현 방법:

1.  **Repository 복제:**
    ```bash
    git clone [https://github.com/madmax0404/kaggle-czii.git](https://github.com/madmax0404/kaggle-czii.git)
    cd kaggle-czii
    ```
2.  **데이터셋 다운로드:**
    * 캐글에서 대회에 참가하세요. [CZII - CryoET Object Identification](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/overview)
    * 데이터를 다운받은 후 알맞은 디렉토리에 저장하세요.
3.  **가상 환경을 생성하고 필요한 라이브러리들을 설치해주세요:**
    ```bash
    conda create -n czii python=3.12 # or venv
    conda activate czii
    pip install -r requirements.txt
    ```
4.  **Jupyter Notebook을 실행해주세요:**
    ```bash
    jupyter notebook notebooks
    ```
    데이터 처리, 모델 훈련 및 평가를 실행하려면 노트북의 단계를 따르세요.

---

## Acknowledgements

데이터셋과 대회 플랫폼을 제공한 CZ Imaging Institute와 Kaggle에 감사드립니다.

본 프로젝트는 다음 오픈소스의 도움을 받았습니다: Python, PyTorch, MONAI, TorchIO, Optuna, MedicalNet, MLFlow, pandas, numpy, matplotlib, seaborn, Jupyter, napari, ultralytics, scikit-learn.

모든 데이터 이용은 대회 규정과 라이선스를 준수합니다.

---

## License

Code © 2025 Jongyun Han (Max). Released under the MIT License.
See the LICENSE file for details.

Note: Datasets are NOT redistributed in this repository.
Please download them from the official Kaggle competition page
and comply with the competition rules/EULA.
