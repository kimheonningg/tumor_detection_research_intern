### These are the papers I read and analyzed to select my research topic and to study relevant background information

1.  Simarro, J., Meyer, M. I., Van Eyndhoven, S., Phan, T. V., Billiet, T., Sima, D. M., &
    Ortibus, E. (2024).
    _A deep learning model for brain segmentation across pediatric and adult
    populations._
    Scientific Reports, 14, 11735. https://doi.org/10.1038/s41598-024-61798-6

2.  Liu, X., Bonner, E. R., Jiang, Z., Roth, H., Packer, R., Bornhorst, M., & Linguraru, M. G.
    (2023).
    _From adult to pediatric: Deep learning-based automatic segmentation of rare
    pediatric brain tumors._
    In Proceedings of SPIE Medical Imaging 2023: Image Processing (Vol. 12464).
    https://doi.org/10.1117/12.2654245

### Analysis of paper 1

Existing brain segmentation tools are optimized for specific age groups: ChildMetrix for children and FreeSurfer / Icobrain v5.9 for adults. This separation hinders consistent brain monitoring across ages. Simarro et al. developed a unified deep learning model trained on T1-weighted MRIs from 390 patients aged 2–81, covering various pathologies and scanner types (Philips, Siemens, GE, Fujifilm). The model, called icobrain-dl, includes preprocessing (bias correction, registration, intensity normalization), and a multi-task U-Net CNN with 2 tasks:

- Task 1: Brain tissue segmentation (background + WM, GM, CSF = 4 regions)
- Task 2: Brain structure segmentation (22 + background = 23 anatomical regions)

Task1 and task2 shared encoder features, and complemented each other.

More details:

- Semi-automated labels with expert correction to reduce labeling cost.
- Patch-based learning (128 x 128 x 128 sized patch).
- Data augmentation using GMM for intensity augmentation.
- Optimizer: Adam.
- Initial LR=0.001, early stopping.
- Loss: weighted sum of soft dice + cross-entropy (α_task1=1, α_task2=10)

Result:

Trained models: icobrain-dl(cross-age generalized model), icobrain-dl-p(pediatric-only), icobrain-dl-a(adult-only)

Performance:

- Pediatric: Dice 82.2% (for generalized model vs 80.8% for pediatric-only),
  HD 3.26mm (for generalized model vs 3.23mm for pediatric-only)
- Adult: Dice 82.6% (for generalized model vs 81.9% for adult-only),
  HD 2.27mm (for generalized model vs 2.37mm for adult-only)

### Analysis of paper 2

This study analyzes Diffuse Midline Glioma -which is a rare but malignant central nervous system tumor that occurs in children- using MRI. Although DMG has a very low incidence, it is known that DMG is a fatal disease. Therefore MRI images are extremely valuable for predicting disease progression and survival. Although adult brain tumors have abundant imaging data and analytical expertise, models trained on adult cases cannot be directly applied to pediatric DMG.

This paper used the pretraining and transfer learning strategy by: First pretraining on adult brain tumor data, and then finetuning on pediatric data through transfer learning. Also, they used the SegResNet architecture.

Segmentation accuracy was compared under four conditions:

1. pretraining only
2. pretraining + finetuning on 50% of the pediatric data
3. no pretraining + full finetuning
4. pretraining + full finetuning

The highest accuracy was achieved with 4. pretraining + full finetuning.
