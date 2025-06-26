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
3.  Fu, J., Bendazzoli, S., Smedby, Ö., & Moreno, R. (2024).
    _Unsupervised domain adaptation for pediatric brain tumor segmentation._
    arXiv preprint arXiv:2406.16848. https://doi.org/10.48550/arXiv.2406.16848
4.  Yen, CT., Tsao, CY. Lightweight convolutional neural network for chest X-ray images classification. 
    Sci Rep 14, 29759 (2024). https://doi.org/10.1038/s41598-024-80826-z
5.  Zhang Y, Chen Z, Yang X. Light-M: An efficient lightweight medical image segmentation framework
    for resource-constrained IoMT. Comput Biol Med. 2024 Mar;170:108088. doi: 10.1016/j.compbiomed.2024.108088. Epub 2024 Feb 3. PMID: 38320339
6.  Zhang, Z., Huang, R., & Huang, N. (2024, October). RepMedSAM: Segment Anything in Medical Images
    with Lightweight CNN. In Proceedings of the CVPR 2024 Workshop on MedSAMonLaptop. https://openreview.net/forum?id=AXeZItjN2z openreview.net
7.  Oktay, O., Schlemper, J., Le Folgoc, L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S.,
    Hammerla, N. Y., Kainz, B., Glocker, B., & Rueckert, D. (2018). Attention U-Net: Learning where to look for the pancreas. arXiv. https://arxiv.org/abs/1804.03999

### Analysis of paper 1

Simarro, J., Meyer, M. I., Van Eyndhoven, S., Phan, T. V., Billiet, T., Sima, D. M., &
Ortibus, E. (2024).
_A deep learning model for brain segmentation across pediatric and adult
populations._
Scientific Reports, 14, 11735. https://doi.org/10.1038/s41598-024-61798-6

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

Liu, X., Bonner, E. R., Jiang, Z., Roth, H., Packer, R., Bornhorst, M., & Linguraru, M. G.
(2023).
_From adult to pediatric: Deep learning-based automatic segmentation of rare
pediatric brain tumors._
In Proceedings of SPIE Medical Imaging 2023: Image Processing (Vol. 12464).
https://doi.org/10.1117/12.2654245

This study analyzes Diffuse Midline Glioma -which is a rare but malignant central nervous system tumor that occurs in children- using MRI. Although DMG has a very low incidence, it is known that DMG is a fatal disease. Therefore MRI images are extremely valuable for predicting disease progression and survival. Although adult brain tumors have abundant imaging data and analytical expertise, models trained on adult cases cannot be directly applied to pediatric DMG.

This paper used the pretraining and transfer learning strategy by: First pretraining on adult brain tumor data, and then finetuning on pediatric data through transfer learning. Also, they used the SegResNet architecture.

Segmentation accuracy was compared under four conditions:

1. pretraining only
2. pretraining + finetuning on 50% of the pediatric data
3. no pretraining + full finetuning
4. pretraining + full finetuning

The highest accuracy was achieved with 4. pretraining + full finetuning.

### Analysis of paper 3

Fu, J., Bendazzoli, S., Smedby, Ö., & Moreno, R. (2024).
_Unsupervised domain adaptation for pediatric brain tumor segmentation._
arXiv preprint arXiv:2406.16848. https://doi.org/10.48550/arXiv.2406.16848

Pediatric brain-tumor segmentation faces two fundamental difficulties:

1. The domain shift between adult and pediatric MRIs reduces generalization. In one study, when a model trained on adult data was applied to pediatric cases, the Dice score for the Tumor Core(TC) region dropped from 0.8788 to 0.2639, highlighting the differences.
2. Pediatric data is scarce.

In this paper, the DA-nnUNet -built on the standard nnUNet architecture, but with a domain classifier with a Gradient Reversal Layer (GRL) added- is proposed. This architecture allows domain-adversarial training so that the shared encoder learns domain-invariant features. The GRL strength \( \alpha \) is initialized at 0 and gradually increased.

The proposed model is compared with the following 8 models:

1. Adult‑only training (nnUNet, BraTS adult)
2. Pediatric‑only training(nnUNet, BraTS‑PEDs)
3. Combined training on adult + pediatric data (ideal upper bound)
4. Pre‑train adult → fine‑tune pediatric
5. Freeze backbone, retrain head on pediatric
6. Same as 5 plus extra fine‑tuning
7. Freeze encoder, fine‑tune rest on pediatric
8. Freeze decoder, fine‑tune rest on pediatric

The biggest performance gain of DA-nnUNet is observed in the TC region.
