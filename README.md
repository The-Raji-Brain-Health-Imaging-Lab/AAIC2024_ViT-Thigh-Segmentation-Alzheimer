## Automated Thigh Tissue Segmentation by Vision Transformers: Application for Early Alzheimer's Diagnosis

<p align="justify"> This repository includes the code and models used in our study on the application of Vision Transformers for automated thigh tissue segmentation to aid in the early diagnosis of Alzheimer's disease. This work is part of our research accepted for poster presentation at the <b><span style="color:purple;">Alzheimer's Association International Conference (AAIC)</span></b> 2024. </p>

-----

### Automated Thigh Tissue Segmentation by Vision Transformers: Application for Early Alzheimer's Diagnosis

<sub><b> Sara Hosseinzadeh Kasani, PhD<sup>1</sup>, </b>
Mahsa Dolatshahi, MD-MPH<sup>1</sup>,
Mahshid Naghashzadeh, MSc<sup>1</sup>, 
Paul K. Commean, BEE<sup>1</sup>, 
Farzaneh Rahmani, MD-MPH<sup>1</sup>,
Jingxia Liu, PhD<sup>2</sup>,
LaKisha Lloyd<sup>1</sup>,
Caitlyn Nguyen, BSc<sup>1</sup>,
Nancy Hantler<sup>1</sup>,
Abby McBee-Kemper<sup>1</sup>,
Maria Ly, MD, PhD<sup>1</sup>,
Gary Z Yu, MD, PhD<sup>1</sup>,
Joseph E. Ippolito, MD, PhD<sup>3</sup>,
Bettina Mittendorfer, PhD<sup>4</sup>,
Claude Sirlin, MD<sup>5</sup>,
John C. Morris, MD<sup>6,7</sup>,
Tammie L.S. Benzinger, MD, PhD<sup>1,3,6</sup>,
and Cyrus A. Raji, MD, PhD<sup>1,2</sup>,
(1) Mallinckrodt Institute of Radiology, Washington University in St. Louis, St. Louis, MO, USA, 
(2) Washington University in St. Louis School of Medicine, St. Louis, MO, USA, 
(3) Washington University in St. Louis, St. Louis, MO, USA, 
(4) Missouri University School of Medicine, Columbia, MO, USA, 
(5) University of California, San Diego, La Jolla, CA, USA, 
(6) Knight Alzheimer Disease Research Center, St. Louis, MO, USA, 
(7) Washington University School of Medicine in St. Louis, St. Louis, MO, USA
</sub>

<p align="justify"><b>Background:</b> Within the research field of neurodegenerative disorders, unbiased analysis of body fat composition, particularly muscle mass, is gaining attention as a potential biological marker for refining Alzheimer’s disease risk. The objective of this study was to employ a deep learning model for fully automated and accurate segmentation of thigh tissues, potentially contributing to early Alzheimer's diagnostics. </p>

<p align="justify"><b>Methods:</b> In an IRB-approved study, 49 participants underwent thigh Dixon MRI scans with a TR=9.99s, TE=2.46s, flip angle=10°, and slice thickness= 5mm. The Dixon Fat/Water images were semi-automatically segmented by an expert operator in all available slices to obtain the bone, intermuscular fat (InterFat), intramuscular fat (IntraFat), Subcutaneous Adipose Tissue (SAT), Muscle, and Gluteus. We trained and compared the performance of baseline and state-of-the-art deep neural networks, namely, UNet, VNet, and two vision transformers (ViTs): UNETR and SwinUNETR. The performance of the trained models was tested on all data sets using a 3-fold cross-validation scheme.</p>

<p align="justify"><b>Results:</b> We found SwinUNETR outperformed the others with a mean dice similarity coefficient 96.20 (± 0.51), 80.91 (± 0.55), 50.56 (± 1.43), 95.26 (± 0.80), 98.70), 86.72 (± 1.12) in Bone, InterFat, IntraFat, SAT, Muscle, and Gluteus, respectively. Bland–Altman analysis and scatter plot (Fig 1) indicated that the differences between manual annotations and predictions by the SwinUNETR model were relatively minor for Bone volume, Intramuscular Fat volume, Muscle volume, and Gluteus volume classes. The overall mean difference is -88.8cm3 with a 95% confidence interval (CI) of [-159.53, -18.13]. Biases [95% CI] for each tissue class were 5.44cm3 [−8.61, 19.50] for Bone volume, 51.74cm3 [−22.72, 126.20] for InterFat volume, 11.15cm3 [2.76, 19.53] for IntraFat volume, -191.09cm3 [−309.54, -72.64] for SAT volume, 15.16cm3 [−6.37, 36.70] for Muscle volume, and 18.76cm3 [2.20, 35.32] for Gluteus volume. </p>

<p align="justify"><b>Conclusions:</b> This study highlighted the use of ViTs for the automated segmentation of thigh tissues in MR which may allow for the detection of subtle changes in muscle mass and fat composition, that are of increasing interests in their associations with the neurodegenerative processes in Alzheimer's disease.</p>
