# Battery-Early-Capacity-Trajectory-Prediction
This repository stores the supplementary materials, code, and datasets for the deterministic prediction models presented in the case studies of our paper: "Empirical Capacity Fade Knowledge-Augmented End-to-end Learning for Battery Early Capacity-Trajectory Prediction".
1. The documents `Supplementary materials_1.docx` and `Supplementary materials_2.docx` totally consist of 58 figures (Figure S1-58) of predicted capacity trajectories from different models and datasets. These figures are too large to be placed in the paper. We use two separate files due to 25 Mb limit for each file.
2. Two publicly available datsets are used in the models: 169 LFP dataset and 48 NMC dataset. The raw data of these two datasets are  available at https://data.matr.io/1/ and https://publications.rwth-aachen.de/record/818642. We load and proprocess the original datasets
and save the relevant data in the folder `./dataset`. All models in this repository run upon the data in `./dataset`.
3. Most Elastic net-involved models are implemented using Matlab and they are presented in the folder `/.Matlab Elastic-related models`. The users need to install CVX toolbox to run the end-to-end-elastic models.
4. Other models are implemented using Python. The users can install all necessary packages by running `pip install -r requirements.txt`
5. We use absolute file paths in some scripts. Therefore, the users need to change the file paths in these scripts for successful running.
6. The models in the folder `./Sequence to Sequence Network` have to be trained on high-performance GPUs. We use Colab to train the models using Tesla P100. 
7. Hyperparameter optimization program is desgined for distributed optimization. The trial number for each distributed core is therb flexible.
8. The machine learning model training and evaluation are implemented using multiprocessing, the trial number can be set to different values depending on the device processor limits and the user's specific application.


