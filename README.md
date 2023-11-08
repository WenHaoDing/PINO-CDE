# PINO-MBD

![PINO-MBD Diagram](docs/Fig1.jpg)

**Recovered displacement field (left) and ground truth (right)**
<img src="docs/Displacement Field.gif" alt="Displacement field for the toy example" width="720" height="250"/>

**Recovered velocity field (left) and ground truth (right)**
<img src="docs/Velocity Field.gif" alt="Velocity field for the toy example" width="720" height="250"/>

**Recovered acceleration field (left) and ground truth (right)**
<img src="docs/Acceleration Field.gif" alt="Acceleration field for the toy example" width="720" height="250"/>

**Solving practical multi-body dynamics problems using a single neural operator**

Abstract: *As a fundamental design tool in many engineering disciplines, multi-body dynamics (MBD) models a complex structure with a differential equation group containing multiple physical quantities. Engineers must yconstantly adjust structures at the design stage, which requires a highly efficient solver. The rise of deep learning technologies has offered new perspectives on MBD. Unfortunately, existing black-box models suffer from poor accuracy and robustness, while the advanced methodologies of single-output operator regression cannot deal with multiple quantities in MBD. To address these challanges, we propose PINO-MBD, a deep learning framework for solving practical MBD problems based on the theory of physics-informed neural operator (PINO). PINO-MBD uses a single network for all quantities in a multi-body system, instead of training dozens, or even hundreds of networks as in the existing literature. We demonstrate the flexibility and feasibility of PINO-MBD for one toy example and two practical applications: vehicle-track coupled dynamics (VTCD) and reliability analysis of a four-storey building. The performance of VTCD indicates that our framework outperforms existing software and machine learning-based methods in terms of efficiency and precision, respectively. For the g reliability analysis, PINO-MBD can provide higher-resolution results in less than a quarter of the time incurred when using the probability density evolution method (PDEM). This framework integrates mechanics and deep learning technologies, and may reveal a new concept for MBD and probabilistic engineering.*
## Requirements
- Pytorch 1.8.0 or later
- wandb
- tqdm
- scipy
- h5py
- numpy
- Matlab R2016b or later

## Data description
### The toy example
- Train set: `5000V3.mat`
- Test set: `500V3.mat`
- Virtual dataset: `VirtualData_8000_V3.mat`
- Weights with EN: `Weights_Medium_5000_V3.mat`
- Weights without EN: `Weights_PINO.mat`
- Weights for the test set: `Weights_Medium_500_V3.mat`
- Weights for the virtual dataset: `Weights_Virtual_V3.mat`
- Structure information (including mode shape functions): `StructureInfo.mat`

### Vehicle-track coupled dynamics (VTCD)
- Train set: `10000V2.mat`
- Test set: `2000V2.mat`
- Virtual dataset: `VirtualData_10000V2.mat`
- Weights with EN: `Weights_10000V2.mat`
- Weights without EN: `Weights_PINO_10000V2.mat`
- Weights for the test set: `Weights_2000V2.mat`
- Weights for the virtual dataset: `Weights_Virtual.mat`

### Reliability assessment for a 4-storey building
- Train set: `150.mat`
- Test set: `30.mat`
- Virtual dataset: `VirtualData_1500.mat`
- Weights with EN: `Weights_Medium_150.mat`
- Weights without EN: `Weights_PINO_150.mat`
- Weights for the test set: `Weights_Medium_30.mat`
- Weights for the virtual dataset: `Weights_Virtual.mat`

## General instruction
In general, using this package to implement PINO-MBD includes several steps in the following flowchart. Among them, the numerical integration implemented by Matlab codes is used to generate the training and testing dataset. The training of PINO-MBD and the solution of MBD are done in the python environment. There are usually three goals for postprocessing: recovering solution (and derivatives) fields, recovering stress fields, and doing reliability assessments. All three goals need to be implemented in the Matlab environment, and all have corresponding visualization codes. It should be noted that achieving these three goals requires the use of the mode shape functions of the structure, as well as the element shape information. We first established the model in ABAQUS and meshed it, and then imported it into ANSYS through hypermesh to obtain the mode shape functions and element shape information. We provide a command stream to export these data in ANSYS, but the model file is not provided due to size limitations.
~~~mermaid
graph LR
	A[A: Generate datasets] --> B[B: Train the PINO-MBD]
	A --> C[C: Generate EN weights]
	C --> B
	B --> D[D: Solve MBDs] --> E[E: Post processing]
	E --> F[F: Recover solution fields]
	E --> G[G: Recover stress fields]
	E --> H[H: Reliability assessment]
~~~
## Detailed instructions for your PINO-MBD
### The toy example
#### section I: generate mode shape functions
```bash
# Step1: establish the model in ABAQUS and mesh it
ABAQUS FEM files\HM\Project2.inp
# Step2: import the model into ANSYS through Hypermesh
# Step3: output mode shape functions
ANSYS_APDL FEM files\HM\Orders.txt
# Step4: write down the cdb file inside ANSYS
```
#### section II: train the PINO-MBD
```bash
# Step1: Generate the training and testing data
Matlab Matlab codes/HM/Generator_Master.m
# Step2: Assemble the train and test datasets (normalize the data at the same time)
Matlab Matlab codes/HM/Dataset_Generation.m
# Step3: Compute the equation normalization (EN) weights using the datasets
Matlab Matlab codes/HM/weights_computation.m
# Step4: Generate the virtual dataset (which needs the normalization information in Step2)
Matlab Matlab codes/HM/Generator_Virtual.m
# Step5: Train the PINO-MBD
python3 Runner_HM_WithVirtual.py --config_path configs/HM/HM_PINO-MBD.yaml --mode train 
# Step6: Solution fields and visualization
Matlab Matlab codes/HM/Visualization_Animation.m
```
### VTCD
#### section I: train the PINO-MBD
```bash
# Step1: Generate the training and testing data
Matlab Matlab codes/VTCD/Generator_Master.m
# Step2: Assemble the train and test datasets (normalize the data at the same time)
Matlab Matlab codes/VTCD/Dataset_Generation.m
# Step3: Compute the equation normalization (EN) weights using the datasets
Matlab Matlab codes/VTCD/weightsComputation.m
# Step4: Generate the virtual dataset (which needs the normalization information in Step2)
Matlab Matlab codes/VTCD/Data_Generator_Virtual.m
# Step5: Train the PINO-MBD
python3 Runner_VTCD_GradNorm.py --config_path configs/VTCD/VTCD.yaml --mode train
```
### The reliability assessment for the 4-storey building
#### section I: generate mode shape functions
```bash
# Step1: establish the model in ABAQUS and mesh it
ABAQUS FEM files\BSA\Building.inp
# Step2: import the model into ANSYS through Hypermesh
# Step3: output mode shape functions
ANSYS_APDL FEM files\BSA\Orders.txt
# Step4: write down the cdb file inside ANSYS
```
#### section II: train the PINO-MBD
```bash
# Step1: Generate the training and testing data
Matlab Matlab codes/BSA/Generator_Master.m
# Step2: Assemble the train and test datasets (normalize the data at the same time)
Matlab Matlab codes/BSA/Dataset_Generation.m
# Step3: Compute the equation normalization (EN) weights using the datasets
Matlab Matlab codes/BSA/Weights_computation.m
# Step4: Generate the virtual dataset (which needs the normalization information in Step2)
Matlab Matlab codes/BSA/Generator_VirtualV2.m
# Step5: Train the PINO-MBD
python3 Runner_BSA_GradNorm.py --config_path configs/BSA/BSA.yaml --mode train 
# Step6: Solution fields and visualization
Matlab Matlab codes/BSA/Visualization_Animation.m
```
#### section III: perform the probability density evolution
```bash
Matlab PDEM/Evolution_Stress.m
```
#### section IV: statistically obtain probability field and compare with PDEM
```bash
# Step1: Generate a large amount of predictions with the trained PINO-MBD
python3 Runner_BSA_GradNorm.py --config_path configs/BSA/BSA.yaml --mode eval_batch 
# Step2: statistically obtain probability field and maximum stree field through Tensor Operations 
python3 post processing/Figure_RA.py
# Step3: Compare with PDEM in Matlab environment
Matlab PDEM/post.m
```
## Reproduce figures in the author's paper and dissertation
Introduction: *This part of the readme file is created for recreating the figures in the author's paper and dissertation.*

**Readers are welcome to use the package as they like. Citation links are provided below. At the same time, the author would like emphasize that PINO-MBD was proposed based on the theory of PINO[3]. The author would like to thank Mr. Zongyi Li and his team members.**

> [1] Ding, W., He, Q., Tong, H., Wang, Q. & Wang, P. Solving practical multi-body dynamics
problems using a single neural operator. Preprint at http://arxiv.org/abs/2210.00222 (2022).
> 
> [2] 丁文灏. 通用多体动力学神经元算子及其在车辆-轨道耦合动力学中的应用研究[D] (RESEARCH ON GENERAL NEURAL
OPERATOR FOR MULTIBODY DYNAMICS AND ITS APPLCIATIONS IN VEHICLETRACK COUPLED DYNAMICS). (2023)
> 
> [3] Li Z et al.Physics-Informed Neural Operator for Learning Partial Differential
Equations.Preprint at https://arxiv.org/pdf/2111.03794.pdf,2021.

Guide for reproducing figures in the online paper: *Plot codes for the online paper are all contained in the dossier named 'plot_paper'. In there you will each part of each figure has its own codes. Before using these codes, you will need to train your PINO-MBD as instructed above.*

Guide for reproducing figures in the author's dissertation. *Plot codes for the dissertation are all contained in the dossier named 'plot_dissertation'. In there you will each part of each figure has its own codes. Before using these codes, you will need to train your PINO-MBD as instructed above.*


## Datasets

Due to the GitHub's file size limit, we have uploaded the datasets for all three examples to Zendo. Users can refer to the following links:

> [1] W. Ding, Q. He, S. J. University, P. Wang, Q. Wang, Solving Coupled Differential Equation Groups Using PINO-CDE (Toy
Example) (Jun. 2023). doi:10.5281/zenodo.8398559.
> 
> [2] W. Ding, Q. He, H. Tong, P. Wang, Q. Wang, Solving Coupled Differential Equation Groups Using PINO-CDE (VTCD) (Jun.
2023). doi:10.5281/zenodo.8398835.
> 
> [3] W. Ding, Q. He, H. Tong, P. Wang, Q. Wang, Solving Coupled Differential Equation Groups Using PINO-CDE (Uncertainty
Propagation) (Jun. 2023). doi:10.5281/zenodo.8414398.
> 