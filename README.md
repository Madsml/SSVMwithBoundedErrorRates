# Set-Valued Support Vector Machine withBounded Error Rates

## Data
### Simulated data
We consider three different simulation scenarios in the main paper. In the first scenario we compare the linear approaches, while in the next twoscenarios we consider nonlinear methods. We have an additional simulation scenario for large number of classes in supplimentary materials in which we compare linear models. The simulation generation codes are in
* Scenario 1: ./JASAcode/simulation_linear
* Scenario 2: ./JASAcode/simulation_kernelgenerator
* Scenario 3: ./JASAcode/simulation_donuts
* Scenario 4: ./JASAcode/simulation_large_class

More detailed discription can be found in Section 5.1 Simulations in the main paper (for Scenario 1-3) and supplimentory material (for Scenario 4).

### Real data
The data sets we used for study are all publicly availalbe and all of them can be found in UCI repository. (https://archive.ics.uci.edu/ml/index.php). 

* The CNAE-9 data contains 1080 documents of free text business descriptions of Brazilian companies from 9 categories. Each document was represented as a vector, where the weight of each word is its frequency in the document. The data set we used is in ./JASAcode/realdata_cnae/cnae_data.

* The well-known hand-written zip code data  (LeCun et al., 1989) is widely used in the classification literature. The original data set consists of 9298 16 x 16 (hence 256 predictos) pixel images of handwritten digits. There are both training and test sets defined in it. Following Lei (2014), Wang and Qiao (2018), we only use a subset of the data containing digits 0,6,8,9 for illustration purpose. The data set we used is in ./JASAcode/realdata_zipcode/zipcode.r.

*  The Vehicle data set (Siebert, 1987) can be found in the UCI Machine Learning Repository. It is a four-class multicategory classification task with 946 observations and 18 predictors in total. We discriminate between silhouettes of model cars, vans and buses. The data set we used is in ./JASAcode/realdata_vehicle/vehicle_data.

More detailed discription can be found in Section 5.3 Real Data Analysis in the main paper.

## Code

### 1. ./JASAcode/R/: 
The folder contains a list of source code.

#### 1.1. ./JASAcode/R/MCSVMwC_nonconvex_cplex_robust: 
This is the code for the proposed SSVM method used in the paper. Within the code, there are
1) Training function mcsvmwc.qp.lp
2) Inference function predict.mcsvmwc
3) Tuning function mcsvmwc_tune and mcsvmwc_tune_robust where mcsvmwc_tune_robust uses a separate data set to calculate the threshold for acceptance region.

#### 1.2. ./JASAcode/R/MCSVMwC_nonconvex_robust: 
This is the code for the proposed SSVM method. However, this code uses R package 'quadprog' to solve the quadratic programming. However, it is worth to note that the 'quadprog' is generally less stable than Cplex hence one may experience exceptions such as 'no solution for quadratic programming'.

#### 1.3. R Code with lr in name: 
The code with 'lr' utilize linear logistic regression to find the acceptance region.
#### 1.4. R Code with knn in name: 
The code with 'knn' utilize kNN model to find the acceptance region.
#### 1.5. R Code with klr in name: 
The code with 'klr' utilize kernel logistic regression model to find the acceptance region.
#### 1.6. R Code with svm in name: 
The code with 'svm' utilize concersional svm or msvm to find the acceptance region.
#### 1.7. R Code with onevall in name: 
The code with 'onevall' use one-verses-all classification to find the regression function which will be plug-in to the Bayes Rule to find the acceptance region. The tuning strategy is to minimize the ambiguity. 
#### 1.8. R Code with tunebyclass in name: 
The code with 'onevall' use one-verses-all classification to find the regression function which will be plug-in to the Bayes Rule to find the acceptance region, but the tuning strategy is to minimize the classification error. Not reported in the paper but can be used as a reference.

### 2. ./JASAcode/simulation_* 
There is one folder for each simulation data scenario. In each folder, there are two R code files. 

* ./JASAcode/simulation_*: This R file is used to replicate the comparison between the proposed SSVM method and all the benchmarks. It is able to run end-to-end. One should be able to replicate the siomulation result in paper if you specify seed = 0,1,...,99.

* ./JASAcode/simulation_*_with_reject_option: For the first three scenarios, there is another R file used to replicate the study for Section 5.2 and compare with classification with reject options proposed in (Zhang et al., 2017) .

### 3. ./JASAcode/realdata_* 
There is one folder for each real data scenario. In each folder, there is an R file to replicate the comparison between the proposed SSVM method and all the benchmarks. It should be able to run end-to-end. In addition, there is a data set used for the study. One should be able to replicate the siomulation result in paper if you specify seed = 0,1,...,99.

## Dependencies
* ./JASAcode/R/MCSVMwC_nonconvex_cplex_robust depends on IBM Cplex (https://www.ibm.com/products/ilog-cplex-optimization-studio) which is a solver for quadratic programming. 
* ./JASAcode/R/mcwc_klr depends on source code used in Menget al. (2020). We include the code with author's concent.
* ./JASAcode/R/mcwc_svm depends on source code used in Lee et al. (2004). The code can be found in (https://www.asc.ohio-state.edu/lee.2272/software.html).
