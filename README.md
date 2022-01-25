# SSVMwithBoundedErrorRates

## What does each folder mean?
### 1. R: 
A list of source code. In particular
#### 1.1. MCSVMwC_nonconvex_cplex_robust: This is the code for the proposed SSVM method used in the paper. Within the code, there are
1) Training function mcsvmwc.qp.lp
2) Inference function predict.mcsvmwc
3) Tuning function mcsvmwc_tune and mcsvmwc_tune_robust where mcsvmwc_tune_robust uses a separate data set to calculate the threshold for acceptance region.

#### 1.2. MCSVMwC_nonconvex_robust: This is the code for the proposed SSVM method. However, this code uses R package 'quadprog' to solve the quadratic programming.

Rest codes are used to generate benchmark results:
#### 1.3. lr: 
The code with 'lr' utilize linear logistic regression to find the acceptance region.
#### 1.4. knn: 
The code with 'knn' utilize kNN model to find the acceptance region.
#### 1.5. klr: 
The code with 'klr' utilize kernel logistic regression model to find the acceptance region.
#### 1.6. svm: 
The code with 'svm' utilize concersional svm or msvm to find the acceptance region.
#### 1.7. onevall: 
The code with 'onevall' use one-verses-all classification to find the regression function which will be plug-in to the Bayes Rule to find the acceptance region. The tuning strategy is to minimize the ambiguity.
#### 1.8. tunebyclass: 
The code with 'onevall' use one-verses-all classification to find the regression function which will be plug-in to the Bayes Rule to find the acceptance region, but the tuning strategy is to minimize the classification error. 

### 2. Simulation_* 
There is one folder for each real data scenario. In each folder, there are two files. There is an R file to replicate the comparison between the proposed SSVM method and all the benchmarks. It should be able to run end-to-end. In addition, there is a data set file used for the study. For the first three scenarios, there is another R file with the name "_with_reject_option", this is used to replicate the study for Section 5.2.

### 3. Realdata_* 
There is one folder for each real data scenario. In each folder, there is an R file to replicate the comparison between the proposed SSVM method and all the benchmarks. It should be able to run end-to-end. In addition, there is a data set file used for study.  


## What is the dependency of the codes? 
MCSVMwC_nonconvex_cplex_robust depends on Cplex which is a solver for quadratic programming. 
