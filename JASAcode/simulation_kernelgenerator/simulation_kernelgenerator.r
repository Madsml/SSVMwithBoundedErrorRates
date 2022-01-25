####################################################################
# set directory and function
####################################################################
# source packages and solvers
source("./../R/MCSVMwC_nonconvex_cplex_robust.R")
source("./../R/mcwc_lr.R")
source("./../R/mcwc_klr.R")
source("./../R/mcwc_knn.R")
source("./../R/mcwc_rf.R")
source("./../R/mcwc_svm.R")
source("./../R/mcwc_lr_onevsall.R")
source("./../R/mcwc_klr_onevsall.R")
source("./../R/mcwc_rf_onevsall.R")
source("./../R/mcwc_svm_onevsall.R")
source("./../R/mcwc_lr_tunebyclass.R")
source("./../R/mcwc_klr_tunebyclass.R")
source("./../R/mcwc_knn_tunebyclass.R")
source("./../R/mcwc_rf_tunebyclass.R")
source("./../R/mcwc_svm_tunebyclass.R")
library(mvtnorm)
# set seed 
procid = 0
seed = 123*procid
set.seed(seed)
###########################################################################################################################
# data generation code
###########################################################################################################################
kernel_generator = function(n,p_all,p_noise) {
  # consider a rotation matrix
  rotation = matrix(c(1/2,sqrt(3)/2,-sqrt(3)/2,1/2),nrow = 2,ncol = 2)
  rotation2 = rotation %*% rotation
  # generate four two dimention uniform distributed box first
  # start with a ball
  #X1
  theta = runif(n,min = 0,max=2*pi)
  r = 2/3*sqrt(runif(n,min = 0,max = 1))
  X1 = cbind(1+r*cos(theta),r*sin(theta))
  #X2
  theta = runif(n,min = 0,max=2*pi)
  r = 2/3*sqrt(runif(n,min = 0,max = 1))
  X2 = cbind(1+r*cos(theta),r*sin(theta)) %*% rotation2
  #X3
  theta = runif(n,min = 0,max=2*pi)
  r = 2/3*sqrt(runif(n,min = 0,max = 1))
  X3 = cbind(1+r*cos(theta),r*sin(theta)) %*% rotation2 %*% rotation2
  # cbind X1, X2, X3 to get the final data
  X = rbind(X1,X2,X3)
  n_con = 0.1 * n
  Y = c(rep(1,n-n_con),rep(2,n),rep(3,n),rep(1,n_con))
  X = rbind(X[Y==1,],X[Y==2,],X[Y==3,])
  Y = c(rep(1,n),rep(2,n),rep(3,n))
  # add some noise to the data
  mu_noise = rep(0,p_noise)
  sigma_noise = diag(rep((1/p_all),p_noise))
  x_noise = rmvnorm((3*n),mu_noise,sigma_noise)
  # combine signal and noise
  X = as.matrix(cbind(X,x_noise))
  return (list(X,Y))
}
###########################################################################################################################
# simulation
###########################################################################################################################
# first lets state the setting of this simulation
n_pool = 20*(1:5)
for (n in n_pool) {
    n = 20
    set.seed(seed)
    p_all = 100
    k = 3
    r = rep(0.05,k)
    # give a name to the model
    filename=paste("./full/p=",p_all,"_n=",n,"/simulation_seed_",seed,"_p=",p_all,"_n=",n,".csv",sep="")
    # get start
    p = 2
    p_noise = p_all - p
    # get the training data
    n_train = n
    train = kernel_generator(n=n_train,p_all=p_all,p_noise=p_noise)
    x_train = train[[1]]
    y_train = train[[2]]
    # get the tuning data
    n_tune = n
    tune = kernel_generator(n=n_tune,p_all=p_all,p_noise=p_noise)
    x_tune = tune[[1]]
    y_tune = tune[[2]]
    # get the threshold data
    x_threshold = x_tune
    y_threshold = y_tune 
    # get the testing data
    n_test = 10000
    test = kernel_generator(n=n_test,p_all=p_all,p_noise=p_noise)
    x_test = test[[1]]
    y_test = test[[2]]    
    # add the same noise to the testing data
    col_name = rep(0,p_all)
    for (p_index in 1:p_all) {
      col_name[p_index] = paste('V',as.character(p_index),sep="")
    }
    colnames(x_train) <- col_name
    colnames(x_threshold) <- col_name
    colnames(x_tune) <- colnames(x_train)
    colnames(x_test) <- colnames(x_train)
    # delete row name
    rownames(x_train) <- NULL
    rownames(x_threshold) <- NULL
    rownames(x_tune) <- NULL
    rownames(x_test) <- NULL
    ###########################################################################################################################
    # train the model
    ###########################################################################################################################    
    # tuning the model
    model_robust = mcsvmwc_tune_robust(x_train=x_train,y_train=y_train,x_tune=x_tune,y_tune=y_tune,x_threshold=x_threshold,y_threshold=y_threshold,r=r,kernel='radial',lambdapool=10^((-8:4)/2),max.iteration = 5,max.violation = 1,rho_pool=10^((-2:2)/4),degree_pool=1)
    Y_predict_robust = predict.mcsvmwc(model = model_robust, X_test = x_test, Y_test = y_test)
    print(c(Y_predict_robust$`ambiguity proportion`,Y_predict_robust$control))
    print(Y_predict_robust$counter)
    index_list = list()
    allowance = rep(1,length(r))
    for (i in 1:length(r)) {
       index_list = c(index_list,list(which(y_test==i)))
       allowance[i] = floor(r[i] * length(which(y_test==i)))
     }
    model_adj = model_robust
    pre_original = predict.mcsvmwc(model_adj,x_test,y_test)
    # define a new kind of threshold
    score_test = pre_original$score
    thresholds = rep(0,length(r))
    for (qq in 1:length(r)) {
      thresholds[qq] = sort(score_test[index_list[[qq]],qq],decreasing = FALSE)[allowance[qq]]
    }
    model_adj$thresholds = thresholds
    pre_adj = predict.mcsvmwc(model_adj,x_test,y_test)
    out_csvm = t(c(c(Y_predict_robust$`ambiguity proportion`,Y_predict_robust$control),c(pre_adj$`ambiguity proportion`,pre_adj$control)))
    r_match = r
    ####################################################################
    # run the logistic regression first and try to tune it a bit
    ####################################################################
    nlambda = 120
    alpha_pool = 0
    out_logit = tune_lr(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=rep(0.05,3), r_match = r_match, nlambda = nlambda, alpha_pool = alpha_pool)
    ####################################################################
    # run kernel logistic regression first and try to tune it a bit
    ####################################################################
    # tune for lambda and rho
    lambda_pool = 10^((-60:60)/20)
    rho_pool = 10^((-2:2)/4)
    degree_pool = 1
    out_klr = tune_klr(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r,kernel='radial',r_match = r_match,degree_pool = degree_pool, rho_pool = rho_pool,lambda_pool = lambda_pool)[[3]]
    out_klr_ova = tune_klr_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r,kernel='radial',r_match = r_match,degree_pool = degree_pool, rho_pool = rho_pool,lambda_pool = lambda_pool)[[3]]
    out_klr_tbc = tune_klr_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r,kernel='radial',r_match = r_match,degree_pool = degree_pool, rho_pool = rho_pool,lambda_pool = lambda_pool)[[3]]
    ####################################################################
    # estimate probability using knn
    ####################################################################
    k_pool = seq(6,40,by=2)
    out_knn = tune_knn(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r, r_match = r_match,k_pool = k_pool)[[3]]
    out_knn_tbc = tune_knn_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r, r_match = r_match,k_pool = k_pool)[[3]]
    ####################################################################
    # estimate probability using random forest
    ####################################################################
    ntree_pool = seq(50,500,by=50)
    mtry_pool=c(0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8)
    out_rf = tune_rf(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r, r_match = r_match,ntree_pool=ntree_pool,mtry_pool=mtry_pool)[[3]]
    out_rf_ova = tune_rf_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r, r_match = r_match,ntree_pool=ntree_pool,mtry_pool=mtry_pool)[[3]]
    out_rf_tbc = tune_rf_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r, r_match = r_match,ntree_pool=ntree_pool,mtry_pool=mtry_pool)[[3]]
    ####################################################################
    # extend traditional svm to classification with confidence
    ####################################################################
    rho_pool = 10^((-2:2)/4)
    gamma_pool = 1/(rho_pool^2)
    cost_pool = 10^((-80:20)/20)
    degree_pool = 1
    out_svm = tune_svm(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='radial',degree_pool=degree_pool,rho_pool = rho_pool)[[3]]
    out_svm_ova = tune_svm_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='radial',degree_pool=degree_pool,gamma_pool = gamma_pool)[[3]]
    out_svm_tbc = tune_svm_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='radial',degree_pool=degree_pool,gamma_pool = gamma_pool)[[3]]
    # combine all the outcomes
    colnames(out_csvm) = colnames(out_svm)
    rownames(out_csvm) = c('CSVM')
    out_matrix = rbind(out_csvm,out_klr,out_klr_ova,out_klr_tbc,out_knn,out_knn_tbc,out_rf,out_rf_ova,out_rf_tbc,out_svm,out_svm_ova,out_svm_tbc)
    write.csv(out_matrix,file=filename,row.names = FALSE)
  }