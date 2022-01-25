####################################################################
# set directory and function
####################################################################
# source packages and solvers
source("./../R/new/MCSVMwC_nonconvex_cplex_robust.R")
source("./../R/new/mcwc_lr.R")
source("./../R/new/mcwc_lr_onevsall.R")
source("./../R/new/mcwc_lr_tunebyclass.R")
source("./../R/new/mcwc_svm.R")
source("./../R/new/mcwc_svm_onevsall.R")
source("./../R/new/mcwc_svm_tunebyclass.R")
library(mvtnorm)
# set seed 
procid = 0
seed = 123*procid
set.seed(seed)
###########################################################################################################################
# data generation
###########################################################################################################################
# first lets state the setting of this simulation
n_pool = 20*(1:5)
p_pool = c(100)
for (n in n_pool) {
  for (jj in 1:length(p_pool)) {
    set.seed(seed)
    p_all = 10
    r = rep(0.05,3)
    # give a name to the model
    filename=paste("./full/p=",p_all,"_n=",n,"/simulation_seed_",seed,"_p=",p_all,"_n=",n,".csv",sep="")
    # get start
    p = 2
    p_noise = p_all - p
    k = 3
    # generate three Gussian distributions
    means = simplex(3)
    # first class
    dir1 = (means[,1] - means[,2])/sqrt(sum((means[,1] - means[,2])*(means[,1] - means[,2])))
    s1 = matrix(c(dir1,-dir1[2],dir1[1]),nrow=2,ncol=2)
    sigma1  = s1%*%diag(c(1,0.2))%*%t(s1)
    # second class
    dir2 = (means[,2] - means[,3])/sqrt(sum((means[,2] - means[,3])*(means[,2] - means[,3])))
    s2 = matrix(c(dir2,-dir2[2],dir2[1]),nrow=2,ncol=2)
    sigma2  = s2%*%diag(c(1,0.2))%*%t(s2)
    # third class
    dir3 = (means[,3] - means[,1])/sqrt(sum((means[,3] - means[,1])*(means[,3] - means[,1])))
    s3 = matrix(c(dir3,-dir3[2],dir3[1]),nrow=2,ncol=2)
    sigma3  = s3%*%diag(c(1,0.2))%*%t(s3)
    # add some noise to the data
    mu_noise = rep(0,p_noise)
    sigma_noise = diag(rep(0.01,p_noise))
    # training set
    x_train1 = rmvnorm(n, sigma = sigma1, mean = means[,1])
    y_train1 = rep(1,n)
    x_train2 = rmvnorm(n, sigma = sigma2, mean = means[,2])
    y_train2 = rep(2,n)
    x_train3 = rmvnorm(n, sigma = sigma3, mean = means[,3])
    y_train3 = rep(3,n)
    # finally generate the full dataset
    #x_train = rbind(x_train1,x_train2,x_train3)
    x_train = cbind(rbind(x_train1,x_train2,x_train3),rmvnorm(3*n, sigma = sigma_noise, mean = mu_noise))
    y_train = c(y_train1,y_train2,y_train3)
    # tuning set
    n_tune = n
    x_tune1 = rmvnorm(n_tune, sigma = sigma1, mean = means[,1])
    y_tune1 = rep(1,n_tune)
    x_tune2 = rmvnorm(n_tune, sigma = sigma2, mean = means[,2])
    y_tune2 = rep(2,n_tune)
    x_tune3 = rmvnorm(n_tune, sigma = sigma3, mean = means[,3])
    y_tune3 = rep(3,n_tune)
    x_tune = cbind(rbind(x_tune1,x_tune2,x_tune3),rmvnorm(3*n_tune, sigma = sigma_noise, mean = mu_noise))
    y_tune = c(y_tune1,y_tune2,y_tune3)
    x_threshold = x_tune
    y_threshold = y_tune
    # testing set
    n_test = 10000
    x_test1 = rmvnorm(n_test, sigma = sigma1, mean = means[,1])
    y_test1 = rep(1,n_test)
    x_test2 = rmvnorm(n_test, sigma = sigma2, mean = means[,2])
    y_test2 = rep(2,n_test)
    x_test3 = rmvnorm(n_test, sigma = sigma3, mean = means[,3])
    y_test3 = rep(3,n_test)
    x_test = cbind(rbind(x_test1,x_test2,x_test3),rmvnorm(3*n_test, sigma = sigma_noise, mean = mu_noise))
    y_test = c(y_test1,y_test2,y_test3)
    ###########################################################################################################################
    # train the model
    ###########################################################################################################################
    # tuning the model
    model_robust = mcsvmwc_tune_robust(x_train=x_train,y_train=y_train,x_tune=x_tune,y_tune=y_tune,x_threshold=x_threshold,y_threshold=y_threshold,r=r,kernel='linear',lambdapool=10^((-8:4)/2),max.iteration = 5,max.violation = 1,rho=1,degree=1)
    Y_predict_robust = predict.mcsvmwc(model = model_robust, X_test = x_test, Y_test = y_test)
    print(c(Y_predict_robust$`ambiguity proportion`,Y_predict_robust$control))
    print(Y_predict_robust$counter)
    # here is a new model from changing the threshold of the original msvmwc
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
    # simply save the outcome
    out_csvm = pre_adj$prediction_matrix
    out_csvm_adj = t(c(c(Y_predict$`ambiguity proportion`,Y_predict$control),c(pre_adj$`ambiguity proportion`,pre_adj$control)))
    r_match = r
    ####################################################################
    # run the logistic regression first and try to tune it a bit
    ####################################################################
    nlambda = 120
    alpha_pool = 0
    lambda_pool = 10^((-60:60)/20)
    out_logit = tune_lr(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r, r_match = r_match, nlambda = nlambda, alpha_pool = alpha_pool)[[1]]
    out_logit_ova = tune_lr_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r, r_match = r_match, lambda_pool = lambda_pool, alpha_pool = alpha_pool)[[1]]
    out_logit_tbc = tune_lr_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r, r_match = r_match, nlambda = nlambda, alpha_pool = alpha_pool)[[1]]
    ####################################################################
    # extend traditional svm to classification with confidence
    ####################################################################
    rho_pool = 1
    gamma_pool = 1/(rho_pool^2)
    cost_pool = 10^((-80:20)/20)
    degree_pool = 1
    out_svm = tune_svm(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='linear',degree_pool=degree_pool,rho_pool = rho_pool)[[1]]
    out_svm_ova = tune_svm_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='linear',degree_pool=degree_pool,gamma_pool = gamma_pool)[[1]]
    out_svm_tbc = tune_svm_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune,x_threshold=x_threshold,y_threshold=y_threshold, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='linear',degree_pool=degree_pool,gamma_pool = gamma_pool)[[1]]
    # combine all the outcomes
    colnames(out_csvm) = colnames(out_svm)
    rownames(out_csvm) = c('CSVM')
    out_matrix = cbind(y_test,out_csvm,out_logit,out_logit_ova,out_logit_tbc,out_svm,out_svm_ova,out_svm_tbc)
    write.csv(out_matrix,file=filename,row.names = FALSE)
  }
}