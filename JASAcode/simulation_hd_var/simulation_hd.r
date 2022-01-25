# source packages and solvers
source("./../R/new/MCSVMwC_nonconvex_cplex_robust.R")
source("./../R/new/mcwc_lr.R")
source("./../R/new/mcwc_lr_onevsall.R")
source("./../R/new/mcwc_lr_tunebyclass.R")
source("require./../R/new/mcwc_svm.R")
source("./../R/new/mcwc_svm_onevsall.R")
source("./../R/new/mcwc_svm_tunebyclass.R")
library(mvtnorm)
# generate a normal and distort it a bit
generate_normal = function(mean1,mean2,cov_start,n) {
  dir = (mean1 - mean2)/sqrt(sum((mean1 - mean2)*(mean1 - mean2)))
  change = (dir %*% t(dir))
  sigma  = change%*%cov_start%*%t(change)
  x = rmvnorm(n, sigma = sigma, mean = mean1)
  return(x)
}
# generate a normal and distort it a bit
generate_simulation = function(k,n,cov_start) {
  X = NULL
  Y = c()
  means = simplex(k)
  for (i in 1:k) {
    # get two mean
    mean1 = means[,i]
    if (i < k) {
      mean2 = means[,i+1]
    } else {
      mean2 = means[,1]
    }
    x = generate_normal(mean1,mean2,cov_start,n)
    y = rep(i,n)
    X = rbind(X,x)
    Y = c(Y,y)
  }
  return(list(X,Y))
}
# add noise
add_noise = function(X,mu_noise,sigma_noise) {
  n = nrow(X)
  noise = rmvnorm(n, sigma = sigma_noise, mean = mu_noise)
  X = X + noise[,1:ncol(X)]
  X = cbind(X,noise[,(ncol(X)+1):ncol(noise)])
  return(X)
}
# set seed 
procid = 0
seed = 123*procid
set.seed(seed)
###########################################################################################################################
# data generation
###########################################################################################################################
# first lets state the setting of this simulation
n = 20
set.seed(seed)
# give a name to the model
# get start
p_noise = 100
k = 15
n_tune = n
n_test = 10000
r = rep(0.1,k)
filename=paste("./outcome/p=",p_noise,"_n=",n,"/simulation_seed_",seed,"_p=",p_noise,"_n=",n,".csv",sep="")
# generate three Gussian distributions
data_train = generate_simulation(k,n,cov_start=diag(rep(1,k-1)))
data_tune = generate_simulation(k,n_tune,cov_start=diag(rep(1,k-1)))
data_test = generate_simulation(k,n_test,cov_start=diag(rep(1,k-1)))
# add some noise to the data
mu_noise = rep(0,p_noise)
sigma_noise = diag(rep(0.05,p_noise))
x_train = add_noise(data_train[[1]],mu_noise,sigma_noise)
y_train = data_train[[2]]
plot(data_train[[1]],col=data_train[[2]])
x_tune = add_noise(data_tune[[1]],mu_noise,sigma_noise)
y_tune = data_tune[[2]]
x_threshold = x_tune
y_threshold = y_tune
x_test = add_noise(data_test[[1]],mu_noise,sigma_noise)
y_test = data_test[[2]]
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
out_matrix = out_csvm
out_matrix = rbind(out_csvm,out_logit,out_logit_ova,out_logit_tbc,out_svm,out_svm_ova,out_svm_tbc)
write.csv(out_matrix,file=filename,row.names = FALSE)