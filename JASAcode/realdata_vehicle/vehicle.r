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
# set seed 
procid = 0
seed = 123*procid
set.seed(seed)
filename=paste("./out/vehicle_seed_",seed,sep="")
####################################################################
# read dataset
####################################################################
load("./vehicle_data")
# ranomly assign training data to training and tuning set with a proportion of 1:1
x.1=X[Y==1,]
x.2=X[Y==2,]
x.3=X[Y==3,]
x.4=X[Y==4,]
s1=sample(1:dim(x.1)[1])
s2=sample(1:dim(x.2)[1])
s3=sample(1:dim(x.3)[1])
s4=sample(1:dim(x.4)[1])
# training set
x_train = rbind( x.1[s1[1:50],],x.2[s2[1:50],],x.3[s3[1:50],] ,x.4[s4[1:50],] )
y_train = c( rep(1,50),rep(2,50),rep(3,50), rep(4,50))
# tuning set
x_tune = rbind(x.1[s1[51:100],],x.2[s2[51:100],],x.3[s3[51:100],],x.4[s4[51:100],] )
y_tune = c( rep(1,50),rep(2,50),rep(3,50), rep(4,50))
# testing set
x_test = rbind(x.1[s1[101:dim(x.1)[1]],],x.2[s2[101:dim(x.2)[1]],],x.3[s3[101:dim(x.3)[1]],],x.4[s4[101:dim(x.4)[1]],] )
y_test = c( rep(1,(dim(x.1)[1]-100)),rep(2,(dim(x.2)[1]-100)),rep(3,(dim(x.3)[1]-100)), rep(4,(dim(x.4)[1]-100)))
####################################################################
# train the data set with mcsvm
####################################################################
# get the coverage setting
k = 4
r = rep(0.04,k)
# tuning the model
model = mcsvmwc_tune_robust(x_train=x_train,y_train=y_train,x_tune=x_tune,y_tune=y_tune,r=r,kernel='linear',lambdapool=10^((-8:4)/2),max.iteration = 5,max.violation = 1,rho_pool=1,degree_pool=1)
Y_predict = predict.mcsvmwc(model = model, X_test = x_test, Y_test = y_test)
print(c(Y_predict$`ambiguity proportion`,Y_predict$control))
print(Y_predict$counter)
# when r_match become 0
for (check in 1:length(r_match)) {
 if (r_match[check] == 0) {
   r_match[check] = r_match[check] + 1/length(which(y_test==check))
 }
}
# here is a new model from changing the threshold of the original msvmwc
index_list = list()
allowance = rep(1,length(r))
for (i in 1:length(r)) {
  index_list = c(index_list,list(which(y_test==i)))
  allowance[i] = floor(r[i] * length(which(y_test==i)))
}
model_adj = model
pre_original = predict.mcsvmwc(model_adj,x_test,y_test)
# define a new kind of threshold
score_test = pre_original$score
thresholds = rep(0,length(r))
for (jj in 1:length(r)) {
  thresholds[jj] = sort(score_test[index_list[[jj]],jj],decreasing = FALSE)[allowance[jj]]
}
model_adj$thresholds = thresholds
pre_adj = predict.mcsvmwc(model_adj,x_test,y_test)
out_csvm = t(c(c(Y_predict$`ambiguity proportion`,Y_predict$control),c(pre_adj$`ambiguity proportion`,pre_adj$control)))
r_match = r
####################################################################
# run the logistic regression first and try to tune it a bit
####################################################################
nlambda = 120
alpha_pool = 0
lambda_pool = 10^((-60:60)/20)
out_lr = tune_lr(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match, nlambda = nlambda, alpha_pool = alpha_pool)[[1]]
out_lr_ova = tune_lr_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match, lambda_pool = lambda_pool, alpha_pool = alpha_pool)[[1]]
out_lr_tbc = tune_lr_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match, nlambda = nlambda, alpha_pool = alpha_pool)[[1]]
####################################################################
# run kernel logistic regression first and try to tune it a bit
####################################################################
# tune for lambda and rho
lambda_pool = 10^((-100:20)/20)
rho_pool = 10^((-1:3)/4)
degree_pool = 1
out_klr = tune_klr(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,kernel='radial',r_match = r_match,degree_pool = degree_pool, rho_pool = rho_pool,lambda_pool = lambda_pool)[[1]]
out_klr_ova = tune_klr_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,kernel='radial',r_match = r_match,degree_pool = degree_pool, rho_pool = rho_pool,lambda_pool = lambda_pool)[[1]]
out_klr_tbc = tune_klr_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,kernel='radial',r_match = r_match,degree_pool = degree_pool, rho_pool = rho_pool,lambda_pool = lambda_pool)[[1]]
####################################################################
# estimate probability using knn
####################################################################
k_pool = seq(6,40,by=2)
out_knn = tune_knn(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match,k_pool = k_pool)[[1]]
out_knn_tbc = tune_knn_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match,k_pool = k_pool)[[1]]
####################################################################
# estimate probability using random forest
####################################################################
ntree_pool = seq(50,500,by=50)
mtry_pool=c(0.3,0.4,0.5,0.6,0.7,0.8)
out_rf = tune_rf(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match,ntree_pool=ntree_pool,mtry_pool=mtry_pool)[[1]]
out_rf_ova = tune_rf_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match,ntree_pool=ntree_pool,mtry_pool=mtry_pool)[[1]]
out_rf_tbc = tune_rf_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match,ntree_pool=ntree_pool,mtry_pool=mtry_pool)[[1]]
####################################################################
# extend traditional svm to classification with confidence
####################################################################
rho_pool = 1
gamma_pool = 1/(rho_pool^2)
cost_pool = 10^((-100:20)/20)
degree_pool = 1
out_svm = tune_svm(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='linear',degree_pool=degree_pool,rho_pool = rho_pool)[[1]]
out_svm_ova = tune_svm_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='linear',degree_pool=degree_pool,gamma_pool = gamma_pool)[[1]]
out_svm_tbc = tune_svm_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='linear',degree_pool=degree_pool,gamma_pool = gamma_pool)[[1]]
# combine all the outcomes
colnames(out_csvm) = colnames(out_svm)
rownames(out_csvm) = c('CSVM')
out_matrix = rbind(out_csvm,out_lr,out_lr_ova,out_lr_tbc,out_knn,out_knn_tbc,out_rf,out_rf_ova,out_rf_tbc,out_svm,out_svm_ova,out_svm_tbc)
write.csv(out_matrix,file=filename,row.names = FALSE)