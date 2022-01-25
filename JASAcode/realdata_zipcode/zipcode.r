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
filename_out=paste("./out/zipcode_seed_",seed,sep="")
####################################################################
# read dataset
####################################################################
load('zip_all.Rdata')
# combine the training and testing data
x.new = rbind(x,x.test)
y.new = c(y,y.test)
# regenerate the training set
x.0=x.new[y.new==1,]
x.1=x.new[y.new==2,]
x.2=x.new[y.new==3,]
x.3=x.new[y.new==4,]
x.4=x.new[y.new==5,]
x.5=x.new[y.new==6,]
x.6=x.new[y.new==7,]
x.7=x.new[y.new==8,]
x.8=x.new[y.new==9,]
x.9=x.new[y.new==10,]
# get the row number of each subset
n0 = nrow(x.0)
n1 = nrow(x.1)
n2 = nrow(x.2)
n3 = nrow(x.3)
n4 = nrow(x.4)
n5 = nrow(x.5)
n6 = nrow(x.6)
n7 = nrow(x.7)
n8 = nrow(x.8)
n9 = nrow(x.9)
# shuffle the index
s0=sample(1:dim(x.0)[1])
s1=sample(1:dim(x.1)[1])
s2=sample(1:dim(x.2)[1])
s3=sample(1:dim(x.3)[1])
s4=sample(1:dim(x.4)[1])
s5=sample(1:dim(x.5)[1])
s6=sample(1:dim(x.6)[1])
s7=sample(1:dim(x.7)[1])
s8=sample(1:dim(x.8)[1])
s9=sample(1:dim(x.9)[1])
# get the training set
x_train = rbind(x.0[s0[1:200],],x.6[s6[1:200],],x.8[s8[1:200],],x.9[s9[1:200],])
y_train = c(rep(1,200),rep(2,200),rep(3,200),rep(4,200))
# get the tuning set
x_tune = rbind(x.0[s0[201:400],],x.6[s6[201:400],],x.8[s8[201:400],],x.9[s9[201:400],])
y_tune = c(rep(1,200),rep(2,200),rep(3,200),rep(4,200))
# rest is the testing set
x_test = rbind(x.0[s0[401:n0],],x.6[s6[401:n6],],x.8[s8[401:n8],],x.9[s9[401:n9],])
y_test = c(rep(1,n0-400),rep(2,n6-400),rep(3,n8-400),rep(4,n9-400))
####################################################################
# train the data set with mcsvm
####################################################################
# get the coverage setting
k = 10
r = rep(0.01,k)
# tuning the model
model_robust = mcsvmwc_tune_robust(x_train=x_train,y_train=y_train,x_tune=x_tune,y_tune=y_tune,r=r,kernel='radial',lambdapool=10^((-10:4)/2),max.iteration = 5,max.violation = 1,rho_pool=10^((0:6)/4),degree_pool=1)
Y_predict_robust = predict.mcsvmwc(model = model_robust, X_test = x_test, Y_test = y_test)
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
for (jj in 1:length(r)) {
  thresholds[jj] = sort(score_test[index_list[[jj]],jj],decreasing = FALSE)[allowance[jj]]
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
out_logit = tune_lr(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=rep(0.05,3), r_match = r_match, nlambda = nlambda, alpha_pool = alpha_pool)[[1]]
####################################################################
# run kernel logistic regression first and try to tune it a bit
####################################################################
# tune for lambda and rho
lambda_pool = 10^((-200:200)/20)
rho_pool = 10^((0:6)/4)
#rho_pool = 1
degree_pool = 1
model_klr = tune_klr(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,kernel='radial',r_match = r_match,degree_pool = degree_pool, rho_pool = rho_pool,lambda_pool = lambda_pool)[[1]]
out_klr_ova = tune_klr_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,kernel='radial',r_match = r_match,degree_pool = degree_pool, rho_pool = rho_pool,lambda_pool = lambda_pool)[[1]]
out_klr_tbc = tune_klr_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,kernel='radial',r_match = r_match,degree_pool = degree_pool, rho_pool = rho_pool,lambda_pool = lambda_pool)[[1]]
####################################################################
# estimate probability using knn
####################################################################
k_pool = seq(6,40,by=2)
model_knn = tune_knn(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match,k_pool = k_pool)[[1]]
out_knn_tbc = tune_knn_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match,k_pool = k_pool)[[1]]
####################################################################
# estimate probability using random forest
####################################################################
ntree_pool = seq(50,500,by=50)
mtry_pool=c(0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8)
model_rf = tune_rf(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match,ntree_pool=ntree_pool,mtry_pool=mtry_pool)[[1]]
out_rf_ova = tune_rf_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match,ntree_pool=ntree_pool,mtry_pool=mtry_pool)[[1]]
out_rf_tbc = tune_rf_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r, r_match = r_match,ntree_pool=ntree_pool,mtry_pool=mtry_pool)[[1]]
####################################################################
# extend traditional svm to classification with confidence
####################################################################
rho_pool = 10^((-2:6)/4)
gamma_pool = 1/(rho_pool^2)
cost_pool = 10^((-100:20)/20)
degree_pool = 1
model_svm = tune_svm(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='radial',degree_pool=degree_pool,rho_pool = rho_pool)[[1]]
out_svm_ova = tune_svm_ova(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='radial',degree_pool=degree_pool,gamma_pool = gamma_pool)[[1]]
out_svm_tbc = tune_svm_tbc(x_train = x_train, y_train = y_train, x_tune = x_tune, y_tune = y_tune, x_test = x_test, y_test = y_test, r=r,r_match = r_match,cost_pool = cost_pool,kernel='radial',degree_pool=degree_pool,gamma_pool = gamma_pool)[[1]]
####################################################################
# combine all the outcomes
####################################################################
colnames(out_csvm) = colnames(out_svm)
rownames(out_csvm) = c('CSVM')
out_matrix = rbind(out_csvm,out_lr,out_lr_ova,out_lr_tbc,out_klr,out_klr_ova,out_klr_tbc,out_knn,out_knn_tbc,out_rf,out_rf_ova,out_rf_tbc,out_svm,out_svm_ova,out_svm_tbc)
write.csv(out_matrix,file=filename_out,row.names = FALSE)