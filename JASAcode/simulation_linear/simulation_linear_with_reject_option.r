# source packages and solvers
library(mvtnorm)
# source codes for rej and refine
library(RnR)
source("/home/wang2/Documents/xingye_paper/scene/R/angle_MLUM_solver.r")
# set seed
procid = 0
seed = 123*procid
set.seed(seed)
###########################################################################################################################
# data generation
###########################################################################################################################
# first lets state the setting of this simulation
n = 40
set.seed(seed)
p_all = 10
# give a name to the model
filename=paste("./reject_new/simulation_seed_",seed,"_p=",p_all,"_n=",n,".csv",sep="")
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
####################################################################
# instead of training a mcsvm, we compare the model myself
####################################################################
# read the outcome
file_to_read = paste("./run/p=",p_all,"_n=",n,"/simulation_seed_",seed,"_p=",p_all,"_n=",n,".csv",sep="")
csvm_out = read.csv(file_to_read)
csvm_out = csvm_out[1,(1:(k+1))]
print(csvm_out)
Y_predict_robust = NULL
Y_predict_robust$`ambiguity proportion` = csvm_out[1]
Y_predict_robust$control = csvm_out[2:(k+1)]
####################################################################
# consider reject and refine options
####################################################################
d_vec=seq(0.05,0.6,by=0.05)
# get a big matrix of all the noncoverage rate and ambiguity
out_matrix = matrix(0,nrow = length(d_vec),ncol = (k+2))
# train a rej and refine model for a specific d and a
for (dindex in 1:length(d_vec)) {
  d = d_vec[dindex]
  # ok which aa do you pick
  aa_small=(k-1-d)/(k*d-d)
  aa_large=(k-1)*(1-d)/d
  aa_mid=(aa_small+aa_large)/2
  aa = aa_large
  # record the ambiguity and non-coverage rate
  total_ambiguity = 0
  total_loss = 0
  real_d = 0
  diff_ambiguity = k
  # train a rej and refine model
  z_rej = dwd_tune(x_train,y_train,x_tune,y_tune,a=aa,d=d,alpha=0,delta=c(0.3,0.25,0.2,0.15,0.1,0.05,0) )
  # check for the noncoverage rate
  pred = pred_refine(x_test,z_rej$best.beta,z_rej$best.beta0,k,z_rej$best.delta)
  ref1.index = numeric(0)
  ref2.index = numeric(0)
  ref3.index = numeric(0)
  loss = 0
  noncover = rep(0,k)
  ambiguity = 0
  for (tt in 1:length(y_test)) {
    if (0 %in% pred[[tt]]) {pred[[tt]]=(1:k)} 
    # add ambiguity
    ambiguity = ambiguity + length(pred[[tt]]) 
    if (length(pred[[tt]])==1) {ref1.index=c(ref1.index,tt)}
    if (length(pred[[tt]])==2) {ref2.index=c(ref2.index,tt)}
    if (length(pred[[tt]])==3) {ref3.index=c(ref3.index,tt)}
    # check noncoverage rate
    if (!(y_test[tt] %in% pred[[tt]])) {
      loss = loss + 1
      noncover[y_test[tt]] = noncover[y_test[tt]] + 1
    }
    if (tt%%5000==0) {print(pred[[tt]])}
  }
  # what is the loss and ambiguity?
  loss = loss/(3*length(y_test))
  ambiguity = ambiguity/length(y_test)
  # then we should check the coverage
  print(ambiguity)
  print(loss)
  print(z_rej$best.delta)
  for (ll in 1:k) {
    noncover[ll] = noncover[ll]/sum(y_test==ll)
    print(sum(y_test==ll))
  }
  print(noncover)
  print(length(ref1.index))
  print(length(ref2.index))
  print(length(ref3.index))
  out_matrix[dindex,] = c(ambiguity,noncover,loss)
}
####################################################################
# combine all the outcomes
####################################################################
# what should we include? ambiguity, loss, and gaps for both method.
noncover_names = c()
for (nameindex in 1:k) {noncover_names = c(noncover_names,paste('noncover',nameindex,sep=''))}
colnames(out_matrix) = c('ambi',noncover_names,'loss')
row.names(out_matrix) = d_vec
# record it
write.csv(out_matrix,file=filename,row.names = FALSE)