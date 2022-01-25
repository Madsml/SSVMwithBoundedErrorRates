# source packages and solvers
library(mvtnorm)
# source codes for rej and refine
library(RnR)
source("../R/angle_MLUM_solver.r")
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
n = 40
set.seed(seed)
p_all = 100
# give a name to the model
filename=paste("./reject_new/simulation_seed_",seed,"_p=",p_all,"_n=",n,".csv",sep="")
# get start
p = 2
p_noise = p_all - p
k = 3
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
# ok let's use kernel outside
# define kernel
Kmat <- function(x, y, kernel="radial", kparam = 1) {
  if (kernel == "polynomial") {
    obj <- (x %*% t(y) + 1)^kparam
  } else if (kernel == "radial") {
    normx <- drop((x^2) %*% rep(1, ncol(x)))
    normy <- drop((y^2) %*% rep(1, ncol(y)))
    temp <- x %*% t(y)
    temp <- (-2 * temp + normx) + outer(rep(1, nrow(x)), normy, "*")
    obj <- exp(-temp * kparam)
  } else obj <- NULL
  obj
}
# define a different training function
dwd_tune2=function(x.train,y.train,x.tune,y.tune,lambda=NULL,alpha=0,a,d,rho=1,delta=NULL,epsilon=1e-4,kparam=NULL)
{
  tune.error = Inf
  if (is.null(lambda)) {lambda = rev(2^((-15:15)/2)) } 
  if (is.null(delta)) {delta = c(0.5,0.4,0.3,0.2,0.1,0)}
  if (is.null(kparam)) {kparam = 10^((2:-2)/2)}
  for (ss in 1:length(kparam))
  {
    kparam.now=kparam[ss]
    K.train = Kmat(x.train,x.train,kparam=kparam.now)
    K.tune = Kmat(x.tune,x.train,kparam=kparam.now)
    z = dwd_solver(K.train,y.train,lambda=lambda,alpha=alpha,a=a,rho=rho,
                   epsilon=epsilon)
    for (ii in rev(1:length(lambda)))
    {
      for (jj in rev(1:length(delta)))
      {
        zz = pred_reject(K.tune,z$beta[[ii]],z$beta0[[ii]],z$k,delta[jj])
        error = sum(as.numeric(zz!=y.tune))-(1-d)*length(which(zz==0))
        if (error<tune.error)
        {
          tune.error = error
          best.lambda=z$lambda[ii]
          best.beta=z$beta[[ii]]
          best.beta0=z$beta0[[ii]]
          best.delta=delta[jj]
          best.kparam=kparam.now
        }
      }
    }
  } # for (ss in 1:length(kparam))
  tt=list(model=z,best.beta=best.beta, best.beta0=best.beta0, best.lambda=best.lambda,best.delta=best.delta,best.kparam=best.kparam)
  return(tt)
}
# let's start
d_vec=seq(0.05,0.2,by=0.01)
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
  z_rej = dwd_tune2(x_train,y_train,x_tune,y_tune,a=aa,d=d,alpha=0,delta=c(0.3,0.25,0.2,0.15,0.1,0.05,0) )
  K_test = Kmat(x_test,x_train,kparam=z_rej$best.kparam)
  # check for the noncoverage rate
  pred = pred_refine(K_test,z_rej$best.beta,z_rej$best.beta0,k,z_rej$best.delta)
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
#    loss = loss + (d/(k-1)*(length(pred[[tt]])-1))
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