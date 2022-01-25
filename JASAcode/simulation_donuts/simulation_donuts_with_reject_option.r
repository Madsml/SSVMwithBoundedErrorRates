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
# data generation
###########################################################################################################################
four_layer_donut = function(n,p,p_noise) {
  # start with a four layer donut
  x1 = matrix(0,nrow=0,ncol=p)
  while (dim(x1)[1]<n) {
    new_varible = c(runif(1,min=0,max=0.65),runif(1,0,(2*pi)))
    x1 = rbind(x1,c((new_varible[1]*cos(new_varible[2])),(new_varible[1]*sin(new_varible[2]))))
  }
  x2 = matrix(0,nrow=0,ncol=p)
  while (dim(x2)[1]<n) {
    new_varible = c(runif(1,min=0.45,max=1.1),runif(1,0,(2*pi)))
    x2 = rbind(x2,c((new_varible[1]*cos(new_varible[2])),(new_varible[1]*sin(new_varible[2]))))
  }
  x3 = matrix(0,nrow=0,ncol=p)
  while (dim(x3)[1]<n) {
    new_varible = c(runif(1,min=0.9,max=1.55),runif(1,0,(2*pi)))
    x3 = rbind(x3,c((new_varible[1]*cos(new_varible[2])),(new_varible[1]*sin(new_varible[2]))))
  }
  x4 = matrix(0,nrow=0,ncol=p)
  while (dim(x4)[1]<n) {
    new_varible = c(runif(1,min=1.35,max=2),runif(1,0,(2*pi)))
    x4 = rbind(x4,c((new_varible[1]*cos(new_varible[2])),(new_varible[1]*sin(new_varible[2]))))
  }
  X= rbind(x1,x2,x3,x4)
  Y = c(rep(1,n),rep(2,n),rep(3,n),rep(4,n))
  # add some noise to the data
  mu_noise = rep(0,p_noise)
  sigma_noise = diag(rep((1/p_all),p_noise))
  x_noise = rmvnorm((4*n),mu_noise,sigma_noise)
  # combine signal and noise
  X = as.matrix(cbind(X,as.data.frame(x_noise)))
  return(list(X,Y))
}
# first lets state the setting of this simulation
n = 40
p_all = 400
# give a name to the model
filename=paste("./reject_new/simulation_seed_",seed,"_p=",p_all,"_n=",n,".csv",sep="")
# get start
p = 2
p_noise = p_all - p
k = 4
# training data
n_train = n
data_train = four_layer_donut(n=n_train,p=p,p_noise=p_noise)
x_train = data_train[[1]]
y_train = data_train[[2]]
# tuning data
n_tune = n
data_tune = four_layer_donut(n=n_tune,p=p,p_noise=p_noise)
x_tune = data_tune[[1]]
y_tune = data_tune[[2]]
x_threshold = x_tune
y_threshold = y_tune
# don't forget the testing set
n_test = 10000
# start with a box
data_test = four_layer_donut(n=n_test,p=p,p_noise=p_noise)
x_test = data_test[[1]]
y_test = data_test[[2]]
###########################################################################################################################
# train the model
###########################################################################################################################
r = rep(0.05,k)
####################################################################
# consider reject and refine options
####################################################################
# ok let's use kernel outside
# define kernel
Kmat <- function(x, y, kernel="polynomial", kparam = 1) {
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
  if (is.null(kparam)) {kparam = 2:4}
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
d_vec=seq(0.6,0.74,by=0.02)
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
  loss = 0
  noncover = rep(0,k)
  ambiguity = 0
  for (tt in 1:length(y_test)) {
    if (0 %in% pred[[tt]]) {pred[[tt]]=(1:k)} 
    # add ambiguity
    ambiguity = ambiguity + length(pred[[tt]]) 
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