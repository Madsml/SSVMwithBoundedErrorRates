get_inner=function(theta_gamma, K, W_aW) {
## inner=<W_j, f(x_i)>, k*1 vector for each observation
## n*k matrix for each lambda, column bind for all lambdas
 K=K+1
 theta_gamma=as.matrix(theta_gamma)
 m=ncol(theta_gamma)
 inner=lapply(1:m, function(j) -K%*%(theta_gamma[, j]*W_aW))
 do.call('cbind', inner)
}