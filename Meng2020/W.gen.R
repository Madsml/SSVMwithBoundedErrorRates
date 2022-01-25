W.gen=function(k) {
 W=matrix(-(1+sqrt(k))/(k-1)^1.5, k-1, k-1)+(k/(k-1))^0.5*diag(k-1)
 cbind(1/(k-1)^0.5, W)
}