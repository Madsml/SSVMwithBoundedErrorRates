itrFit=function(x, a, y, p=1/k, c=1.2, method='svm', kernel='linear', epsilon=1/2/median(d)^2, d=2, lambda=5^(-5:5), cv=F, tunning=T) {
 n=length(a)
 a=as.factor(a)
 level.name=levels(a)
 a=as.numeric(a)
 k=max(a)
 lambda=sort(lambda, decreasing=T)
 m=length(lambda)
 ind=as.matrix(dist(a, 'manhattan')) #ind=ind_{ai=al}-1/(k-1)*ind_{ai\neq al}
 ind[which(ind!=0)]=-1/(k-1)
 ind[which(ind==0)]=1
 if (any(p<0 | p>1)) stop('p must be between 0 to 1!')
 if (length(c)>1 | c<1) stop('c must be a number no less than 1!')
 w=rep(y/p, length=n)
 cminus=c-1
 x=as.matrix(x)
 if (kernel=='linear') {
  ker=x%*%t(x)
 } else if (kernel=='gaussian') {
  d=as.matrix(dist(x))
  ker=exp(-epsilon*d^2)
 } else if (kernel=='polynomial') {
  ker=(1+x%*%t(x))^d
 } else stop('kernel must be one of the followings: linear, gaussian, polynomial.')
 K=ker+diag(n)*0.01*mean(diag(ker))
 WWK=ind*(K+1)
 method=match.arg(method, c('svm', 'dwd',  'exponential', 'logistic', 'square'), several = T)
 if (method=='svm') {
  diagK=diag(K)+1
  W_aW=matrix(-1/(k-1), n, k) #W_aW=<W_ai, W_j>
  W_aW[cbind(1:n, a)]=1
  if (cv) {
   inner=matrix(0, n, k*m)
   folds=sample.int(n)%%5+1
   for (i in 1:5) {
    id=folds==i
    theta_gamma=svmfit_C(WWK[!id, !id], diagK[!id], w[!id], cminus, lambda)
    inner[id, ]=get_inner(theta_gamma, ker[id, !id], W_aW[!id, ])
   }
  } else {
   theta_gamma=svmfit_C(WWK, diagK, w, cminus, lambda)
   inner=get_inner(theta_gamma, ker, W_aW)
  }
  dim(inner)=c(n, k, m)
 } else {
  Wbasis=W.gen(k)
  W=t(Wbasis[, a, drop=F])
  inner=array(0, c(n, k, m))
  if (cv) {
   folds=sample.int(n)%%5+1
   for (i in 1:5) {
    id=folds==i
    A=diffit_C(WWK[!id, !id], K[!id, !id], W[!id, , drop=F], w[!id], cminus, lambda, method)
    for (j in 1:m) {
     inner[id, , j]=cbind(1, ker[id, !id])%*%A[, , j]%*%Wbasis
    }
   }
  } else {
   A=diffit_C(WWK, K, W, w, cminus, lambda, method)
   for (j in 1:m) {
    inner[, , j]=cbind(1, ker)%*%A[,  ,j]%*%Wbasis
   }
  }
 }
 obj=numeric(m)
 for (i in 1:m) {
  rule=sapply(1:n, function(j) which.max(inner[j, , i]))
  obj[i]=sum(a!=rule)
 }
 opt=which.min(obj)
 if (opt==1 | opt==m) warning('The lambda may not be optimal. Another range may be considered.')
 optlambda=lambda[opt]
 optinner=inner[, , opt]
 res=list(obj_value=c(itr=obj[opt]), optlambda=optlambda, A=A, c=c, level=level.name, x=x, a=a, y=y, method=method, kernel=ker)
 if (method=='svm') {
  if (cv) {
   theta_gamma=svmfit_C(WWK, diagK, w, cminus, optlambda)
   inner=get_inner(theta_gamma, ker, W_aW)
  } else {
   theta_gamma=theta_gamma[, opt]
   inner=optinner
  }
  res$theta_gamma=as.numeric(theta_gamma)
  class(res)=c('itrfit.svm', 'itrfit')
 } else {
  if (cv) {
   A=diffit_C(WWK, K, W, w, cminus, lambda[1:opt], method)[, , opt]
   inner=cbind(1, ker)%*%A%*%Wbasis
  } else {
   A=A[, , opt]
   inner=optinner
  }
  res$coef=A
  class(res)=c('itrfit.dif', 'itrfit')
 }
 attr(res$kernel, 'type')=kernel
 if (kernel=='gaussian') attr(res$kernel, 'epsilon')=epsilon
 if (kernel=='polynomial') attr(res$kernel, 'degree')=d
 if (c>1 & tunning) {
  tunning=tune(optinner, a, y, p, c)
  obj=tunning$obj_value
  res$obj_value=c(res$obj_value, itrrnr=obj)
  if (cv) tunning=tune(inner, a, y, p, c)
  res$predict=tunning$rule
  colnames(res$predict)=level.name
  rownames(res$predict)=rownames(x)
  attr(res$predict, 'outcome.ratio')=c
  class(res$predict)=c('ITR', 'matrix')
  res$refine_par=tunning$refine_par
 } else {
  colnames(inner)=level.name
  rownames(inner)=rownames(x)
  res$predict=pred(inner)
 }
 return(res)
}