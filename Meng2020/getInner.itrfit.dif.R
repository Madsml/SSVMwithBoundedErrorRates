getInner.itrfit.dif=function(obj, newdata=NULL, index=NULL) {
 level=obj$level
 k=length(level)
 var=colnames(obj$x)
 id=rownames(obj$x)
 if (is.null(index)) A=obj$coef else A=obj$A[, , index]
 K=obj$kernel
 if (!is.null(newdata)) {
  x=obj$x
  newx=as.matrix(newdata)
  if (any(colnames(newx)!=var)) stop('testing set must have the same structure as the training set!')
  id=rownames(newdata)
  kernel=attr(K, 'type')
  if (kernel=='linear') {
   K=newx%*%t(x)
  } else if (kernel=='gaussian') {
   epsilon=attr(K, 'epsilon')
   K=exp(-epsilon*as.matrix(pdist(newx, x))^2)
  } else if (kernel=='polynomial') {
   d=attr(K, 'degree')
   K=(1+newx%*%t(x))^d
  }
 }
 inner=cbind(1, K)%*%A%*%W.gen(k)
 colnames(inner)=level
 rownames(inner)=id
 return(inner)
}