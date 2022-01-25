predict.itrfit=function(obj, newdata=NULL, index=NULL, option='refine', delta=NULL, fence=NULL, ...) {
 inner=getInner(obj, newdata, index)
 c=obj$c
 if (option=='refine' & c>1) {
  if (is.null(delta)) {
   if (is.null(obj$refine_par)) stop('tunning parameter "delta" is missing!')
   else delta=obj$refine_par[1]
  }
  if (is.null(fence)) {
   if (is.null(obj$refine_par)) stop('tunning parameter "fence" is missing!')
   else fence=obj$refine_par[2]
  }
  rule=pred_refine(inner, delta, fence)
  attr(rule, 'outcome.ratio')=c
  class(rule)=c('ITR', 'matrix')
 } else {
  rule=pred(inner)
 }
 return(rule)
}