# load the package needed
require(randomForest)
# an advanced way to tune classification with confidence and give both ordinary and robust implementation 
tune_rf_tbc = function(x_train,y_train,x_tune,y_tune,x_threshold=x_tune,y_threshold=y_tune,x_test,y_test,r,r_match,ntree_pool = seq(50,500,by=50),mtry_pool=c(0.3,0.4,0.5,0.6,0.7,0.8)) {
  class_number = length(r)
  # get the dimension
  p = ncol(x_train)
  n = nrow(x_train)
  # create a big n_test*k matrix telling people which class each observation belongs to
  n_test = length(y_test)
  count_mat = matrix(0,nrow=n_test,ncol= class_number)
  adjust_mat = matrix(0,nrow=n_test,ncol= class_number)
  score_mat = matrix(0,nrow=n_test,ncol=class_number)
  for (ii in 1:class_number) {
    # generate a temporary label for training and tuning
    y_train_temp = 2 * (0.5 - as.numeric(y_train == ii))
    y_tune_temp = 2 * (0.5 - as.numeric(y_tune == ii))
    y_test_temp = 2 * (0.5 - as.numeric(y_test == ii))   
    y_threshold_temp = 2 * (0.5 - as.numeric(y_threshold == ii))
    # save the index for two classes of training and tuning data 
    index_train = which(y_train_temp == -1)
    index_tune = which(y_tune_temp == -1)
    index_test = which(y_test_temp == -1)
    index_threshold = which(y_threshold_temp == -1)
    # get the threshold for tuning set 
    thresh_robust = max(floor(r[ii]*length(index_threshold)),1)
    thresh_adjust = max(floor(r_match[ii]*length(index_test)),1)
    # initialize the misclassification rate (criterion used here)
    errorrate = 1
    for (i in 1:length(ntree_pool)) {
      for (j in 1:length(mtry_pool)) {
        # get current parameter
        ntree = ntree_pool[i]
        mtry = mtry_pool[j]
        # train the model with specified parameters
        model = randomForest(x=x_train,y=as.factor(y_train_temp),ntree=ntree,mtry=(mtry*ncol(x_train)))
        # scores for threshold set
        score_threshold = predict(object = model,newdata = x_threshold,type = 'prob')[,2]
        score_threshold_neg = score_threshold[index_threshold]
        threshold_tune = sort(score_threshold_neg,decreasing = TRUE)[thresh_robust]
        # scores for tuning set
        score_tune = predict(object = model,newdata = x_tune,type = 'prob')[,2]
        # check whether the missclassification rate decrease
        yhat = as.numeric(score_tune>0)-as.numeric(score_tune<=0)
        errorrate_temp = length(which(yhat!=y_tune_temp))/length(y_tune_temp)
        if (errorrate_temp <= errorrate) {
          errorrate = errorrate_temp
          model_best = model
          threshold_robust = threshold_tune
        }
      }
    }
    
  # test the model performance
  # calculate scores for testing data
  score_test = predict(model_best,x_test,type = 'prob')[,2]
  # get adjusted threshold
  score_test_neg = score_test[index_test]
  threshold_test = sort(score_test_neg,decreasing = TRUE)[thresh_adjust]
  # write the outcome onto counter and adj matrix
  yhat_robust = rep(0,length(y_test)) - as.numeric(score_test<=threshold_robust)
  yhat_adjust = rep(0,length(y_test)) - as.numeric(score_test<=threshold_test)
  count_mat[which(yhat_robust==-1),ii] = 1
  adjust_mat[which(yhat_adjust==-1),ii] = 1
  score_mat[,ii] = -score_test
  }
  # fill in NULL region with a prediction of one vs all classifier
  for (i in 1:n_test) {
    if (max(count_mat[i,]) == 0) {count_mat[i,which.max(score_mat[i,])] = 1}
    if (max(adjust_mat[i,]) == 0) {adjust_mat[i,which.max(score_mat[i,])] = 1}
  }
  # now let's see the performance
  ambiguity_robust = sum(count_mat)/n_test
  ambiguity_adjust = sum(adjust_mat)/n_test
  # prepare for the performance chart
  simulation = data.frame(matrix(0,nrow = 1, ncol = (2*(class_number+1))))
  simulation[1,1] = ambiguity_robust
  simulation[1,(class_number+2)] = ambiguity_adjust
  robust_name = c()
  adjust_name = c()
  for (jj in 1:class_number) {
    robust_name = c(robust_name,paste('uncover_class_',jj,'_robust',sep = ""))
    adjust_name = c(adjust_name,paste('uncover_class_',jj,'_adjust',sep = ""))
    simulation[1,(jj+1)] = 1 - (sum(count_mat[,jj]*as.numeric(y_test==jj))/sum(as.numeric(y_test==jj)))
    simulation[1,(jj+ class_number+2)] = 1 - (sum(adjust_mat[,jj]*as.numeric(y_test==jj))/sum(as.numeric(y_test==jj)))
  }
  colnames(simulation) = c('ambiguity_robust',robust_name,
                           'ambiguity_adjust',adjust_name)
  rownames(simulation) = c('ranForest_tbc')
  # return performance of adjusted classification with confidence
  return (list(simulation,count_mat,adjust_mat))
}