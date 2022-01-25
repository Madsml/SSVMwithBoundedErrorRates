# load the package needed
if (!require(glmnet)) {install.packages(glmnet)}
library(glmnet)
# an advanced way to tune classification with confidence and give both ordinary and robust implementation 
tune_lr_ova = function(x_train,y_train,x_tune,y_tune,x_threshold=x_tune,y_threshold=y_tune,x_test,y_test,r,r_match,lambda_pool = 10^(-8:4),alpha_pool = 0) {
  class_number = length(r)
  # get the dimension
  p = ncol(x_train)
  n = nrow(x_train)
  # need fix alpha first, but in our study we actually donot 
  for (i in 1:length(alpha_pool)) {
    # get current parameter
    alpha = alpha_pool[i]
    # for this method, we need to pre-train all the models
    model_list = list()
    prob_list = list()
    prob_threshold_list = list()
    for (qq in 1:class_number) {
      # generate a temporary label for training and tuning
      y_train_temp = 2 * (0.5 - as.numeric(y_train == qq))
      model = glmnet(x = x_train,y = y_train_temp,family='binomial',alpha=alpha,lambda = rev(lambda_pool))
      model_list[[qq]] = model
      prob_list[[qq]] = predict(model,newx=x_tune,type='response')
      prob_threshold_list[[qq]] = predict(model,newx=x_threshold,type='response')
    }
    # initialize the ambiguity
    ambiguity = class_number + 1
    alpha_best = 0
    j_best =  0
    threshold_robust_temp = rep(0,class_number)
    # Then we can talk about parameter selection 
    for (j in 1:length(lambda_pool)) {
      # create a big n_tune*k matrix
      n_tune = length(y_tune)
      count_mat_tune = matrix(0,nrow=n_tune,ncol=class_number)
      score_mat_tune = matrix(0,nrow=n_tune,ncol=class_number)
      threshold_robust_temp = rep(0,class_number) 
      for (ii in 1:class_number) {
        y_train_temp = 2 * (0.5 - as.numeric(y_train == ii))
        # generate a temporary label for tuning and thresholding
        y_tune_temp = 2 * (0.5 - as.numeric(y_tune == ii))
        y_threshold_temp = 2 * (0.5 - as.numeric(y_threshold == ii))
        # save the index for two classes of training and tuning data 
        index_threshold = which(y_threshold_temp == -1)
        index_tune = which(y_tune_temp == -1)
        # scores for threholding set to get the thresholds
        thresh_robust = floor(r[ii]*length(index_threshold))
        score_threshold = prob_threshold_list[[ii]][,j]
        score_threshold_neg = score_threshold[index_threshold]
        threshold_tune = sort(score_threshold_neg,decreasing = TRUE)[thresh_robust]
        threshold_robust_temp[ii] = threshold_tune
        # scores for tuning set
        score_tune = prob_list[[ii]][,j]
        yhat_tune = rep(0,length(y_tune)) - as.numeric(score_tune<=threshold_tune)
        count_mat_tune[which(yhat_tune==-1),ii] = 1
        score_mat_tune[,ii] = -score_tune
      }
      # fill in NULL region with a prediction of one vs all classifier
      for (jj in 1:n_tune) {
        if (max(count_mat_tune[jj,]) == 0) {count_mat_tune[jj,which.max(score_mat_tune[jj,])] = 1}
      }
      # check whether the ambiguity rate decrease
      ambiguity_temp = sum(count_mat_tune)/(n_tune)
      #      print(ambiguity_temp)
      if (ambiguity_temp < ambiguity) {
        ambiguity = ambiguity_temp
        threshold_robust = threshold_robust_temp
        j_best = j
        model_best = model_list
        alpha_best = alpha
      }
    }
  }
  # create a big n_test*k matrix telling people which class each observation belongs to
  n_test = length(y_test)
  count_mat = matrix(0,nrow=n_test,ncol=class_number)
  adjust_mat = matrix(0,nrow=n_test,ncol=class_number)
  score_mat = matrix(0,nrow=n_test,ncol=class_number)
  # calculate scores for testing data and calculate the thereshold
  for (jj in 1:class_number) {
    # use jjth class as the null hypothesis, generate a temporary label for training and testing
    y_test_temp = 2 * (0.5 - as.numeric(y_test == jj))
    index_test = which(y_test_temp == -1)
    # get the score for jjth class
    score_test = predict(model_best[[jj]],newx=x_test,type='response')[,j_best]
    score_mat[,jj] = -score_test
    # give predictions for robust implementation
    yhat_robust = rep(0,length(y_test)) - as.numeric(score_test<=threshold_robust[jj])
    count_mat[which(yhat_robust==-1),jj] = 1
    # give predictions for adjust implementation
    thresh_adjust = max(floor(r_match[jj]*length(index_test)),1)
    score_test_neg = score_test[index_test]
    threshold_test = sort(score_test_neg,decreasing = TRUE)[thresh_adjust]
    yhat_adjust = rep(0,length(y_test)) - as.numeric(score_test<=threshold_test)
    adjust_mat[which(yhat_adjust==-1),jj] = 1
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
    simulation[1,(jj+class_number+2)] = 1 - (sum(adjust_mat[,jj]*as.numeric(y_test==jj))/sum(as.numeric(y_test==jj)))
  }
  colnames(simulation) = c('ambiguity_robust',robust_name,
                           'ambiguity_adjust',adjust_name)
  rownames(simulation) = c('Logit_ova')
  # return performance of adjusted classification with confidence
  print(c(alpha,lambda_pool[j_best]))
  return(list(simulation,count_mat,adjust_mat))
}