# load the package needed
if (!require(e1071)) {install.packages(e1071)}
library(e1071)
# an advanced way to tune classification with confidence and give both ordinary and robust implementation 
tune_svm_ova = function(x_train,y_train,x_tune,y_tune,x_threshold=x_tune,y_threshold=y_tune,x_test,y_test,r,r_match,cost_pool = 10^(-3:3),kernel='radial',gamma_pool = 1,degree_pool=2) {
  class_number = length(r)
  # get the dimension
  p = ncol(x_train)
  n = nrow(x_train)
  # initialize the ambiguity
  ambiguity = class_number + 1
  best_degree = 0
  best_gamma =  0
  best_cost = 0
  threshold_robust_temp = rep(0,class_number)
  model_list = list()
  # the tuning procedure to find best tuning parameter
  for (i in 1:length(degree_pool)) {
    for (j in 1:length(gamma_pool)) {
      for (k in 1:length(cost_pool)) {
        # get current parameter
        degree = degree_pool[i]
        gamma = gamma_pool[j]
        cost = cost_pool[k]
        # print(c(i,k))
        # create a big n_tune*k matrix
        n_tune = length(y_tune)
        count_mat_tune = matrix(0,nrow=n_tune,ncol=class_number)
        score_mat_tune = matrix(0,nrow=n_tune,ncol=class_number)
        for (ii in 1:class_number) {
          y_train_temp = 2 * (0.5 - as.numeric(y_train == ii))
          # generate a temporary label for tuning and thresholding
          y_tune_temp = 2 * (0.5 - as.numeric(y_tune == ii))
          y_threshold_temp = 2 * (0.5 - as.numeric(y_threshold == ii))
          # save the index for two classes of tuning and thresholding data 
          index_threshold = which(y_threshold_temp == -1)
          index_tune = which(y_tune_temp == -1)
          # train the model with specified parameters
          model = svm(x=x_train,y=as.factor(y_train_temp),degree=degree,gamma=gamma,kernel = kernel,cost = cost,probability = TRUE)
          # scores for threholding set to get the thresholds
          thresh_robust = floor(r[ii]*length(index_threshold))
          score_threshold = attr(predict(model,newdata=x_threshold,probability = TRUE),'probabilities')[,2]
          score_threshold_neg = score_threshold[index_threshold]
          threshold_tune = sort(score_threshold_neg,decreasing = TRUE)[thresh_robust]
          threshold_robust_temp[ii] = threshold_tune
          # scores for tuning set
          score_tune = attr(predict(model,newdata=x_tune,probability = TRUE),'probabilities')[,2]
          yhat_tune = rep(0,length(y_tune)) - as.numeric(score_tune<=threshold_tune)
          count_mat_tune[which(yhat_tune==-1),ii] = 1
          score_mat_tune[,ii] = -score_tune
          model_list[[ii]] = model
        }
        # fill in NULL region with a prediction of one vs all classifier
        for (jj in 1:n_tune) {
          if (max(count_mat_tune[jj,]) == 0) {count_mat_tune[jj,which.max(score_mat_tune[jj,])] = 1}
        }
        # check whether the ambiguity rate decrease
        ambiguity_temp = sum(count_mat_tune)/(n_tune)
        if (ambiguity_temp < ambiguity) {
          ambiguity = ambiguity_temp
          threshold_robust = threshold_robust_temp
          model_best = model_list
          best_degree = degree
          best_gamma =  gamma
          best_cost = cost
        }
      }
    }
  }
  # test the model performance
  # create a big n_test*k matrix telling people which class each observation belongs to
  n_test = length(y_test)
  count_mat = matrix(0,nrow=n_test,ncol=class_number)
  adjust_mat = matrix(0,nrow=n_test,ncol=class_number)
  score_mat = matrix(0,nrow=n_test,ncol=class_number)
  # calculate scores for testing data and calculate the thereshold
  for (jj in 1:class_number) {
    # use jjth class as the null hypothesis, generate a temporary label for training and testing
    y_train_temp = 2 * (0.5 - as.numeric(y_train == jj))
    y_test_temp = 2 * (0.5 - as.numeric(y_test == jj))
    index_test = which(y_test_temp == -1)
    # get the score for jjth class
    score_test = attr(predict(model_best[[jj]],x_test,probability = TRUE),'probabilities')[,2]
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
  rownames(simulation) = c('SVM_ova')
  # return performance of adjusted classification with confidence
  print(c(best_degree,best_gamma,best_cost))
  return(list(simulation,count_mat,adjust_mat))
}