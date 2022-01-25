#############################################################################################################################
#kernel function
#############################################################################################################################
LinearKern = function(x,y) {
  kxx=tcrossprod(x,y)
  return(kxx)
}
GaussKern = function(x,y,rho=1) {
  x.one=x
  y.one=y
  x.one[]=1
  y.one[]=1
  kxx=exp((2*tcrossprod(x,y)-tcrossprod(x*x,y.one)-tcrossprod(x.one,y*y))/(rho)^2)
  return(kxx)
}
PolyKern = function (x,y,degree=1) {
  kxx=(tcrossprod(x,y))^degree
  return(kxx)
}
#############################################################################################################################
# Generate a k dimensional simplex in k-1 dimensional space
#############################################################################################################################
simplex = function(k) {
  W = matrix(0,nrow = k-1,ncol = k)
  W[,1] = rep(1,(k-1))/sqrt(k-1)
  for (i in 2:k) {
    W[,i] = -((1+sqrt(k))/((k-1)^(3/2)))*rep(1,(k-1)) + ((k/(k-1))^(1/2))*(diag(nrow = k-1)[,(i-1)])
  } 
  return(W)
}
#############################################################################################################################
# Vectorization
#############################################################################################################################
# Vectorization
vectorize = function(X) {
  n = nrow(X)
  p = ncol(X)
  m_vec = rep(0,(n*p))
  for (i in 1:p) {
    m_vec[(((i-1)*n)+1):(i*n)] = X[,i]
  }
  return(m_vec)
}
vectorize = function(X) {
  return(as.vector(X))
}
#############################################################################################################################
# Take Derivative
#############################################################################################################################
dconcave <- function(X,Y,k=NULL,lambda,initial=NULL,kernel='linear',rho=1,degree=1,...){
  n = nrow(X)
  p = ncol(X)
  # get the number of classes
  if (is.null(k)) {k = max(Y) - min(Y) + 1}
  # generate simplex 
  W = simplex(k)
  # calculate the input matrix
  if (kernel == 'linear') {
    K=LinearKern(X,X)
  } else if (kernel == 'radial') {
    K=GaussKern(X,X,rho=rho)
  } else if (kernel == 'polynomial') {
    K=PolyKern(X,X,degree=degree)  
  } else (return("Invalid kernel"))
  # get matrix of Y for calculation convenience
  Y_mat = matrix(1,nrow = n,ncol = k)
  for (i in 1:n) {
    Y_mat[i,Y[i]] = 0
  }
  # initialize last iteration
  if (length(initial) == 0) {initial <- c(rep(0,(k-1)*(n+1)),0)}
  beta = matrix(initial[1:(n*(k-1))],nrow=n,ncol=k-1)
  b = initial[((k-1)*n+1):((k-1)*(n+1))]
  epsilon = initial[(k-1)*(n+1)+1]
  # let's calculate the derivative
  f = t(beta)%*%K + matrix(rep(b,n),nrow = k-1,ncol = n)
  ## the indicator part
  fw = t(f)%*%W
  ind = matrix(as.numeric(((fw + epsilon)*Y_mat)>0),nrow = nrow(fw), ncol = ncol(fw))
  # write the derivative down
  c_for_beta = t(lambda * W %*% t(ind))
  c_for_b = lambda * W %*% t(ind) %*% rep(1,n)
  c_for_epsilon = lambda * sum(ind)
  return(c(vectorize(c_for_beta),c_for_b,c_for_epsilon)) 
}
#############################################################################################################################
# check the loss
#############################################################################################################################
loss = function(X,Y,point,k=NULL,lambda,kernel='linear',rho=1,degree=1,...) {
  n = nrow(X)
  p = ncol(X)
  # get the number of classes
  if (is.null(k)) {k = max(Y) - min(Y) + 1}
  # generate simplex 
  W = simplex(k)
  # calculate the input matrix
  if (kernel == 'linear') {
    K=LinearKern(X,X)
  } else if (kernel == 'radial') {
    K=GaussKern(X,X,rho=rho)
  } else if (kernel == 'polynomial') {
    K=PolyKern(X,X,degree=degree)  
  } else (return("Invalid kernel"))
  # get matrix of Y for calculation convenience
  Y_mat = matrix(1,nrow = n,ncol = k)
  for (i in 1:n) {
    Y_mat[i,Y[i]] = 0
  }
  # get the parameters
  beta = matrix(point[1:(n*(k-1))],nrow=n,ncol=k-1)
  b = point[((k-1)*n+1):((k-1)*(n+1))]
  epsilon = point[(k-1)*(n+1)+1]
  # let's calculate the derivative
  f = t(beta)%*%K + matrix(rep(b,n),nrow = k-1,ncol = n)
  ## the indicator part
  fw = t(f)%*%W
  ind_vex = matrix(as.numeric(((1 + fw + epsilon)*Y_mat)>0),nrow = nrow(fw), ncol = ncol(fw))
  ind_cav = matrix(as.numeric(((fw + epsilon)*Y_mat)>0),nrow = nrow(fw), ncol = ncol(fw))
  # calculate the main loss
  main_loss = lambda * ((sum(ind_vex * ((1 + fw + epsilon)*Y_mat))) - (sum(ind_cav * ((fw + epsilon)*Y_mat))))
  # calculate the penalty
  penalty = 0.5 * sum(diag(t(beta) %*% K %*% beta))
  total_loss = main_loss + penalty 
  return(total_loss)
}
#############################################################################################################################
# Train
#############################################################################################################################
mcsvmwc.qp.lp=function(X,Y,initial=NULL,k=NULL,weight=NULL,lambda,r,kernel='linear',jitter=1e-7,rho=1,degree=1,max_iteration=20,...){
  n = nrow(X)
  p = ncol(X)
  # get the number of classes
  if (is.null(k)) {k = length(r)}
  # start with a weight
  if (is.null(weight)) {weight = rep(1,n)}
  # generate simplex 
  W = simplex(k)
  # generate Y matrix
  Y_matrix = matrix(-1,nrow = n,ncol = k)
  for (i in 1:n) {
    Y_matrix[i,Y[i]] = 1
  }
  # aggregate error rate
  Y_count = matrix(0,nrow = n,ncol = k)
  for (i in 1:n) {
    Y_count[i,Y[i]] = 1
  }
  ### Y matrix part
  Y_vec = vectorize(Y_matrix)
  Y_part = Y_vec%*%t(Y_vec)
  ### W part 
  W_part = t(W)%*%W
  class_size = apply(Y_count,2,sum)
  zeta = r*class_size
  # then do quadratic programming: min_{b} -t(d)b + 1/2t(b)Db, s.t. t(A)b >= b_0
  ## quadratic(D) part
  ### kernel part
  if (kernel == 'linear') {
    K=LinearKern(X,X)
  } else if (kernel == 'radial') {
    K=GaussKern(X,X,rho=rho)
  } else if (kernel == 'polynomial') {
    K=PolyKern(X,X,degree=degree)  
  } else (return("Invalid kernel"))
  ### finally get D
  D_OnlyAlpha = (W_part %x% K)*Y_part
  for(i in (2:(k*n))) {
    for (j in (1:(i-1))) {
      D_OnlyAlpha[i,j] = D_OnlyAlpha[j,i]
    }
  }
  D = matrix(0,nrow = (n*k+k), ncol = (n*k+k))
  D[1:(n*k),1:(n*k)] = D_OnlyAlpha
  ## A part
  ### equality
  A_1 = (t(W) %x% matrix(1,nrow=n,ncol=1))*(matrix(1,nrow=1,ncol=(k-1)) %x% Y_vec)
  ### constrain for alphas
  #### greater than 0
  A_2 = diag(1,n*k)
  #### bounded by lambda or theta
  A_3 = diag(-1,n*k)
  A_theta = matrix(0,nrow=n*k,ncol=k)
  for (i in 1:n) {
    A_theta[(((Y[i]-1)*n)+i),Y[i]] = weight[i]
  }
  ### last constraint
  A_4 = -t(Y_vec)
  ### get A
  A = matrix(0,nrow = ((2*n*k)+k),ncol = ((n*k)+k))
  A[1:(k-1),1:(n*k)] = t(A_1)
  A[k:((n*k)+k-1),1:(n*k)] = A_2
  A[(n*k+k):((2*n*k)+k-1),1:(n*k)] = A_3
  A[(n*k+k):((2*n*k)+k-1),((n*k)+1):((n*k)+k)] = A_theta
  A[(2*n*k)+k,1:(n*k)] = A_4
  # start the d.v. loop
  indicator = 0
  while((indicator < max_iteration)) {
    ### get derivative at the point point abtained in last iteration
    if (length(initial) == 0) {initial <- c(rep(0,(k-1)*(n+1)),0)}
    dev_last = dconcave(X=X,Y=Y,k=k,lambda=lambda,initial=initial,kernel=kernel,rho=rho,degree=degree)
    c_for_beta = matrix(dev_last[1:(n*(k-1))],nrow=n,ncol=k-1)
    c_for_b = dev_last[((k-1)*n+1):((k-1)*(n+1))]
    c_for_epsilon = dev_last[(k-1)*(n+1)+1]
    ## d part depends on last iteration
    d = rep(0,(k*n+k))
    d[1:(n*k)] =  1 - vectorize((K%*%c_for_beta%*%W)*Y_matrix) 
    for (j in 1:k) {
      d[k*n+j]=-zeta[j]
    }
    ## b_0 part depends on last iteration
    b_0 = rep(0,((2*k*n)+k))
    b_0[1:(k-1)] = -c_for_b
    b_0[((n+1)*k):((2*n*k)+k-1)] = (-lambda)*(vectorize(Y_count-Y_matrix))
    b_0[(2*k*n)+k] = c_for_epsilon
    #solve
    require(quadprog)
    #  evalue = eigen(D)
    #  m = min(evalue$values)
    if (indicator == 0) {Dhat = D}
    while (det(Dhat) < 1e-180) {
      Dhat = D+diag(rep((jitter),nrow(D)))
      jitter = jitter * 2
    }
    start_time = proc.time()
    s = solve.QP(Dhat,d,t(A),b_0,meq = k-1)
    total_time = proc.time() - start_time
    print(total_time)
    outcome = s$solution
    ## outcome of dual
    alpha=matrix(outcome[1:(n*k)],nrow=n,ncol=k)
    theta=outcome[(n*k+1):(n*k+k)]
    ## from dual to the primal
    alpha_tilde = alpha * Y_matrix
    beta = (alpha_tilde %*% t(W)) + c_for_beta
    ## we can have slope in the linear case
    slope = t(X) %*% alpha_tilde %*% t(W)
    scale = sum(diag(t(slope)%*%slope))
    # then do linear programming to find b and epsilon: min_{b} obj * b, s.t. con * b >=< rhs
    ## use beta to get predicted scores
    xbeta = K %*% beta 
    yxbeta = (xbeta %*% W) * Y_matrix
    # linear programming
    ## objective
    obj_cav = c_for_b/lambda
    obj = c(vectorize(1 - Y_count),-obj_cav,obj_cav,(-(c_for_epsilon/lambda)))
    ## coefficients
    ### sum of eta for each class
    first = matrix(0,nrow=k,ncol=(((n+2)*k)-1))
    for (i in 1:n) {
      first[Y[i],(i+(Y[i]-1)*n)] = weight[i]
    }
    ### xi and eta should be greater than 0
    #  second = diag(1, nrow = n*k ,ncol = (n+1)*k)
    ### xi and eta should be lower bounded by hinged loss
    third = cbind(diag(1, nrow = n*k ,ncol = n*k), t(W) %x% rep(1,n) * matrix(rep(Y_vec,k-1),nrow=n*k,ncol=k-1), -t(W) %x% rep(1,n) * matrix(rep(Y_vec,k-1),nrow=n*k,ncol=k-1), Y_vec)
    ### epsilon should be greater than 0
    #  fourth = c(rep(0,((n+1)*k)-1),1)
    ### combine to get the condition matrix
    #  con=rbind(first,second,third,fourth)
    con=rbind(first,third)
    ## direction of constraints
    #  dir=c(rep("<=",k),rep(">=",(2*n*k+1)))
    dir=c(rep("<=",k),rep(">=",(n*k)))
    ## right-hand side
    #  rhs=c(zeta,rep(0,(n*k)),rep(1,n*k)-vectorize(yxbeta),0)
    rhs=c(zeta,rep(1,n*k)-vectorize(yxbeta))
    ## use package "lpsolve" to do optimization
    require("lpSolve")
    linear_programming = lp(direction = "min", objective.in=obj, const.mat=con, const.dir=dir, const.rhs = rhs)
    linear_outcome = linear_programming$solution
    ## get outcome
    ### matrix of eta and xi
    xi_eta_vec=linear_outcome[1:(n*k)]
    xi_eta_matrix = matrix(xi_eta_vec,nrow=n,ncol=k)
    ### intercept term
    b = linear_outcome[(n*k+1):((n+1)*k-1)] - linear_outcome[((n+1)*k):((n+2)*k-2)]
    ### epsilon
    epsilon=linear_outcome[(n+2)*k-1]
    ### eta
    eta_matrix = xi_eta_matrix * Y_count
    eta_vec = apply(eta_matrix,1,sum)
    ### xi
    xi_matrix = xi_eta_matrix - eta_matrix
    # check whether we need another loop
    indicator = indicator + 1
    this_point = c(vectorize(beta),b,epsilon)
    print(loss(X=X,Y=Y,point=this_point,k=k,lambda=lambda,kernel=kernel,rho=rho,degree=degree))
    print(indicator)
    if (sum(abs(this_point-initial)) < 1e-4) {break}
    initial = this_point
  }
  #let's think of the new weight
  nw = rep(1,n)
  indexlarge = which(eta_vec > 1)
  if (length(indexlarge)!=0) {
    nw[indexlarge] = 1/eta_vec[indexlarge]
  }
  thresholds = rep((-epsilon),k)
  return(list("X_train"=X,"Y_train"=Y,"beta"=beta,"slope" = slope,"scale" = scale,"b"=b,'epsilon'=epsilon,
              'thresholds'=thresholds,'xi'=xi_matrix,'eta'=eta_vec,'original_weight'=weight,'newweight'=nw,
              'lambda'=lambda,'kernel'=kernel,'rho'=rho,'degree'=degree,'k'=k,...))
}
#############################################################################################################################
# predict
#############################################################################################################################
predict.mcsvmwc=function(model,X_test,Y_test=NULL) {
  #extract infomation from the original model
  beta=model$beta
  b = model$b
  epsilon = model$epsilon
  thresholds = model$thresholds
  kernel = model$kernel
  k = model$k
  W = simplex(k)
  X_train=model$X_train
  # calculate kernel matrix
  if (kernel == 'linear') {
    K=LinearKern(X_test,X_train)
  } else if (kernel == 'radial') {
    K=GaussKern(X_test,X_train,model$rho)
  } else if (kernel == 'polynomial') {
    K=PolyKern(X_test,X_train,model$degree)
  }
  # let's start with the original prediction
  n_test = nrow(X_test)
  # get the score
  numy = (K%*%beta + (matrix(1,nrow=n_test,ncol=1) %x% matrix(b,nrow=1,ncol=(k-1))))%*%W
  Y_predict = matrix(0,nrow=n_test,ncol=k)
  Y_predict_list = rep(list(c()),n_test)
  Y_predict_counter = rep(0,k)
  for (i in 1:n_test) {
    for (j in 1:k) {
      if (numy[i,j]>=thresholds[j]) {
        Y_predict[i,j] = 1
        Y_predict_list[[i]] = c(Y_predict_list[[i]],j)
      }
    }
    if (max(Y_predict[i,]) == 0) {
      Y_predict[i,which.max(numy[i,])] = 1
      Y_predict_list[[i]] = c(Y_predict_list[[i]],which.max(numy[i,]))
    }
  }
  for (ii in 1:n_test) {
    Y_predict_counter[length(Y_predict_list[[ii]])] = Y_predict_counter[length(Y_predict_list[[ii]])] + 1
  }
  # get some evaluation of model performance
  ambiguity.prop = sum(Y_predict)/n_test
  if (is.null(Y_test)) {
    return(list('prediction_matrix'=Y_predict,'prediction_list'=Y_predict_list,'ambiguity proportion'=ambiguity.prop,'error1'=NULL,'error2'=NULL,'error'=NULL,'control1'=NULL,'control2'=NULL))
  } else {
    error = rep(0,k)
    controlerror = rep(0,k)
    for (j in 1:k) {
      # count error rate
      if (length(which(sapply(Y_predict_list, FUN=function(x) j %in% x))) != 0) {
      error[j]=1-(length(intersect(which(sapply(Y_predict_list, FUN=function(x) j %in% x)),which(Y_test==j)))/length(which(sapply(Y_predict_list, FUN=function(x) j %in% x))))
      } else {error[j] = 0}
      # count control rate
      if (length(which(Y_test==j)) != 0) {
      controlerror[j]=1-(length(intersect(which(sapply(Y_predict_list, FUN=function(x) j %in% x)),which(Y_test==j)))/length(which(Y_test==j)))
      } else {controlerror[j] = 0}
    }
    return(list('prediction_matrix'=Y_predict,'prediction_list'=Y_predict_list,'score'=numy,'ambiguity proportion'=ambiguity.prop,'error'=error,'control'=controlerror,'counter'=Y_predict_counter))
  }
}
#############################################################################################################################
# tuning
#############################################################################################################################
mcsvmwc_tune = function(x.train,y.train,x.tune,y.tune,r,lambdapool=10^((-8:8)/2),kernel='linear',max.iteration = 20,max.violation = 1,degree_pool=1,rho_pool=1,...) {
  n = nrow(x.train)
  final_ambiguity = Inf
  for (degree in degree_pool) {
    for (rho in rho_pool) {
      ambiguity = Inf
      weight = rep(1,n)
      w_0 = rep(0,n)
      for (i in 1:length(lambdapool)) {
        print(paste('lambda = ',lambdapool[i]))
        indicator = 1
        while ((max(abs(weight-w_0))>0.01 && indicator <= max.iteration) || (indicator == 1)) {
          print(paste('iteration for weight = ',indicator))
          w_0 = weight
          model=mcsvmwc.qp.lp(X=x.train,Y=y.train,weight=w_0,lambda=lambdapool[i],r=r,kernel=kernel,degree=degree,rho=rho,...)
          weight=model$newweight
          indicator=indicator+1
        }
        pre = predict.mcsvmwc(model,x.tune,y.tune)
        temp = pre$'ambiguity proportion'
        tempcon = pre$control
        if (temp < ambiguity && max(tempcon-r) <= 0) {
          ambiguity = temp
          finallambda=lambdapool[i]
          finalmodel = model
        }
      }
      if (ambiguity==Inf) {
        return ('Cannot control coverage rate, please try again!')
      } 
      # do a more careful search
      print(paste("betterlambda=",finallambda,sep = ''))
      accurate=10^((-10:10)/20)
      newpool = finallambda*accurate
      weight = finalmodel$original_weight
      w_0 = rep(0,n)
      for (ii in 1:length(newpool)) {
        print(paste('lambda = ',newpool[ii]))
        indicator = 1
        while ((max(abs(weight-w_0))>0.01 && indicator <= max.iteration) || (indicator == 1)) {
          print(paste('iteration for weight = ',indicator))
          w_0 = weight
          model=mcsvmwc.qp.lp(X=x.train,Y=y.train,weight=w_0,lambda=newpool[ii],r=r,kernel=kernel,degree=degree,rho=rho,...)
          weight=model$newweight
          indicator=indicator+1
        }
        pre = predict.mcsvmwc(model,x.tune,y.tune)
        temp = pre$'ambiguity proportion'
        tempcon = pre$control
        if (temp < ambiguity && max(tempcon-r) <= 0) {
          ambiguity = temp
          finallambda=newpool[ii]
          finalmodel = model
        }
      }
      if (ambiguity < final_ambiguity) {
        finalmodel_tuned = finalmodel
        final_ambiguity = ambiguity
      }
    }
  }
  return(finalmodel_tuned)
}
#############################################################################################################################
# robust tuning
#############################################################################################################################
mcsvmwc_tune_robust = function(x.train,y.train,x.tune,y.tune,r,lambdapool=10^((-8:8)/2),kernel='linear',max.iteration = 20,max.violation = 1,degree_pool=1,rho_pool=1,...) {
  n = nrow(x.train)
  index_list = list()
  allowance = rep(1,length(r))
  for (i in 1:length(r)) {
    index_list = c(index_list,list(which(y.tune==i)))
    allowance[i] = floor(r[i] * length(which(y.tune==i)))
  }
  final_ambiguity = Inf
  for (degree in degree_pool) {
    for (rho in rho_pool) {
      ambiguity = Inf
      weight = rep(1,n)
      w_0 = rep(0,n)
      for (i in 1:length(lambdapool)) {
        print(paste('lambda = ',lambdapool[i]))
        indicator = 1
        while ((max(abs(weight-w_0))>0.01 && indicator <= max.iteration) || (indicator == 1)) {
          print(paste('iteration for weight = ',indicator))
          w_0 = weight
          model=mcsvmwc.qp.lp(X=x.train,Y=y.train,weight=w_0,lambda=lambdapool[i],r=r,kernel=kernel,degree=degree,rho=rho,...)
          weight=model$newweight
          indicator=indicator+1
        }
        pre_original = predict.mcsvmwc(model,x.tune,y.tune)
        # define a new kind of threshold
        score_tune = pre_original$score
        thresholds = rep(0,length(r))
        for (jj in 1:length(r)) {
          thresholds[jj] = sort(score_tune[index_list[[jj]],jj],decreasing = FALSE)[allowance[jj]]
        }
        model$thresholds = thresholds
        pre = predict.mcsvmwc(model,x.tune,y.tune)
        temp = pre$'ambiguity proportion'
        tempcon = pre$control
        if (temp <= ambiguity && max(tempcon-r) <= 0) {
          ambiguity = temp
          finallambda=lambdapool[i]
          finalmodel = model
        }
      }
      if (ambiguity==Inf) {
        return ('Cannot control coverage rate, please try again!')
      } 
      # do a more careful search
      print(paste("betterlambda=",finallambda,sep = ''))
      accurate=10^((-10:10)/20)
      newpool = finallambda*accurate
      weight = finalmodel$original_weight
      w_0 = rep(0,n)
      for (ii in 1:length(newpool)) {
        print(paste('lambda = ',newpool[ii]))
        indicator = 1
        while ((max(abs(weight-w_0))>0.01 && indicator <= max.iteration) || (indicator == 1)) {
          print(paste('iteration for weight = ',indicator))
          w_0 = weight
          model=mcsvmwc.qp.lp(X=x.train,Y=y.train,weight=w_0,lambda=newpool[ii],r=r,kernel=kernel,degree=degree,rho=rho,...)
          weight=model$newweight
          indicator=indicator+1
        }
        pre_original = predict.mcsvmwc(model,x.tune,y.tune)
        # define a new kind of threshold
        score_tune = pre_original$score
        thresholds = rep(0,length(r))
        for (jj in 1:length(r)) {
          thresholds[jj] = sort(score_tune[index_list[[jj]],jj],decreasing = FALSE)[allowance[jj]]
        }
        model$thresholds = thresholds
        pre = predict.mcsvmwc(model,x.tune,y.tune)
        temp = pre$'ambiguity proportion'
        tempcon = pre$control
        if (temp <= ambiguity && max(tempcon-r) <= 0) {
          ambiguity = temp
          finallambda=newpool[ii]
          finalmodel = model
        }
      }
      if (ambiguity < final_ambiguity) {
        finalmodel_tuned = finalmodel
        final_ambiguity = ambiguity
      }
    }
  }
  return(finalmodel_tuned)
}