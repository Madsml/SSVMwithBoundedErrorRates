pred=function(inner) {
 d=dim(inner)
 res=matrix(0, d[1], d[2])
 sapply(1:d[1], function(i) which.max(inner[i, ]))
}