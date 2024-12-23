library(BGLR)

#Read corrected data and CV index for Body Length
dat = read.table("MiceBL.txt",sep=",")
index = read.table("MiceBLindex.txt",sep=",",header=TRUE)
n = dim(dat)[1]
#print(n)
Xp = dim(dat)[2]
#print(Xp)
X = dat[,3:Xp] #Skip col 2 with EBVs
y = dat[,1]

X = scale(X,center=TRUE,scale=TRUE) # Scale the markers

# 10 CV folds
corfold1 = c(1:10)
MSEfold1 = c(1:10)
corfold2 = c(1:10)
MSEfold2 = c(1:10)
for (fold in 1:10){
#set.seed((fold)) #For repeatable folds
#randind = sample(c(1:n),replace = FALSE)
randind = index[,fold]
split = 1270 #Training split
Xperm = X[randind,]
ncolXperm = dim(X)[2]
xtrain = Xperm[1:split,]
xtest = Xperm[c(split+1):n,]

yperm = y[randind]
ytest = yperm[c(split+1):n]
ytrain = yperm
ytrain[c(split+1):n] = NA

# G-matrix
G = tcrossprod(Xperm)/ncolXperm

niter = 15000
burnin = 5000

# Fit RKHS model
fm1 = BGLR(y=ytrain, ETA=list(G=list(K=G, model='RKHS')),
            nIter=niter, burnIn=burnin)

# Fit BayesC model
fm2 = BGLR(y=ytrain, ETA=list(list(X=Xperm, model='BayesC')),
            nIter=niter, burnIn=burnin)

yHat1 = fm1$yHat # Extract predicted y-test for RKHS

# Calculate test MSE and correlation for RKHS
RMSEP1 = sum((ytest-yHat1[c(split+1):n])^2)/length(yHat1[c(split+1):n])
 MSEfold1[fold] = RMSEP1
cor1 = cor(ytest,yHat1[c(split+1):n])
 corfold1[fold] = cor1

yHat2 = fm2$yHat # Extract predicted y-test for BayesC

# Calculate test MSE and correlation for BayesC
RMSEP2 = sum((ytest-yHat2[c(split+1):n])^2)/length(yHat2[c(split+1):n])
 MSEfold2[fold] = RMSEP2
cor2 = cor(ytest,yHat2[c(split+1):n])
 corfold2[fold] = cor2
}

#RKHS result BL
print(MSEfold1/2) #MSE hould be divided by 2 to get same scale as in Julia
mean(MSEfold1)/2
sd(MSEfold1/2)
corfold1
mean(corfold1)
sd(corfold1)


#BayesC result BL
print(MSEfold2/2) #MSE should be divided by 2
mean(MSEfold2)/2
sd(MSEfold2/2)
corfold2
mean(corfold2)
sd(corfold2)



#Read corrected data and CV index for Body Mass Index
dat = read.table("MiceBMI.txt",sep=",")
index = read.table("MiceBLindex.txt",sep=",",header=TRUE)
n = dim(dat)[1]
#print(n)
Xp = dim(dat)[2]
#print(Xp)
X = dat[,3:Xp] #Skip col 2
y = dat[,1]

X = scale(X,center=TRUE,scale=TRUE)

# 10 CV folds
corfold1 = c(1:10)
MSEfold1 = c(1:10)
corfold2 = c(1:10)
MSEfold2 = c(1:10)
for (fold in 1:10){
#set.seed((fold)) #For repeatable folds
#randind = sample(c(1:n),replace = FALSE)
randind = index[,fold]
split = 1270 #2175 #2521 3152
Xperm = X[randind,]
ncolXperm = dim(X)[2]
xtrain = Xperm[1:split,]
xtest = Xperm[c(split+1):n,]

yperm = y[randind]
ytest = yperm[c(split+1):n]
ytrain = yperm
ytrain[c(split+1):n] = NA

G = tcrossprod(Xperm)/ncolXperm

niter = 15000
burnin = 5000

fm1 = BGLR(y=ytrain, ETA=list(G=list(K=G, model='RKHS')),
            nIter=niter, burnIn=burnin)

#BayesC
fm2 = BGLR(y=ytrain, ETA=list(list(X=Xperm, model='BayesC')),
            nIter=niter, burnIn=burnin)

yHat1 = fm1$yHat

RMSEP1 = sum((ytest-yHat1[c(split+1):n])^2)/length(yHat1[c(split+1):n])
 MSEfold1[fold] = RMSEP1
cor1 = cor(ytest,yHat1[c(split+1):n])
 corfold1[fold] = cor1

yHat2 = fm2$yHat

RMSEP2 = sum((ytest-yHat2[c(split+1):n])^2)/length(yHat2[c(split+1):n])
 MSEfold2[fold] = RMSEP2
cor2 = cor(ytest,yHat2[c(split+1):n])
 corfold2[fold] = cor2
}

#RKHS result BMI
print(MSEfold1/2)
mean(MSEfold1)/2
sd(MSEfold1/2)
corfold1
mean(corfold1)
sd(corfold1)


#BayesC result BMI
print(MSEfold2/2) 
mean(MSEfold2)/2
sd(MSEfold2/2)
corfold2
mean(corfold2)
sd(corfold2)









#Cleve data
dat = read.csv("Cleve2.csv",header = FALSE, sep = ",")
index = read.table("Clevetr2index.txt",sep=",",header=TRUE)
n = dim(dat)[1]
print(n)
Xp = dim(dat)[2]
print(Xp)
X = dat[,3:Xp] #Skip col 2
y = dat[,1]
#y = yBLcorr

#X = scale(X)
X<-scale(X,center=TRUE,scale=TRUE)

corfold1 = c(1:10)
MSEfold1 = c(1:10)
corfold2 = c(1:10)
MSEfold2 = c(1:10)
for (fold in 1:10){
#set.seed((fold)) #For repeatable folds
#randind = sample(c(1:n),replace = FALSE)
randind = index[,fold]
split = 1900 #2175 #2521 3152
Xperm = X[randind,]
ncolXperm = dim(X)[2]
xtrain<-Xperm[1:split,]
xtest<-Xperm[c(split+1):n,]

yperm = y[randind]
ytest <- yperm[c(split+1):n]
ytrain <- yperm
#ytrain <- y
ytrain[c(split+1):n] <- NA

#G <- tcrossprod(X)


G<-tcrossprod(Xperm)/ncolXperm

#EVD <- eigen(G)
niter = 15000
burnin = 5000

fm1 <- BGLR(y=ytrain, ETA=list(G=list(K=G, model='RKHS')),
            nIter=niter, burnIn=burnin)

#BayesC
fm2 <- BGLR(y=ytrain, ETA=list(list(X=Xperm, model='BayesC')),
            nIter=niter, burnIn=burnin)

yHat1<-fm1$yHat

RMSEP1<-sum((ytest-yHat1[c(split+1):n])^2)/length(yHat1[c(split+1):n])
 MSEfold1[fold] = RMSEP1
cor1 = cor(ytest,yHat1[c(split+1):n])
 corfold1[fold] = cor1

yHat2<-fm2$yHat

RMSEP2<-sum((ytest-yHat2[c(split+1):n])^2)/length(yHat2[c(split+1):n])
 MSEfold2[fold] = RMSEP2
cor2 = cor(ytest,yHat2[c(split+1):n])
 corfold2[fold] = cor2
}

> #RKHS
> print(MSEfold1/2) #Should be divided by 2
 [1] 0.4596580 0.4436349 0.4940862 0.4591491 0.4810294 0.4668845 0.4664440
 [8] 0.4380885 0.4676505 0.4672580
> 
> mean(MSEfold1)/2
[1] 0.4643883
> 
> sd(MSEfold1/2)
[1] 0.01619477
> corfold1
 [1] 0.4773495 0.4845896 0.4790279 0.4917279 0.5009119 0.4537814 0.5051226
 [8] 0.4874187 0.4688069 0.4815014
> mean(corfold1)
[1] 0.4830238
> sd(corfold1)
[1] 0.01495767
> 
> #BayesC
> print(MSEfold2/2) #Should be divided by 2
 [1] 0.4597104 0.4438761 0.4928430 0.4593092 0.4813390 0.4674003 0.4663586
 [8] 0.4387745 0.4673882 0.4677798
> mean(MSEfold2)/2
[1] 0.4644779
> sd(MSEfold2/2)
[1] 0.01582848
> corfold2
 [1] 0.4768094 0.4843319 0.4807033 0.4916417 0.5004087 0.4530079 0.5052172
 [8] 0.4860491 0.4692547 0.4806203
> mean(corfold2)
[1] 0.4828044
> sd(corfold2)
[1] 0.01498057




dat = read.csv("Cleve4.csv",header = FALSE, sep = ",")
index = read.table("Clevetr4index.txt",sep=",",header=TRUE)
n = dim(dat)[1]
print(n)
Xp = dim(dat)[2]
print(Xp)
X = dat[,3:Xp] #Skip col 2
y = dat[,1]
#y = yBLcorr

#X = scale(X)
X<-scale(X,center=TRUE,scale=TRUE)

corfold1 = c(1:10)
MSEfold1 = c(1:10)
corfold2 = c(1:10)
MSEfold2 = c(1:10)
for (fold in 1:10){
#set.seed((fold)) #For repeatable folds
#randind = sample(c(1:n),replace = FALSE)
randind = index[,fold]
split = 2206 #2175 #2521 3152
Xperm = X[randind,]
ncolXperm = dim(X)[2]
xtrain<-Xperm[1:split,]
xtest<-Xperm[c(split+1):n,]

yperm = y[randind]
ytest <- yperm[c(split+1):n]
ytrain <- yperm
#ytrain <- y
ytrain[c(split+1):n] <- NA

#G <- tcrossprod(X)


G<-tcrossprod(Xperm)/ncolXperm

#EVD <- eigen(G)
niter = 15000
burnin = 5000

fm1 <- BGLR(y=ytrain, ETA=list(G=list(K=G, model='RKHS')),
            nIter=niter, burnIn=burnin)

#BayesC
fm2 <- BGLR(y=ytrain, ETA=list(list(X=Xperm, model='BayesC')),
            nIter=niter, burnIn=burnin)

yHat1<-fm1$yHat

RMSEP1<-sum((ytest-yHat1[c(split+1):n])^2)/length(yHat1[c(split+1):n])
 MSEfold1[fold] = RMSEP1
cor1 = cor(ytest,yHat1[c(split+1):n])
 corfold1[fold] = cor1

yHat2<-fm2$yHat

RMSEP2<-sum((ytest-yHat2[c(split+1):n])^2)/length(yHat2[c(split+1):n])
 MSEfold2[fold] = RMSEP2
cor2 = cor(ytest,yHat2[c(split+1):n])
 corfold2[fold] = cor2
}


> #RKHS
> print(MSEfold1/2) #Should be divided by 2
 [1] 1.961687 2.116343 2.182929 2.136827 2.157524 2.451560 2.128580 2.073192
 [9] 2.118396 2.267770
> 
> mean(MSEfold1)/2
[1] 2.159481
> 
> sd(MSEfold1/2)
[1] 0.128744
> corfold1
 [1] 0.4575229 0.4432512 0.4796992 0.4431370 0.4651085 0.4351992 0.4649647
 [8] 0.4441302 0.4533484 0.4508065
> mean(corfold1)
[1] 0.4537168
> sd(corfold1)
[1] 0.01337797
> 
> #BayesC
> print(MSEfold2/2) #Should be divided by 2
 [1] 1.960089 2.115165 2.180982 2.133903 2.158525 2.447316 2.130227 2.072478
 [9] 2.113795 2.263003
> mean(MSEfold2)/2
[1] 2.157548
> sd(MSEfold2/2)
[1] 0.1277496
> corfold2
 [1] 0.4583319 0.4436332 0.4804217 0.4443576 0.4647527 0.4367708 0.4644977
 [8] 0.4444365 0.4552321 0.4527624
> mean(corfold2)
[1] 0.4545197
> sd(corfold2)
[1] 0.01304604





