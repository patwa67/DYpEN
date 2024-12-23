library(BGLR)
data(mice)

phendat = mice.pheno # Original phenotype and covariate data 
phendat$Date.Month = as.factor(phendat$Obesity.Date.Month)
phendat$Date.Year = as.factor(phendat$Obesity.Date.Year)

# Linear model to correct for important covariates: Body length
lmBL1 = lm(Obesity.BodyLength ~ Date.Month+Date.Year+GENDER+CoatColour+CageDensity+
           Litter,data=phendat)
summary(lmBL1)
anova(lmBL1)
yBLcorr = resid(lmBL1) # Residuals serve as new phenotype
#hist(yBLcorr,25)

library(Matrix)
library(INLA)
A <- Matrix(mice.A,sparse=TRUE) # A-matrix
Ainv<-as(solve(A),  "dgTMatrix") # Inverse of A-matrix
Ainv[ abs(Ainv) < sqrt(.Machine$double.eps) * max(Ainv) ] = 0

# Set up the data for INLA
ID<-matrix(seq(1,dim(mice.A)[1]),dim(mice.A)[1],1)
dataf<-data.frame(cbind(ID,yBLcorr))
names(dataf) <- c("ID","y")

# Bayesian GBLUP model in INLA
model = y ~ 1 + f(ID, model="generic0", 
        Cmatrix=Ainv)
fit = inla(model, data=dataf, verbose=TRUE,
           control.compute=list(dic=T,mlik=T,cpo=T)) 

summary(fit)

# Improved estimates
h = inla.hyperpar(fit, diff.logdens=15)

summary(h)
#plot(h)
#str(h)

ebvres = data.frame(h$summary.random) 
ebvmean = ebvres[,2]# Breeding values
data = cbind(yBLcorr,ebvmean,mice.X) # Combine y,EBVs and markers. Save to file
write.table(data, file = "MiceBL.txt", append = FALSE, sep = ",",
            row.names = FALSE,col.names = FALSE)

# Heritability BL
her = (1/h$summary.hyperpar$mean[2])/((1/h$summary.hyperpar$mean[2])+(1/h$summary.hyperpar$mean[1]))


# Linear model to correct for important covariates: Body Mass Index
lmBMI = lm(Obesity.BMI ~ Date.Month+Date.Year+GENDER+CoatColour+CageDensity+
           Litter,data=phendat)
summary(lmBMI)
anova(lmBMI)
yBMIcorr = resid(lmBMI)
#hist(yBMIcorr,25)

library(Matrix)
library(INLA)
A <- Matrix(mice.A,sparse=TRUE)
Ainv<-as(solve(A),  "dgTMatrix")
Ainv[ abs(Ainv) < sqrt(.Machine$double.eps) * max(Ainv) ] = 0

ID<-matrix(seq(1,dim(mice.A)[1]),dim(mice.A)[1],1)
data2<-data.frame(cbind(ID,yBMIcorr))
names(data2) <- c("ID","y")


model = y ~ 1 + f(ID, model="generic0", 
        Cmatrix=Ainv)

fit2 = inla(model, data=data2, verbose=TRUE,
           control.compute=list(dic=T,mlik=T,cpo=T)) 

summary(fit2)

h2 = inla.hyperpar(fit2, diff.logdens=15)

summary(h2)
#plot(h2)
#str(h2)

ebvres = data.frame(h2$summary.random)
ebvmean = ebvres[,2]
data = cbind(yBMIcorr,ebvmean,mice.X)
write.table(data, file = "MiceBMI.txt", append = FALSE, sep = ",",
            row.names = FALSE,col.names = FALSE)
