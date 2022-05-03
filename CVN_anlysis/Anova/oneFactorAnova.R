#install.packages("data.table")
library(data.table)
library(readxl)
library(rstatix)
data=read_excel('voxelVolumes_treatments.xlsx')
data=data[  , -c(1,2,3)]
dim(data)
data=na.omit(data)

#data=transpose(data, keep.names = "col")
#traits=read.csv('AD_DECODEpower.csv')

#removing AD and MCI
#indexofremoval=traits$risk_for_ad >1
#sum(indexofremoval)
#dim(traits[indexofremoval,])
#traits=traits[-which(indexofremoval),]


# traits$RAVLT_FORGETTING
# traits$genotype
# traits$sex
# traits$age
# traits$Risk
# traits$RAVLT_IMMEDIATE
#traits$RAVLT_PERCENTFORGETTING=abs(traits$RAVLT_FORGETTING)/traits$AVLT_Trial5
# a=which(as.matrix(traits[4, ])==12, arr.ind=T)
# a
#hist(traits$RAVLT_PERCENTFORGETTING)
#traits$RAVLcategorical=traits$RAVLT_PERCENTFORGETTING
## mak eit as a facotor 0.25 low, 0.5 medium low, 0.75 medium , 1 high 
#traits$RAVLcategorical[traits$RAVLcategorical<=0.25]="Low"
#traits$RAVLcategorical[ traits$RAVLcategorical<=0.5 & traits$RAVLcategorical>0.25]="Medlow"
#traits$RAVLcategorical[ traits$RAVLcategorical<=0.75 & traits$RAVLcategorical>0.5]="Med"
#traits$RAVLcategorical[ traits$RAVLcategorical<=1 & traits$RAVLcategorical>0.75]="High"




#listdatanames=as.numeric(substr(data$col, start = 2, stop = 6))
#listtraitnames=traits$MRI
#length(intersect(listdatanames,listtraitnames))
#comonlist=intersect(listdatanames,listtraitnames)
#matrixdata=matrix(0, length(comonlist),(dim(data)[2]+dim(traits)[2]))

#for (i in 1:length(comonlist)) {
  #cat(which(as.numeric(substr(data$col, start = 2, stop = 6))==comonlist[i]),  "\n")
  #indexvol=which(as.numeric(substr(data$col, start = 2, stop = 6))==comonlist[i])
  #indextrait=which(traits$MRI==comonlist[i])

#  a=data[indexvol,]
#  b=traits[indextrait,]
#  #length(c(a,b)) 
#  matrixdata[i,]=unlist(c(a,b))
#}
#colnames(matrixdata)=names(c(a,b))

#matrixdata=as.data.frame(matrixdata)

#for (i in 2:dim(data)[2]) {
#  matrixdata[,i]=as.numeric(matrixdata[, i])
#}
#matrixdata$age=as.numeric(matrixdata$age)

library(jmv)









pvalsresults=matrix(NA,(dim(data)[2]-1)  , 3 )
rownames(pvalsresults)=names(data)[2:dim(data)[2]]
colnames(pvalsresults)= c( "treatment", "homogen", "Sha-Wilk norm" )

len = dim(data)[2]
for (i in 1:(len-1))  {
   
  tempname=rownames(pvalsresults)[i]
  res.aov <- anova_test(get(tempname) ~ Treatment, data = data)
  a = get_anova_table(res.aov)
  p = a$p
  pvalsresults[i,1] <- p
  
}
pvalsresultsadjusted <- pvalsresults[pvalsresults[,1]<=0.05,]

#Error in Anova.III.lm(mod, error, singular.ok = singular.ok, ...) : 
#  there are aliased coefficients in the model
#Note: model has aliased coefficients
#sums of squares computed by model comparison
###### THis error is beacasue recval% and sex are present and teh residual are very close
pvalsresults[pvalsresults[,2]<=0.05,]

pvalsresultsadjusted=pvalsresults

###adjust pvalues Benjamini & Hochberg
for (j in 1:dim(pvalsresultsadjusted)[2]) {
  pvalsresultsadjusted[,j] = p.adjust(pvalsresultsadjusted[,j], "fdr") #Benjamini & Hochberg
}

#Error in p.adjust(pvalsresultscopy[, j], "BH", n = dim(data)[1]) : 
#  n >= lp is not TRUE
#### DUE TO SMALL SAMPLE SIZE THE P-VALUES CANNOT BE CORRECTED
sig = pvalsresultsadjusted[pvalsresultsadjusted[,1]<=0.05,] #Adjusted P-values

sig = sig[,1]
posthoc=matrix(NA,length(sig),4)
for (i in 1:length(sig)) {
  tempname=names(sig)[i]
  res.aov <- aov(get(tempname) ~ Treatment, data = data)
  tuk=tukey_hsd(res.aov)
  posthoc[,1]=sig
  posthoc[,2:4]=tuk$p.adj
}

colnames(posthoc)=c("FDR","ST","SW","TW")
rownames(posthoc)=names(sig)
