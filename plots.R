require(Hmisc)
setwd("~/Work/palm/Persistent&Generative Models/CBRA/Model/")

group.CI <- function(response,factors=list(),flabel=list(),ci=0.95) {
	if (!is.list(factors))
		stop("Param 'factors' must be a list!")
	if (!is.list(flabel))
		stop("Param 'flabel' must be a list!")
	if (length(flabel)>0 && length(flabel) != length(factors))
		stop("Param 'flabel' must be same length as param 'factors'!")
	if (length(factors)>0)
		d.count = split(response,factors)
	else
		d.count = response
	d.ucl = as.data.frame(t(sapply(d.count,function(x){a<-mean(x);s<-sd(x);n<-length(x);error<-qt(ci+(1-ci)/2,df=n-1)*s/sqrt(n);c(a,a-error,a+error)})))
	colnames(d.ucl) = c("mean","lower","upper")
	if (length(factors)>1)
		conditions = as.data.frame(t(sapply(rownames(d.ucl),function(x){unlist(strsplit(x,".",fixed=T))})))
	else
		conditions = as.data.frame(rownames(d.ucl))
	if (length(flabel)>0)
		colnames(conditions) = flabel
	else
		colnames(conditions) = lapply(1:ncol(conditions),function(x){paste("F",x,sep="")})
	d = cbind(conditions,d.ucl)
	rownames(d) = NULL
	return(d)
}

while (TRUE) {
	d = read.delim("myers-v2-eg1.txt",header=F)
	d = d[d$V3>0,]
        d = d[d$V1<max(d$V1)-max(d$V1)%%20,] #<=max may have to be <max
	d = d[d$V1>=max(d$V1)-20,]
	d.ucl = group.CI(d$V5, list(d$V2,d$V4), flabel=list("dr","cond"))
  d.ucl$dr = ordered(d.ucl$dr,levels=1:15)
	p = xYplot(Cbind(mean,lower,upper)~numericScale(dr),group=as.factor(cond),data=d.ucl,type="l",label.curves=F,main=paste("Saliency+Uncertainty,Trials = ",max(d$V1),sep=""),ylab="Fixations",xlab="Distractor Ratio",auto.key=T)
	print(p)
	Sys.sleep(5)
}



while (TRUE) {

d1 = read.delim("data/model1.txt",header=F)
#d2 = read.delim("data/model2.txt",header=F)
#d3 = read.delim("data/model3.txt",header=F)
#d4 = read.delim("data/model4.txt",header=F)

d1.ucl = ucl(d1[d1$V1>=max(d1$V1)-100,])
#d2.ucl = ucl(d2[d2$V1>=max(d2$V1)-100,])
#d3.ucl = ucl(d3[d3$V1>=max(d3$V1)-100,])
#d4.ucl = ucl(d4[d4$V1>=max(d4$V1)-100,])

p1 = xYplot(Cbind(mean,lower,upper)~numericScale(dr),group=as.factor(cond),data=d1.ucl,type="l",label.curves=F,main=paste("Saliency+Uncertainty,Trials = ",max(d1$V1),sep=""),ylab="Fixations",xlab="Distractor Ratio")
#p2 = xYplot(Cbind(mean,lower,upper)~numericScale(dr),group=as.factor(cond),data=d2.ucl,type="l",label.curves=F,main=paste("Saliency,Trials = ",max(d2$V1),sep=""),ylab="Fixations",xlab="Distractor Ratio")
#p3 = xYplot(Cbind(mean,lower,upper)~numericScale(dr),group=as.factor(cond),data=d3.ucl,type="l",label.curves=F,main=paste("Uncertainty,Trials = ",max(d3$V1),sep=""),ylab="Fixations",xlab="Distractor Ratio")
#p4 = xYplot(Cbind(mean,lower,upper)~numericScale(dr),group=as.factor(cond),data=d4.ucl,type="l",label.curves=F,main=paste("Activation,Trials = ",max(d4$V1),sep=""),ylab="Fixations",xlab="Distractor Ratio")

#png("~/4models.png",width=800,height=600)
print(p1)
#print(p1,split=c(1,1,2,2),more=T)
#print(p2,split=c(2,1,2,2),more=T)
#print(p3,split=c(1,2,2,2),more=T)
#print(p4,split=c(2,2,2,2),more=F)
#dev.off()

Sys.sleep(15)

}
