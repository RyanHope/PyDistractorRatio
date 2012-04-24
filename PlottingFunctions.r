library(sciplot)



lineplot.CI(d1$ratio,d1$nFix,group=list(d1$Gauss_Blur),cex=1.2,lty=1,type="l",xlab="DR",ylab="Mean Fixations",err.width=.05,cex.lab=1.5,col=2,legend=TRUE,lwd=1.8,main="Title",cex.main=1.8,cex.axis=2,ylim=c(0,30))

lineplot.CI(d1$ratio,d1$nFix,group=list(d1$Gauss_Blur),cex=1.2,lty=1,type="l",xlab="DR",ylab="Mean Fixations",err.width=.05,cex.lab=1.5,col=c(1:10),legend=TRUE,lwd=1.8,main="Title",cex.main=1.8,cex.axis=2,ylim=c(0,30),fixed=TRUE)

