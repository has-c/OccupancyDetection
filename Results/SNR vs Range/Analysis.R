library(s20x)
df <- read.csv("snrVSRange.csv")

#initial plot
plot(x = df$Range,y = df$SNR)

#log transform response 
plot(x = df$Range,y = log(df$SNR))

#fit multiplicative model
snrRange.fit = lm(log(SNR) ~ Range, data=df)

#check model assumptions
plot(snrRange.fit, which=1) #constant scatter
normcheck(snrRange.fit) #normally distributed
cooks20x(snrRange.fit) #no unduly influential points

#model summary
summary(snrRange.fit)

#find confidence interval of various 
predDf <- data.frame(Range = c(1,3,5))
predict(snrRange.fit, predDf, interval = "confidence")
