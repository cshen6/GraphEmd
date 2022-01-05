#pacman::p_load(irlba, igraph, mclust, Matrix, lubridate, scales, vegan, broom, tidyverse, gmmase, ggrepel)

suppressMessages(require(igraph))
suppressMessages(require(mclust))
suppressMessages(require(tidyverse))
suppressMessages(require(doParallel))
registerDoParallel(detectCores()/2)

# Build a 3-SBM
nk <- 150
K <- 3
nrho <- rep(nk, K)
n <- sum(nrho)
set.seed(1234)

B <- rbind(c(.1,.01,.05),
           c(.01,.1,.025),
           c(.05,.025,.15))

rho <- rep(1/3,K)
Y <- rep(1:K, each=n/K)


source("GraphEncoder.R")

nmc <- 10

out <- matrix(0, nmc, 7)
for (i in 1:nmc) {
  set.seed(123+i)
  g <- sample_sbm(n, B, nrho); A <- g[]
  tm.km1 <- system.time(km1 <- GraphEncoder(A, 3))[3]
  tm.kmK <- system.time(kmK <- GraphEncoder(A, 2:5))[3]
  out[i,] <-  c(i, max(km1$Y), max(kmK$Y), adjustedRandIndex(km1$Y,Y), adjustedRandIndex(kmK$Y,Y), tm.km1, tm.kmK)
}

out <- as.data.frame(out)
names(out) <- c("i", "K.km3", "K.km2:5", "ari.km3", "ari.km2:5", "tm.km3", "tm.km2:5")

out %>% select(i, contains("K.")) %>% gather("method","Khat", -i) %>%
    ggplot(aes(x=method, y=Khat, color=method, fill=method)) +
    geom_jitter(width=0.2, height=0, alpha=0.9) +
#    geom_boxplot(width=0.1, notch = TRUE, color="black", fill=NA, alpha=0.5, outlier.size=0) +
    geom_boxplot(notch = TRUE, alpha=0.5, outlier.size=0) + theme(legend.position = "none")
