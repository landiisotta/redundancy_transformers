# Redundancy PPL comparisons
require(data.table)
require(ggplot2)
require(ggpmisc)

res <- fread('./experiments.txt',
             colClasses = c("character", "character", "numeric"))
names(res) <- c('tr_redu', 'ts_redu', 'ppl')

results <- data.table()
results
i <- 1
for (trredu in c('15', '55', '510', '1010', '1020')){
  for (tsredu in c('00', '15', '55', '510', '1010', '1020')){
    n1 <- res[tr_redu=='00' & ts_redu==tsredu, ppl]
    n2 <- res[tr_redu==trredu & ts_redu==tsredu, ppl]
    results <- rbind(results, data.table(i, trredu, tsredu, ((n2 - n1)/n1)*100))
  }
  i <- i+1
}

names(results) <- c('model', 'tr_redu', 'ts_redu', 'relative_ppl')
results

results$tr_redu <- factor(results$tr_redu, levels = c('15', '55', '510', '1010', '1020'))
results$ts_redu <- factor(results$ts_redu, levels = c('00', '15', '55', '510', '1010', '1020'))
results$model <- factor(results$model, levels = c('1', '2', '3', '4', '5', '6'))

results

p <- ggplot(data = results, aes(x = ts_redu, y = relative_ppl, group = tr_redu, 
                                colour = model))
p + 
  geom_line(lwd=1) +
  geom_point(size=2) +
  scale_colour_discrete(name="Training set redundancy: ", 
                        labels=c("1% words - 5 sentences",
                                 "5% words - 5 sentences",
                                 "5% words - 10 sentences",
                                 "10% words - 10 sentences",
                                 "10% words - 20 sentences")) + 
  theme(legend.position="right", aspect.ratio = 1/1) +
  labs(y = "Relative PPL (%)", x = "Test set redundancy") +
  scale_x_discrete(labels=c("00" = "0%w 0s", 
                            "55" = "5%w 5s", 
                            "15" = "1%w 5s",
                            "510" = "5%w 10s", 
                            "1010" = "10%w 10s", 
                            "1020" = "10%w 20s")) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1)) +
  geom_smooth(method='lm', formula= y~x, se=FALSE, lty='dashed', lwd=0.5) +
  stat_poly_eq(aes(label = after_stat(p.value.label)),
               formula = y~x, parse = TRUE, size = 5,
               label.x = 'right')

# Fit regression line
ppl15 <- results[tr_redu=='15']$relative_ppl
ppl55 <- results[tr_redu=='55']$relative_ppl
ppl510 <- results[tr_redu=='510']$relative_ppl
ppl1010 <- results[tr_redu=='1010']$relative_ppl
ppl1020 <- results[tr_redu=='1020']$relative_ppl
x <- c(1:6)

m1 <- summary(lm(ppl15 ~ x))
m2 <- summary(lm(ppl55 ~ x))
m3 <- summary(lm(ppl510 ~ x))
m4 <- summary(lm(ppl1010 ~ x))
m5 <- summary(lm(ppl1020 ~ x))
