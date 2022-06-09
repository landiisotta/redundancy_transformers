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
for (trredu in c('05', '010', '020', '10', '15', '110', '120', '50', '55', '510', '520', '100', '105', '1010', '1020')){
  for (tsredu in c('00', '05', '010', '020', '10', '15', '110', '120', '50', '55', '510', '520', '100', '105', '1010', '1020')){
    n1 <- res[tr_redu=='00' & ts_redu==tsredu, ppl]
    n2 <- res[tr_redu==trredu & ts_redu==tsredu, ppl]
    results <- rbind(results, data.table(i, trredu, tsredu, ((n2 - n1)/n1)*100))
  }
  i <- i+1
}

names(results) <- c('model', 'tr_redu', 'ts_redu', 'relative_ppl')
results

results$tr_redu <- factor(results$tr_redu, levels = c('05', '010', '020', '10', '15', '110', '120', '50', '55', '510', '520', '100', '105', '1010', '1020'))
results$ts_redu <- factor(results$ts_redu, levels = c('00', '05', '010', '020', '10', '15', '110', '120', '50', '55', '510', '520', '100', '105', '1010', '1020'))
results$model <- factor(results$model, levels = c('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'))

results

p <- ggplot(data = results, aes(x = ts_redu, y = relative_ppl, group = tr_redu, 
                                colour = model))
p + 
  geom_line(lwd=1) +
  geom_point(size=2) +
  scale_colour_discrete(name="Training set redundancy: ", 
                        labels=c("0% words - 5 sentences",
                                 "0% words - 10 sentences",
                                 "0% words - 20 sentences",
                                 "1% words - 0 sentences",
                                 "1% words - 5 sentences",
                                 "1% words - 10 sentences",
                                 "1% words - 20 sentences",
                                 "5% words - 0 sentences",
                                 "5% words - 5 sentences",
                                 "5% words - 10 sentences",
                                 "5% words - 20 sentences",
                                 "10% words - 0 sentences",
                                 "10% words - 5 sentences",
                                 "10% words - 10 sentences",
                                 "10% words - 20 sentences")) + 
  theme(legend.position="right", aspect.ratio = 1/1) +
  labs(y = "Relative PPL (%)", x = "Test set redundancy") +
  scale_x_discrete(labels=c("00" = "0%w 0s", 
                            "05" = "0%w 5s",
                            "010" = "0%w 10s",
                            "020" = "0%w 20s", 
                            "10" = "1%w 0s", 
                            "15" = "1%w 5s",
                            "110" = "1%w 10s",
                            "120" = "1%w 20s",
                            "50" = "5%w 0s", 
                            "55" = "5%w 5s", 
                            "510" = "5%w 10s",
                            "520" = "5%w 20s",
                            "100" = "10%w 0s",
                            "105" = "10%w 5s",
                            "1010" = "10%w 10s", 
                            "1020" = "10%w 20s")) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1)) +
  geom_smooth(method='lm', formula= y~x, se=FALSE, lty='dashed', lwd=0.5) +
  stat_poly_eq(aes(label = after_stat(p.value.label)),
               formula = y~x, parse = TRUE, size = 5,
               label.x = 'right')

# Fit regression line
ppl05 <- results[tr_redu=='05']$relative_ppl
ppl010 <- results[tr_redu=='010']$relative_ppl
ppl020 <- results[tr_redu=='020']$relative_ppl
ppl10 <- results[tr_redu=='10']$relative_ppl
ppl15 <- results[tr_redu=='15']$relative_ppl
ppl110 <- results[tr_redu=='110']$relative_ppl
ppl120 <- results[tr_redu=='120']$relative_ppl
ppl50 <- results[tr_redu=='50']$relative_ppl
ppl55 <- results[tr_redu=='55']$relative_ppl
ppl510 <- results[tr_redu=='510']$relative_ppl
ppl520 <- results[tr_redu=='520']$relative_ppl
ppl100 <- results[tr_redu=='100']$relative_ppl
ppl105 <- results[tr_redu=='105']$relative_ppl
ppl1010 <- results[tr_redu=='1010']$relative_ppl
ppl1020 <- results[tr_redu=='1020']$relative_ppl
x <- c(1:16)

m1 <- summary(lm(ppl05 ~ x))
m2 <- summary(lm(ppl010 ~ x))
m3 <- summary(lm(ppl020 ~ x))
m4 <- summary(lm(ppl10 ~ x))
m5 <- summary(lm(ppl15 ~ x))
m6 <- summary(lm(ppl110 ~ x))
m7 <- summary(lm(ppl120 ~ x))
m8 <- summary(lm(ppl50 ~ x))
m9 <- summary(lm(ppl55 ~ x))
m10 <- summary(lm(ppl510 ~ x))
m11 <- summary(lm(ppl520 ~ x))
m12 <- summary(lm(ppl100 ~ x))
m13 <- summary(lm(ppl105 ~ x))
m14 <- summary(lm(ppl1010 ~ x))
m15 <- summary(lm(ppl1020 ~ x))
