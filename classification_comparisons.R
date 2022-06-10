require(data.table)
require(gridExtra)

# Smoking challenge
smk <- fread('experiments_smoking_challenge.csv',
             colClasses = c(ws_training="character", 
                            ws_test="character"))
smk_test <- smk[fold=='test']
smk_test$ws_training <- factor(smk_test$ws_training, levels=c('00', '15', '55',
                                                              '510', '1010', '1020'))
smk_test$ws_test <- factor(smk_test$ws_test, levels=c('00', '15', '55',
                                                              '510', '1010', '1020'))

g1 <- ggplot(data=smk_test, aes(x=ws_test, y=f1_macro, fill=ws_training)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=F) +
  labs(y = "F1 macro", x = "", 
       fill = "Training set redundancy: ") +
  scale_x_discrete(labels=c("00" = "0%w 0s", 
                            "55" = "5%w 5s", 
                            "15" = "1%w 5s",
                            "510" = "5%w 10s", 
                            "1010" = "10%w 10s", 
                            "1020" = "10%w 20s")) +
  scale_fill_discrete(name="Training set redundancy: ", 
                      labels=c("0% words - 0 sentences", 
                               "1% words - 5 sentences",
                               "5% words - 5 sentences",
                               "5% words - 10 sentences",
                               "10% words - 10 sentences",
                               "10% words - 20 sentences")) +
  theme(text = element_text(size=24),
        axis.text.x = element_text(angle=45, hjust=1))


g2 <- ggplot(data=smk_test, aes(x=ws_test, y=f1_micro, fill=ws_training)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=FALSE) +
  labs(y = "F1 micro", x = "Test set redundancy") +
  scale_x_discrete(labels=c("00" = "0%w 0s", 
                            "55" = "5%w 5s", 
                            "15" = "1%w 5s",
                            "510" = "5%w 10s", 
                            "1010" = "10%w 10s", 
                            "1020" = "10%w 20s")) +
  scale_fill_discrete(name="Training set redundancy: ", 
                      labels=c("0% words - 0 sentences", 
                               "1% words - 5 sentences",
                               "5% words - 5 sentences",
                               "5% words - 10 sentences",
                               "10% words - 10 sentences",
                               "10% words - 20 sentences")) +
  theme(text = element_text(size=24),
        axis.text.x = element_text(angle=45, hjust=1))

grid.arrange(g1, g2, nrow = 1)

classes <- c('NON-SMOKER', 'CURRENT SMOKER', 'SMOKER', 'PAST SMOKER', 'UNKNOWN')

gnonsmoker <- ggplot(data=smk_test, aes(x=ws_test, y=f1_class0, fill=ws_training)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=FALSE) +
  ggtitle("Non-smoker") +
  labs(y = "F1", x = "") +
  scale_x_discrete(labels=c("00" = "0%w 0s", 
                            "55" = "5%w 5s", 
                            "15" = "1%w 5s",
                            "510" = "5%w 10s", 
                            "1010" = "10%w 10s", 
                            "1020" = "10%w 20s")) +
  scale_fill_discrete(name="Training set redundancy: ", 
                      labels=c("0% words - 0 sentences", 
                               "1% words - 5 sentences",
                               "5% words - 5 sentences",
                               "5% words - 10 sentences",
                               "10% words - 10 sentences",
                               "10% words - 20 sentences")) +
  theme(text = element_text(size=19),
        axis.text.x = element_text(angle=45, hjust=1))

gcsmoker <- ggplot(data=smk_test, aes(x=ws_test, y=f1_class1, fill=ws_training)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=FALSE) +
  ggtitle("Current smoker") +
  labs(y = "", x = "Test set redundancy") +
  scale_x_discrete(labels=c("00" = "0%w 0s", 
                            "55" = "5%w 5s", 
                            "15" = "1%w 5s",
                            "510" = "5%w 10s", 
                            "1010" = "10%w 10s", 
                            "1020" = "10%w 20s")) +
  scale_fill_discrete(name="Training set redundancy: ", 
                      labels=c("0% words - 0 sentences", 
                               "1% words - 5 sentences",
                               "5% words - 5 sentences",
                               "5% words - 10 sentences",
                               "10% words - 10 sentences",
                               "10% words - 20 sentences")) +
  theme(text = element_text(size=19),
        axis.text.x = element_text(angle=45, hjust=1))

gsmoker <- ggplot(data=smk_test, aes(x=ws_test, y=f1_class2, fill=ws_training)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=FALSE) +
  ggtitle("Smoker") +
  labs(y = "", x = "") +
  scale_x_discrete(labels=c("00" = "0%w 0s", 
                            "55" = "5%w 5s", 
                            "15" = "1%w 5s",
                            "510" = "5%w 10s", 
                            "1010" = "10%w 10s", 
                            "1020" = "10%w 20s")) +
  scale_fill_discrete(name="Training set redundancy: ", 
                      labels=c("0% words - 0 sentences", 
                               "1% words - 5 sentences",
                               "5% words - 5 sentences",
                               "5% words - 10 sentences",
                               "10% words - 10 sentences",
                               "10% words - 20 sentences")) +
  theme(text = element_text(size=19),
        axis.text.x = element_text(angle=45, hjust=1))

gpsmoker <- ggplot(data=smk_test, aes(x=ws_test, y=f1_class3, fill=ws_training)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=FALSE) +
  ggtitle("Past smoker") +
  labs(y = "F1", x = "") +
  scale_x_discrete(labels=c("00" = "0%w 0s", 
                            "55" = "5%w 5s", 
                            "15" = "1%w 5s",
                            "510" = "5%w 10s", 
                            "1010" = "10%w 10s", 
                            "1020" = "10%w 20s")) +
  scale_fill_discrete(name="Training set redundancy: ", 
                      labels=c("0% words - 0 sentences", 
                               "1% words - 5 sentences",
                               "5% words - 5 sentences",
                               "5% words - 10 sentences",
                               "10% words - 10 sentences",
                               "10% words - 20 sentences")) +
  theme(text = element_text(size=19),
        axis.text.x = element_text(angle=45, hjust=1))

gusmoker <- ggplot(data=smk_test, aes(x=ws_test, y=f1_class4, fill=ws_training)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=F) +
  ggtitle("Unknown") +
  labs(y = "", x = "Test set redundancy") +
  scale_x_discrete(labels=c("00" = "0%w 0s", 
                            "55" = "5%w 5s", 
                            "15" = "1%w 5s",
                            "510" = "5%w 10s", 
                            "1010" = "10%w 10s", 
                            "1020" = "10%w 20s")) +
  scale_fill_discrete(name="Training set redundancy: ", 
                      labels=c("0% words - 0 sentences", 
                               "1% words - 5 sentences",
                               "5% words - 5 sentences",
                               "5% words - 10 sentences",
                               "10% words - 10 sentences",
                               "10% words - 20 sentences")) +
  theme(text = element_text(size=19),
        axis.text.x = element_text(angle=45, hjust=1))

grid.arrange(gnonsmoker, gcsmoker, gsmoker, gpsmoker, gusmoker, nrow=2)


# Cohort selection
cohort_met <- fread('./experiments_cohort_selection_challenge_MET.csv', 
             colClasses = c(ws_training="character", 
                            ws_test="character"))
cohort_notmet <- fread('./experiments_cohort_selection_challenge_NOTMET.csv', 
                    colClasses = c(ws_training="character", 
                                   ws_test="character"))
cohort_met$label <- rep("MET", nrow(cohort_met))
cohort_notmet$label <- rep("NOTMET", nrow(cohort_notmet))

cohort <- rbind(cohort_met, cohort_notmet)
cohort_test <- cohort[fold=='test']

cohort_test <- subset(cohort_test, 
                      select = c('ws_training', 'ws_test', 'label', 
                                 names(cohort_test)[grep('f1_class', names(cohort_test))]))
cohort_test$ws_training <- factor(cohort_test$ws_training, levels=c('00', '15', '55',
                                                              '510', '1010', '1020'))
cohort_test$ws_test <- factor(cohort_test$ws_test, levels=c('00', '15', '55',
                                                      '510', '1010', '1020'))

fscores <- cohort_test[, .(mean(f1_class0), 
                mean(f1_class1), 
                mean(f1_class2),
                mean(f1_class3),
                mean(f1_class4),
                mean(f1_class5),
                mean(f1_class6),
                mean(f1_class7),
                mean(f1_class8),
                mean(f1_class9),
                mean(f1_class10),
                mean(f1_class11),
                mean(f1_class12)), 
            by = .(ws_training, ws_test)]
names(fscores) <- c('ws_training', 'ws_test', names(cohort)[grep('f1_class', names(cohort))])
fscores$fmean <- apply(fscores[,3:15], 1, mean)
fscores

gcohort <- ggplot(data=fscores, aes(x=ws_test, y=fmean, fill=ws_training)) + 
  geom_bar(stat="identity", position=position_dodge()) +
  labs(y = "", x = "Test set redundancy") +
  scale_x_discrete(labels=c("00" = "0%w 0s", 
                            "55" = "5%w 5s", 
                            "15" = "1%w 5s",
                            "510" = "5%w 10s", 
                            "1010" = "10%w 10s", 
                            "1020" = "10%w 20s")) +
  scale_fill_discrete(name="Training set redundancy: ", 
                      labels=c("0% words - 0 sentences", 
                               "1% words - 5 sentences",
                               "5% words - 5 sentences",
                               "5% words - 10 sentences",
                               "10% words - 10 sentences",
                               "10% words - 20 sentences")) +
  theme(text = element_text(size=19),
        axis.text.x = element_text(angle=45, hjust=1))
gcohort


# Hyperparameters (epochs, window size, learning rate) selection
data <- fread('experiments_smoking_challenge.csv', colClasses = c(ws_training="character",
                                                                  ws_test="character",
                                                                  epoch="character"),
              fill = TRUE)
data[ , fac := do.call(paste, c(.SD, sep = "::")), 
      .SDcols=c("cl_method", "learning_rate", "window_size")]
data
trval <- data[fold=='train/val']
test <- data[fold=='test']

lossdt <- data.table()
lossdt$fold <- rep(c('train', 'val'), each=nrow(trval))
lossdt$epochs <- c(trval$epoch, trval$epoch)
lossdt$loss <-  c(trval$tr_loss, trval$val_loss)
lossdt$fac <- c(trval$fac, trval$fac)

# Train/val loss
plots <- list()
i <- 1
for (g in sort(unique(lossdt$fac))){
  p <- ggplot(data = lossdt[fac==g], aes(x = epochs, y = loss, group = fold, 
                                 colour = fold)) + 
    geom_line(lwd=1) +
    geom_point(size=2) +
    ggtitle(g)
  plots[[i]] <- p
  i <- i+1
}
do.call(grid.arrange, plots)


# Test set performance
g1 <- ggplot(data=test, aes(x=fac, y=f1_macro)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=F) +
  geom_text(aes(label=round(f1_macro, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# g1

g2 <- ggplot(data=test, aes(x=fac, y=f1_micro)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=F) +
  geom_text(aes(label=round(f1_macro, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# g2

g3 <- ggplot(data=test, aes(x=fac, y=p_macro)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=F) +
  geom_text(aes(label=round(f1_macro, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# g3

g4 <- ggplot(data=test, aes(x=fac, y=p_micro)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=F) +
  geom_text(aes(label=round(f1_macro, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# g4

g5 <- ggplot(data=test, aes(x=fac, y=r_macro)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=F) +
  geom_text(aes(label=round(f1_macro, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# g5

g6 <- ggplot(data=test, aes(x=fac, y=r_micro)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=F) +
  geom_text(aes(label=round(f1_macro, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# g6

grid.arrange(g1, g2, g3, g4, g5, g6, nrow = 3)



classes <- c('NON-SMOKER', 'CURRENT SMOKER', 'SMOKER', 'PAST SMOKER', 'UNKNOWN')

gnonsmoker <- ggplot(data=test, aes(x=fac, y=f1_class0)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=FALSE) +
  ggtitle("Non-smoker") +
  labs(y = "F1", x = "") +
  geom_text(aes(label=round(f1_class0, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# gnonsmoker

gcsmoker <- ggplot(data=test, aes(x=fac, y=f1_class1)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=FALSE) +
  ggtitle("Current smoker") +
  labs(y = "F1", x = "") +
  geom_text(aes(label=round(f1_class1, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# gcsmoker

gsmoker <- ggplot(data=test, aes(x=fac, y=f1_class2)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=FALSE) +
  ggtitle("Smoker") +
  labs(y = "F1", x = "") +
  geom_text(aes(label=round(f1_class2, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# gsmoker

gpsmoker <- ggplot(data=test, aes(x=fac, y=f1_class3)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=FALSE) +
  ggtitle("Past smoker") +
  labs(y = "F1", x = "") +
  geom_text(aes(label=round(f1_class3, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# gpsmoker

gusmoker <- ggplot(data=test, aes(x=fac, y=f1_class4)) + 
  geom_bar(stat="identity", position=position_dodge(), show.legend=FALSE) +
  ggtitle("Unknown") +
  labs(y = "F1", x = "") +
  geom_text(aes(label=round(f1_class4, 2)), 
            position=position_dodge(width=0.9), 
            vjust=-0.25, angle=45, size=3) +
  theme(text = element_text(size=12),
        axis.text.x = element_text(angle=45, hjust=1))
# gusmoker

grid.arrange(gnonsmoker, gcsmoker, gsmoker, gpsmoker, gusmoker, nrow=2)
