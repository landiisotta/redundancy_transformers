require(data.table)
require(gridExtra)
require(ggplot2)

# Smoking challenge
smk <- fread('experiments_smoking_challenge.csv',
             colClasses = c(ws_training="character", 
                            ws_test="character"))
config <- c("cl_method", "learning_rate", "val_split",      
            "window_size", "sequence_length", "batch_size")

feat <- c()
for (i in 1:nrow(smk)){
  feat <- c(feat, paste(smk[i, ..config], collapse='-'))
}
smk$feat <- feat

test <- smk[fold=='test']
train <- smk[fold!='test']

# Loss
loss <- reshape(train, varying=c('tr_loss', 'val_loss'), direction='long', 'loss')[1:6]
loss$fold <- rep(c('train', 'val'), nrow(loss)/2)
loss$feat <- apply(loss[, c('fold', 'feat')], 1, function(x) paste(x, collapse='-'))

p <- ggplot(data = loss, aes(x = epoch, y = loss, group = fold, 
                                colour = feat))
p + 
  geom_line(lwd=1) +
  geom_point(size=2)
  # scale_colour_discrete(name="Training set redundancy: ", 
  #                       labels=c("1% words - 5 sentences",
  #                                "5% words - 5 sentences",
  #                                "5% words - 10 sentences",
  #                                "10% words - 10 sentences",
  #                                "10% words - 20 sentences")) + 
  # theme(legend.position="right", aspect.ratio = 1/1) +
  # labs(y = "Relative PPL (%)", x = "Test set redundancy") +
  # scale_x_discrete(labels=c("00" = "0%w 0s", 
  #                           "55" = "5%w 5s", 
  #                           "15" = "1%w 5s",
  #                           "510" = "5%w 10s", 
  #                           "1010" = "10%w 10s", 
  #                           "1020" = "10%w 20s")) +
  # theme(text = element_text(size=12),
  #       axis.text.x = element_text(angle=45, hjust=1)) +
  # geom_smooth(method='lm', formula= y~x, se=FALSE, lty='dashed', lwd=0.5) +
  # stat_poly_eq(aes(label = after_stat(p.value.label)),
  #              formula = y~x, parse = TRUE, size = 5,
  #              label.x = 'right')
