require(data.table)
require(gridExtra)

# Smoking challenge
smk <- fread('experiments_smoking_challenge.csv',
             colClasses = c(ws_training="character", 
                            ws_test="character"))

test <- smk[fold=='test']
train <- smk[fold!='test']
