1.  Load data

```{r Time-varying model}
library(installr)
library(devtools)
library(Rcpp)
library(pracma)
library(pec)
library(AUC)
library(risksetROC)
library(rfSLAM)#remotes::install_github("https://github.com/mattrosen/rfSLAM")
library(dplyr)
library(tidyverse)
library(magrittr)
library(dplyr)
library(pROC)
library(splines)
library(grid)
library(Rmisc)
library(gridExtra)
library(ggsci)
library(dplyr)

#load data
cohort = read.csv("cpiu_AT_AE_90days_shock.csv")
cohort <- dplyr::select(cohort, -X)

#Normalize numeric features
cols_to_scale <- c("Age", "BMI", "Sodium", "Potassium", "Kreatinine")
scaled_train_or = scale(cohort[,cols_to_scale])
cohort[,cols_to_scale] = scaled_train_or

formula <- as.formula(paste("EventDuringCPIU ~ Age+Sex+AtrialArrhythmia+OHCA+CRTD+SICD+VR+DR+LowLVEF+ReducedLVEF+NormalLVEF+PAF+ICM+NICM+HCM+DCM+VA+NSVT+PCI+QRS_Duration+CABG+MyocardialInfarction+CVA+COPD+DiabetesMellitus+BMI+Hypertension+CHD+Sodium+Potassium+Kreatinine+Implantation_indication+Vitamine_K+Antiaritmica_soort.Sotalol+Antiaritmica_soort.Digoxine+ARB+Antiaritmica_soort.Amiodarone+Betablokker+NOAC+Aldosteronremmer+0+1+2+3+4+5+6+7+8+9+10+11+12+13+14+15"))


cohort$EventDuringCPIU = as.factor(cohort$EventDuringCPIU)
cats        <- as.integer(c(rep(1, 44)))
cat_weights <- c(1.0) # arbitrary here, so it doesn't impact training
categories <- list()
categories$categories.weights <- cat_weights
categories$var.assignments    <- cats

cohort <- cohort %>%  
    mutate(PatientID = as.factor(PatientID))

train_patients_arr <- read.csv('train_patients.csv')$patient_id
test_patients_arr <- read.csv('test_patients.csv')$patient_id

# Convert patient IDs to factors
cohort <- cohort %>% 
  mutate(PatientID = as.factor(PatientID))

# Select corresponding patients from cohort
train_cohort <- cohort %>% filter(PatientID %in% train_patients_arr)
test_cohort <- cohort %>% filter(PatientID %in% test_patients_arr)
```

2.  Grid search model

```{r}
library(caret)
colnames(train_cohort)[colnames(train_cohort) == 'PatientID'] <- 'pid'
# Define a range of values for ntree and mtry
ntree_range <- c(50, 100, 150, 200)
mtry_range <- c(2, 4, 6)
nodize_range <- c(1.5, 2.5, 3)
poisson_range <- c('poisson.split1')

# Initialize an empty dataframe to store results
auc_df_fold <- data.frame()

# Perform grid search
for (ntree in ntree_range) {
  for (mtry in mtry_range) {
      for (node in nodize_range) 
            for (poisson in poisson_range) 

        {
      
      # Get unique Patient IDs
      unique_patients <- unique(train_cohort$pid)
      
      # Split the unique Patient IDs into train_patients and test_patients
      train_patients <- sample(unique_patients, floor(0.8 * length(unique_patients)), replace = FALSE)
      test_patients <- setdiff(unique_patients, train_patients)
      
      # Split the train data into train_cv and test_cv based on the train_patients and test_patients
      train_cv <- train_cohort[train_cohort$pid %in% train_patients, ]
      test_cv <- train_cohort[train_cohort$pid %in% test_patients, ]
      
      
      # Train the model with the current hyperparameters
      trained_model <- rfSLAM::rfsrc(formula, data = train_cv, ntree = ntree, na.action = "na.impute", splitrule = poisson, stratifier = train_cv$Interval, risk.time = train_cv$EventTimeDuringCPIU, mtry = mtry, nodesize = node , do.trace = TRUE, k_for_alpha = 2, membership = TRUE, seed = 4, bootsrap = "by.user", var.used = "by.tree", verbose=FALSE, importance=FALSE)
      
      # Predict probabilities on the test_cv fold
      p.cpiu.new <- risk.adjust.new.be(rf = trained_model, status = train_cv$EventDuringCPIU, rt = train_cv$EventTimeDuringCPIU, new_data = test_cv)
      test_cv$p.hat <- 1 - exp(-p.cpiu.new) # predicted event probabilities
      
      # Calculate AUCs of the model on the test_cv fold
      df_rf <- rf.auc(test_cv, 'EventDuringCPIU', 'Interval')
      
      # Calculate a mean AUC
      auc_fold_single <- auc.smooth.return.single(test_cv, df_rf, 'EventDuringCPIU', 'Interval')
      
      # Calculate AUC for each 60 days interval
      auc_fold_smooth <- auc.smooth(test_cv, df_rf, 'EventDuringCPIU', 'Interval')
      
      # Make dataframe with all AUC values
      auc_auc <- auc_fold_smooth['auc']
      auc_lci <- auc_fold_smooth['V4']
      auc_uci <- auc_fold_smooth['V6']
      auc_cpiu <- auc_fold_smooth['cpiu']
      auc_df_fold <- rbind(auc_df_fold, data.frame(fold = i, cpiu = auc_cpiu, auc = auc_auc, auc_lci = auc_lci, auc_uci = auc_uci, ntree = ntree, mtry = mtry, node = node, poisson = poisson))
      }
  }
}

# Calculate the mean of "auc" for each combination of "ntree" and "mtry"
means <- aggregate(auc ~ ntree + mtry + node + poisson, auc_df_fold, mean)
highest_auc_row <- means[which.max(means$auc), ]

# View the results
print(highest_auc_row)

```

3.  Internal cross-validation

```{r}
set.seed(321)
colnames(train_cohort)[colnames(train_cohort) == 'PatientID'] <- 'pid'

library(caret)
# Get unique patient IDs
patient_ids <- unique(train_cohort$pid)

# Create a vector of fold assignments for each patient ID
folds <- createFolds(patient_ids, k = 5, list = FALSE, returnTrain = TRUE)

# Add the fold column to the dataframe
train_cohort$fold <- folds[match(train_cohort$pid, patient_ids)]
train_cohort$risk <- NA 

# Create an empty dataframe to store the AUC values
auc_df <- data.frame(fold = numeric(), auc = numeric())
auc_df_fold <- data.frame(fold = numeric(), cpiu = numeric(), auc_auc = numeric(), auc_lci = numeric(), auc_uci = numeric())

for(i in 1:5){
  #Assign train and test folds
  testing <- train_cohort[train_cohort$fold == i,]
  training <- train_cohort[train_cohort$fold != i,]
  #Train the model on the trainings fold
   cv_model <- rfSLAM::rfsrc(formula, data = training, ntree = 100, na.action = "na.impute", splitrule = "poisson.split1", stratifier = training$Interval, risk.time = training$EventTimeDuringCPIU, mtry = 2, nodesize = 2.5 , do.trace = TRUE, k_for_alpha = 2, membership = TRUE, seed = 4, bootsrap = "by.user", var.used = "by.tree", verbose=FALSE, importance=FALSE)
   
  #Predict probabilities on the test fold
  p.cpiu.new <- risk.adjust.new.be(rf = cv_model, status = training$EventDuringCPIU, rt = training$EventTimeDuringCPIU, new_data = testing)
  testing$p.hat <- 1-exp(-p.cpiu.new) # predicted event probabilities
  
  #Calculate AUCs of the model on the test fold
  df_rf <- rf.auc(testing, 'EventDuringCPIU', 'Interval')
  
  #Calculate a mean AUC
  auc_fold_single <- auc.smooth.return.single(testing, df_rf, 'EventDuringCPIU', 'Interval')
  
  #Calculate AUC for each 60 days interval
  auc_fold_smooth <- auc.smooth(testing, df_rf, 'EventDuringCPIU', 'Interval')
  
  #Make dataframe with all AUC values
  auc_auc <- auc_fold_smooth['auc']
  auc_lci <- auc_fold_smooth['V4']
  auc_uci <- auc_fold_smooth['V6']
  auc_cpiu <-  auc_fold_smooth['cpiu']  
    auc_df_fold <- rbind(auc_df_fold, data.frame(fold = i, cpiu = auc_cpiu,  auc = auc_auc, auc_lci = auc_lci, auc_uci = auc_uci))

}
#Calculate mean AUCs for each fold
means_cv <- aggregate(auc_df_fold[, c('auc')], 
                       by = list(auc_df_fold$cpiu), 
                       FUN = function(x) mean(x, na.rm = TRUE))


std_cv <- aggregate(auc_df_fold[, c('auc')], 
                       by = list(auc_df_fold$cpiu), 
                       FUN = function(x) sd(x, na.rm = TRUE))


merged_df <- merge(means_cv, std_cv, by = "Group.1")
colnames(merged_df) <- c("cpiu", "mean_auc", "std_auc")

write.csv(merged_df, file = "RF-SLAM_Interval_validation_90.csv", row.names = FALSE)
```

4.  Train model

```{r}
#REAL RFSLAM
trained_model <- rfSLAM::rfsrc(formula, data = train_cohort, ntree = 100, na.action = "na.impute", splitrule = "poisson.split1", stratifier = train_cohort$Interval, risk.time = train_cohort$EventTimeDuringCPIU, mtry = 2, nodesize = 2.5 , do.trace = TRUE, k_for_alpha = 2, membership = TRUE, seed = 4, bootsrap = "by.user", var.used = "by.tree", verbose=FALSE, importance=FALSE)
#saveRDS(trained_model, "trained_model_90days.rds")
#trained_model <- readRDS("trained_model_90days.rds")


#RFSLAM BASELINE
#trained_model <- rfSLAM::rfsrc(formula, data = train_cohort, ntree = 100, na.action = "na.impute", #splitrule = "poisson.split1", risk.time = train_cohort$EventTimeDuringCPIU, mtry = 2, nodesize = 2.5 , #do.trace = TRUE, k_for_alpha = 2, membership = TRUE, seed = 4, bootsrap = "by.user", var.used = #"by.tree", verbose=FALSE, importance=FALSE)
#saveRDS(trained_model, "trained_model_90days.rds")
#trained_model <- readRDS("trained_model_90days.rds")
```


5.  Bootstrap on hold-out test

```{r}
# Set the number of bootstrap iterations
num_iterations <- iter

# Initialize an empty data frame to store the results
auc_df_fold <- data.frame()

n <- unique(test_cohort$PatientID)
n_sample <- round(0.80 * length(n)) # number of patient IDs to sample (rounded to the nearest integer)

for (i in 1:num_iterations) {
  test_idx <- sample(unique(test_cohort$PatientID), size = n_sample, replace = TRUE)
  test_bootstrap <- test_cohort[test_cohort$PatientID %in% test_idx, ]

  # Rename PatientID column to pid in the test set
  colnames(test_bootstrap)[colnames(test_bootstrap) == 'PatientID'] <- 'pid'

  # Predict probabilities on the test set using the trained model
  p.cpiu.new <- risk.adjust.new.be(rf = trained_model, status = train_cohort$EventDuringCPIU, rt =
                                     train_cohort$EventTimeDuringCPIU, new_data = test_bootstrap)
  
  test_bootstrap$p.hat <- 1 - exp(-p.cpiu.new) # predicted event probabilities
  
  # Calculate AUCs of the model on the test set
  df_rf <- rf.auc(test_bootstrap, 'EventDuringCPIU', 'Interval')
  
  # Calculate a mean AUC
  auc_fold_single <- auc.smooth.return.single(test_bootstrap, df_rf, 'EventDuringCPIU', 'Interval')
  auc_fold_smooth <- auc.smooth(test_bootstrap, df_rf, 'EventDuringCPIU', 'Interval')
  
  # Make dataframe with all AUC values for this iteration
  auc_auc <- auc_fold_smooth['auc']
  auc_lci <- auc_fold_smooth['V4']
  auc_uci <- auc_fold_smooth['V6']
  auc_cpiu <-  auc_fold_smooth['cpiu']  
  auc_df_iteration <- data.frame(fold = i, cpiu = auc_cpiu, auc = auc_auc, auc_lci = auc_lci, auc_uci = auc_uci)
  
  # Append the results of this iteration to the overall results data frame
  auc_df_fold <- rbind(auc_df_fold, auc_df_iteration)

}

means_bt <- aggregate(auc_df_fold[, c('auc')], 
                       by = list(auc_df_fold$cpiu), 
                       FUN = function(x) mean(x, na.rm = TRUE))

medians_bt <- aggregate(auc_df_fold[, c('auc')], 
                       by = list(auc_df_fold$cpiu), 
                       FUN = function(x) median(x, na.rm = TRUE))

std_bt <- aggregate(auc_df_fold[, c('auc')], 
                       by = list(auc_df_fold$cpiu), 
                       FUN = function(x) sd(x, na.rm = TRUE))

merged_df <- merge(means_bt, std_bt, by = "Group.1")

merged_df <- merged_df[merged_df$Group.1 < 40, ]

colnames(merged_df) <- c("cpiu", "mean_auc", "std_auc")

```

#6.0 Calibration plot

```{r}


df <- train_cohort

set.seed(321)
colnames(df)[colnames(df) == 'PatientID'] <- 'pid'

library(caret)
# Get unique patient IDs
patient_ids <- unique(df$pid)

# Create a vector of fold assignments for each patient ID
folds <- createFolds(patient_ids, k = 5, list = FALSE, returnTrain = TRUE)
# Add the fold column to the dataframe
df$fold <- folds[match(df$pid, patient_ids)]
df$risk <- NA 


# obtain predicted event rates
p.cpiu.be.ppl <- risk.adjust.be(rf = trained_model, status = train_cohort$EventDuringCPIU, rt = train_cohort$EventTimeDuringCPIU, alpha.tm = 0)

shuffled <- df
## 5-fold cross-validation for performance metric

set.seed(321)
shuffled$p.hat <- 1-exp(-p.cpiu.be.ppl) # predicted event probabilities 
shuffled$ni.sca <- as.numeric(shuffled$EventDuringCPIU)-1 #Binary predicted evnet rate


for(i in 1:5){ 
  
  testing <- shuffled[shuffled$fold == i,] 
  
  training <- shuffled[shuffled$fold != i,] 
  
  lr_model = glm(as.factor(ni.sca) ~ ns(p.hat,2)*ns(Interval,2),data = training, family = binomial)
  
  #p.hat.df ispredicted event probabilities
  p.hat.df <- data.frame(p.hat = testing$p.hat, Interval = testing$Interval)
  
  #Get the predicted probabilites on test
  lr_probs = predict(lr_model,  
                     newdata = p.hat.df, 
                     type = "response")
  
  shuffled[shuffled$fold == i,"risk"] <- lr_probs
  
} 

## calibration with smooth curve

db2 <- shuffled

g1_prim <-  mutate(db2, bin = ntile(risk, 10)) %>% 
  # Bin prediction into 10ths
  dplyr::group_by(bin) %>%
  dplyr::mutate(n = n(), # Get ests and CIs
         bin_pred = mean(as.numeric(risk)), 
         bin_prob = mean(as.numeric(EventDuringCPIU)-1), 
         se = sqrt((bin_prob * (1 - bin_prob)) / n), 
         ul = bin_prob + 1.96 * se, 
         ll = bin_prob - 1.96 * se) %>%
  ungroup() %>%
  ggplot(aes(x = bin_pred, y = bin_prob, ymin = ll, ymax = ul)) +
  geom_pointrange(size = 0.75, color = "grey40") +
  scale_y_continuous(limits = c(0, 0.5), breaks = seq(0, 0.5, by = 0.1)) +
  scale_x_continuous(limits = c(0, 0.5), breaks = seq(0, 0.5, by = 0.1)) +
  geom_abline() +
  xlab("") +
  ylab("\nObserved Probability") +
  theme_bw() +theme(legend.position = "bottom",
        axis.title.x = element_text(vjust = -0.5, size=24),
        axis.title.y = element_text(hjust = 0.5, vjust=5, size =24),
      axis.text = element_text(size = 24)) + 
  theme(plot.title = element_text(hjust = 0.5,face = "bold")) +
  ggtitle("")

# 
g2_prim <- ggplot(db2, aes(x = risk)) +
  geom_histogram(fill = "grey40", bins = 200) +
  scale_x_continuous(limits = c(0, 0.5), breaks = seq(0, 0.5, by = 0.1)) +
  xlab("\nPredicted Probability") +
  ylab("") +
  theme_bw() +
  scale_y_continuous(limits = c(0, 4000),breaks = seq(0, 4000, by = 500)) +
   theme(panel.grid.minor = element_blank(), 
        text = element_text(size=24),
        panel.background = element_rect(fill = "transparent"),
        plot.background = element_rect(fill = "transparent"),
        panel.border = element_rect(colour = "black", fill=NA, size=1))
g_prim <- arrangeGrob(g1_prim, g2_prim, respect = TRUE, heights = c(1, 0.6), ncol = 1)

path <- paste0("Calibration_plot.tiff")

ggsave(path, plot = g_prim, dpi = 300, width = 10, height = 12)
