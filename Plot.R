######### 1 Creating Forest plot  #############

#Setup environment
rm(list = ls())
load("~/R.Config.Rdata")
setwd(master.location)
setwd(paste0(master.location,"/Zainab-Project - Cancer TARGET"))
# Install packages and load
source(paste0(toolbox.path,"/R scripts/ikin.function.R"))
required.packages <- c("stringr", "survival","RColorBrewer", "forestplot")
ikin(required.packages)

# Set parameters
Cancer = "PanCancer"
Gene.set = "Selected.pathways"
Surv.cutoff.years = 10
#cluster= "S4"

# Load data
clinical_data = read.csv(paste0("./Data/Clinical_Data/TARGET-AllCancers_clinicalL.csv"))
load(paste0("./Analysis/after_split/Signature_Enrichment/GSEA_pediatric_onlySelected.pathways.Rdata"))
load("./Analysis/after_split/Signature_Enrichment/Thorsson_clustering/030.Thorsson_samples_cancers_10000repeats_HM_clusters_km6_pediatric_only_V3.Rdata")
#annotation = annotation[which(annotation$cluster %in% cluster),]
ES = ES[,which(colnames(ES) %in% annotation_all$Sample)]
#clinical_data = clinical_data[which(clinical_data$submitter_id %in% annotation$Sample),]

load(paste0("./Analysis/after_split/Signature_Enrichment/order_signatures/",Gene.set,".sig.p.value.Rdata"))


clinical_data = clinical_data [!duplicated(clinical_data$submitter_id),]

rownames(clinical_data) = clinical_data$submitter_id

ES = t(ES)

for (i in 1:ncol(ES)){
  col = colnames(ES)[i]
  ES[, col] = (ES[, col] - min(ES[, col]))/(max(ES[,col])-min(ES[,col]))
}

clinical_data = merge(clinical_data, ES, by = "row.names")

HR_table = data.frame(Signature = colnames(ES), p_value = NA, HR = NA, CI_lower = NA, CI_upper = NA)

i=1
for (i in 1:ncol(ES)){
  Group.of.interest = colnames(ES)[i]
  Y = Surv.cutoff.years * 365
  # time / event object creation
  TS.Alive = clinical_data[clinical_data$vital_status == "Alive", c("vital_status", "days_to_last_follow_up", Group.of.interest)]
  colnames(TS.Alive) = c("Status","Time", Group.of.interest)
  TS.Alive$Time = as.numeric(as.character(TS.Alive$Time))
  TS.Alive$Time[TS.Alive$Time > Y] = Y
  
  TS.Dead = clinical_data[clinical_data$vital_status == "Dead", c("vital_status", "days_to_last_follow_up", Group.of.interest)]
  colnames(TS.Dead) = c("Status","Time", Group.of.interest)
  TS.Dead$Time = as.numeric(as.character(TS.Dead$Time))
  TS.Dead$Status[which(TS.Dead$Time> Y)] = "Alive"
  TS.Dead$Time[TS.Dead$Time > Y] = Y
  
  TS.Surv = rbind (TS.Dead,TS.Alive)
  TS.Surv$Time = as.numeric(as.character(TS.Surv$Time))
  TS.Surv$Status <- TS.Surv$Status == "Dead"
  TS.Surv = subset(TS.Surv,TS.Surv$Time > 1)                               # remove patients with less then 1 day follow up time
  
  uni_variate = coxph(formula = Surv(Time, Status) ~ get(Group.of.interest), data = TS.Surv)
  summary = summary(uni_variate)
  HR = summary$conf.int[1]
  CI_lower = summary$conf.int[3]
  CI_upper = summary$conf.int[4]
  p_value = summary$coefficients[5]
  HR_table$p_value[which(HR_table$Signature == Group.of.interest)] = p_value
  HR_table$CI_lower[which(HR_table$Signature == Group.of.interest)] = CI_lower
  HR_table$CI_upper[which(HR_table$Signature == Group.of.interest)] = CI_upper
  HR_table$HR[which(HR_table$Signature == Group.of.interest)] = HR
}

#write.csv(HR_table,file = paste0("./Analysis/after_split/Survival_Analysis/Figure.5A.HR_table_",Cancer,"_ES_",Gene.set,"_cutoff_", Surv.cutoff.years,".csv"))

HR_table$Signature = gsub(".*] ", "",HR_table$Signature)
HR_table$Signature = gsub("SHC1/pSTAT3 Signature", "SHC1 pSTAT3 Signature",HR_table$Signature)
HR_table$Signature = gsub("Hypoxia/Adenosine Immune Cell Suppression", "Hypoxia Adenosine Immune Cell Suppression",HR_table$Signature)
HR_table$Signature = gsub("[()]","",HR_table$Signature)

#load(paste0("./Analysis/after_split/Signature_Enrichment/order_signatures/",Gene.set,".sig.p.value.Rdata"))
HR_table  = HR_table[which(HR_table$Signature %in% order_signatures),]

#dir.create("./Analysis/after_split/Survival_Analysis", showWarnings = FALSE)
#save(HR_table, file = paste0("./Analysis/after_split/Survival_Analysis/HR_table_",Cancer,"_ES_",Gene.set,"_cutoff_", Surv.cutoff.years,".Rdata"))

#Create the forest plot
#load(paste0("./Analysis/after_split/Survival_Analysis/HR_table_",Cancer,"_ES_",Gene.set,"_cutoff_", Surv.cutoff.years,".Rdata"))
HR_table = HR_table[order(HR_table$HR),]



## Forest plot seperate script
n_Signature = nrow(HR_table)
x = n_Signature + 2

HR.matrix = as.matrix(HR_table)
rownames(HR.matrix) = HR.matrix[,1]
HR.matrix = HR.matrix[,-c(1)]

mode(HR.matrix) = "numeric"

HR_table = HR_table[order(HR_table$HR),]

# Cochrane data from the 'rmeta'-package
cochrane_from_rmeta =
  structure(list(
    mean  = as.numeric(c(NA,HR_table$HR[1:n_Signature]), NA),
    lower = c(NA,HR_table$CI_lower[c(1:n_Signature)], NA),
    upper = c(NA,HR_table$CI_upper[c(1:n_Signature)], NA)),
    Names = c("mean", "lower", "upper"),
    row.names = c(NA, -x),
    class = "data.frame")



HR_table$p_value = signif(HR_table$p_value, 3)
HR_table$HR = signif(HR_table$HR, 3)
tabletext<-cbind(
  c("Signature", as.character(HR_table$Signature)[c(1:n_Signature)]),
  c("p-value", HR_table$p_value[c(1:n_Signature)]),
  c("HR",      HR_table$HR[c(1:n_Signature)]))

genes_order = rownames(HR.matrix)
save(genes_order,file = paste0("./Analysis/after_split/Signature_Enrichment/",Gene.set,"_genes_order.Rdata"))

dir.create(paste0("./Figures/after_split/Forest_plots/",Gene.set,"_forest.plots"), showWarnings = FALSE)
pdf(file = paste0("./Figures/after_split/Forest_plots/",Gene.set,"_forest.plots/",Gene.set,"_subtypes ",Cancer,"_Forest_plot_ES",Gene.set,"_cutoff_", Surv.cutoff.years, ".pdf"),
    height = 7, width = 7)
dev.new()
forestplot(mean = HR.matrix[,"HR"],
           lower = HR.matrix[,"CI_lower"],
           upper = HR.matrix[,"CI_upper"],
           labeltext = tabletext[-1,],
           new_page = FALSE,
           zero = 1,
           # is.summary=c(TRUE,rep(FALSE,n_Signature),TRUE,rep(FALSE,n_Signature),TRUE,FALSE),
           #    clip=c(0.001,2000),
           xlog=TRUE,
           #  xlim = c(0, 4 , 8 ,20),
           boxsize = .115,
           vertices = FALSE,
           col=fpColors(box="darkblue",line="darkgrey"),
           txt_gp = fpTxtGp(label = gpar(fontsize = 8), xlab = gpar(fontsize = 8),
                            ticks = gpar(fontsize = 10))
)
dev.off()
