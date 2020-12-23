# CGATCPred
Files:
1.Dataset

1) lncSim.mat and disSim_Jaccard.mat store lncRNA similarity matrix and disease similarity matrix, respectively;

2) interMatrix.mat stores known lncRNA-disease association information;

3) lncRNA_Name.txt and diseases_Name.txt store lncRNA ids and disease ids, respectively;

2.Code
1) gKernel.m: function computing Gaussian interaction profile kernel;

2) pca_energy.m: function extracting feature vectors via PCA;

3) SIMC.m : function completing matrix;

4) SIMCLDA: predict potential lncRNA-disease associations; 
