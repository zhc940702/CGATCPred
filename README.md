# CGATCPred
Files:


1.data.rar
Ch_one.txt Ch_two.txt, Ch_three.txt, Ch_four.txt, Ch_five.txt, Ch_six.txt, Ch_seven.txt store SMSim, SMExp, SMDat, SMTex, SMCom, SMcp and SMsub respectively.

Drug_ATC_label.pckl stores known drug-ATC code associations.

glove_wordEmbedding.pkl stores ATC label word embeddings.

2.Code

Extra_label_matrix.py: function computing label correlation matrix;

single_label.py: computing evaluation metric;

network_kfold.py: network framework;

cross_validation.py: ten-fold cross-validation function.

