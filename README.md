# CGATCPred
Files:

1.data
Ch_one.pckl Ch_two.pckl, Ch_three.pckl, Ch_four.pckl, Ch_five.pckl, Ch_six.pckl, Ch_seven.pckl store SMSim, SMExp, SMDat, SMTex, SMCom, SMcp and SMsub respectively.

Drug_ATC_label.pckl stores known drug-ATC code associations.

glove_wordEmbedding.pkl stores ATC label word embeddings.

2.Code

Extra_label_matrix.py: function computing label correlation matrix;

single_label.py: computing evaluation metric;

network_kfold.py: network framework;

cross_validation.py: ten-fold cross-validation function.

# Requirements
* python == 3.6
* pytorch == 1.6
* Numpy == 1.16.2
* scikit-learn == 0.21.3

# How to run our code
python cross_validation.py --rawdata_dir /Your path
