Our project is built on top of mmaction2. To reproduce our results, please use the following commands:

AVEC 2014 Training:
bash ./tools/dist_train.sh configs/depression/exp2.py num-gpus --seed 0

AVEC 2014 Testing:
bash ./tools/dist_test.sh configs/depression/exp2.py checkpoint-dirs num-gpus


AVEC 2013 Training:
bash ./tools/dist_train.sh configs/depression/exp_avec2013.py num-gpus --seed 0

AVEC 2013 Testing:
bash ./tools/dist_test.sh configs/depression/exp_avec2013.py checkpoint-dirs num-gpus