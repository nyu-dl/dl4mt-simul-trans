THEANO_FLAGS=device=gpu$1 python simultrans_train.py --sample 5 --batchsize 10 --target 10 --sinit 1 --gamma 1 --recurrent True --Rtype 10 --coverage True | tee .log/$1.log 

