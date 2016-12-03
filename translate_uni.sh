#!/usr/bin/env bash
dataset=dev
model="/work/jg5223/work/SimulTrans/.pretrained/model_wmt15_bpe2k_uni_en-ru.npz"
dict="/scratch/jg5223/data/wmt15/ruen/train/all_ru-en.en.tok.bpe.word.pkl"
dict_rev="/scratch/jg5223/data/wmt15/ruen/train/all_ru-en.ru.tok.bpe.word.pkl"
source="/scratch/jg5223/data/wmt15/ruen/${dataset}/newstest2013-src.en.tok.bpe"
saveto="./enrugreedy.out"
reference="/scratch/jg5223/data/wmt15/ruen/${dataset}/newstest2013-src.ru.tok"

# pyenv local anaconda-2.4.0
THEANO_FLAGS="floatX=float32, device=cpu" python translate_uni.py -p 8 -k 1 $model $dict $dict_rev $source $saveto

./data/multi-bleu.perl $reference < $saveto
