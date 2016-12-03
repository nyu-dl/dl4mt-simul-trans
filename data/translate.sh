#!/usr/bin/env bash

model=".pretrained/model_wmt15_bpe2k_uni_en-ru.npz"
dict="/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/train/all_ru-en.en.tok.bpe.word.pkl"
dict_rev="/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/train/all_ru-en.ru.tok.bpe.word.pkl"
source="/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/dev/newstest2013-src.en.tok.bpe"
saveto=".translate/standard.trans"

python translate_uni.py $model $dict $dict_rev $source $saveto