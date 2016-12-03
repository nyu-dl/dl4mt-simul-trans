#!/bin/bash

# ref=" /misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/ruen/dev/newstest2013-ref.ru.tok"
# sed -i 's/@@ //g' $1

DIR="/work/jg5223/work/SimulTrans/.translate/"

./data/multi-bleu.perl $DIR/ref.txt < $DIR/test.txt
