#!/bin/bash

FAIL=0

echo "starting"

python sgdrun.py --pt 100 --network rnn31 --save asuzjrnn100ptonly31 --isz 0 --end 100000 --epochs 60 --gpu 7 --etabin 2.4 &
python sgdrun.py --pt 200 --network rnn31 --save asuzjrnn200ptonly31 --isz 0 --end 100000 --epochs 60 --gpu 7 --etabin 2.4 &
python sgdrun.py --pt 500 --network rnn31 --save asuzjrnn500ptonly31 --isz 0 --end 100000 --epochs 60 --gpu 6 --etabin 2.4 &
python sgdrun.py --pt 1000 --network rnn31 --save asuzjrnn1000ptonly31 --isz 0 --end 100000 --epochs 60 --gpu 6 --etabin 2.4 &

python sgdrun.py --pt 100 --network rnn21 --save asuzjrnn100ptonly21 --isz 0 --end 100000 --epochs 60 --gpu 5 --etabin 2.4 &
python sgdrun.py --pt 200 --network rnn21 --save asuzjrnn200ptonly21 --isz 0 --end 100000 --epochs 60 --gpu 5 --etabin 2.4 &
python sgdrun.py --pt 500 --network rnn21 --save asuzjrnn500ptonly21 --isz 0 --end 100000 --epochs 60 --gpu 4 --etabin 2.4 &
python sgdrun.py --pt 1000 --network rnn21 --save asuzjrnn1000ptonly21 --isz 0 --end 100000 --epochs 60 --gpu 4 --etabin 2.4 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python npzpred.py --save asuzjrnn200ptonly21 --pt 200 --end 100000 --isz 0 --gpu 5 &
python npzpred.py --save asuzjrnn1000ptonly21 --pt 1000 --end 100000 --isz 0 --gpu 5 &
python npzpred.py --save asuzjrnn100ptonly21 --pt 100 --end 100000 --isz 0 --gpu 4 &
python npzpred.py --save asuzjrnn500ptonly21 --pt 500 --end 100000 --isz 0 --gpu 4 & 
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
python npzpred.py --save asuzjrnn200ptonly31 --pt 200 --end 100000 --isz 0 --gpu 7 &
python npzpred.py --save asuzjrnn1000ptonly31 --pt 1000 --end 100000 --isz 0 --gpu 7 &
python npzpred.py --save asuzjrnn100ptonly31 --pt 100 --end 100000 --isz 0 --gpu 6 &
python npzpred.py --save asuzjrnn500ptonly31 --pt 500 --end 100000 --isz 0 --gpu 6 

echo $FAIL
