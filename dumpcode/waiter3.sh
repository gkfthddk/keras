#!/bin/bash

FAIL=0

echo "starting"

python rmsrun.py --pt 100 --network rnn31 --save asuzjrnn100ptonly31rms --isz 0 --end 100000 --epochs 100 --gpu 2 &
python rmsrun.py --pt 200 --network rnn31 --save asuzjrnn200ptonly31rms --isz 0 --end 100000 --epochs 100 --gpu 2 &
python rmsrun.py --pt 500 --network rnn31 --save asuzjrnn500ptonly31rms --isz 0 --end 100000 --epochs 100 --gpu 3 &
python rmsrun.py --pt 1000 --network rnn31 --save asuzjrnn1000ptonly31rms --isz 0 --end 100000 --epochs 100 --gpu 3 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
python sgdrun.py --pt 100 --network rnn31 --save asuzjrnn100ptonly331 --isz 0 --end 100000 --epochs 100 --gpu 2 --channel 32 &
python sgdrun.py --pt 200 --network rnn31 --save asuzjrnn200ptonly331 --isz 0 --end 100000 --epochs 100 --gpu 2 --channel 32 &
python sgdrun.py --pt 500 --network rnn31 --save asuzjrnn500ptonly331 --isz 0 --end 100000 --epochs 100 --gpu 3 --channel 32 &
python sgdrun.py --pt 1000 --network rnn31 --save asuzjrnn1000ptonly331 --isz 0 --end 100000 --epochs 100 --gpu 3 --channel 32&
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python npzpred.py --save asuzjrnn200ptonly31rms --pt 200 --end 100000 --isz 0 --gpu 0 &
python npzpred.py --save asuzjrnn1000ptonly31rms --pt 1000 --end 100000 --isz 0 --gpu 0 &
python npzpred.py --save asuzjrnn100ptonly31rms --pt 100 --end 100000 --isz 0 --gpu 1 &
python npzpred.py --save asuzjrnn500ptonly31rms --pt 500 --end 100000 --isz 0 --gpu 1 
python npzpred.py --save asuzjrnn200ptonly331 --pt 200 --end 100000 --isz 0 --gpu 2 &
python npzpred.py --save asuzjrnn1000ptonly331 --pt 1000 --end 100000 --isz 0 --gpu 2 &
python npzpred.py --save asuzjrnn100ptonly331 --pt 100 --end 100000 --isz 0 --gpu 3 &
python npzpred.py --save asuzjrnn500ptonly331 --pt 500 --end 100000 --isz 0 --gpu 3 

echo $FAIL
