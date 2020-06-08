#!/bin/bash

FAIL=0

echo "starting"

#python sgdrun.py --pt 1000 --network rnn21 --save asuzjrnn1000ptonly2 --isz 0 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000
#python sgdrun.py --pt 1000 --network rnn21 --save asuzjrnn1000ptonly2 --isz 0 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000
#python sgdrun.py --pt 1000 --network rnn21 --save asuzjrnn1000ptonly2 --isz 0 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000
#python sgdrun.py --pt 1000 --network rnn21 --save asuzjrnn1000ptonly2 --isz 0 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000
python brun.py --pt 100 --network rnn31 --save asuzjrnn100ptonlyb --isz 0 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.815 --ptmax 1.159 & 
python brun.py --pt 1000 --network rnn31 --save asuzjrnn1000ptonlyb --isz 0 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python brun.py --pt 200 --network rnn31 --save asuzjrnn200ptonlyb --isz 0 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.819 --ptmax 1.123 & 
python brun.py --pt 500 --network rnn31 --save asuzjrnn500ptonlyb --isz 0 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 & 

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getpred.py --save asuzjrnn200ptonlyb --pt 200 --end 100000 --isz 0 --gpu 3 &
python getpred.py --save asuzjrnn1000ptonlyb --pt 1000 --end 100000 --isz 0 --gpu 3 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getpred.py --save asuzjrnn100ptonlyb --pt 100 --end 100000 --isz 0 --gpu 3 &
python getpred.py --save asuzjrnn500ptonlyb --pt 500 --end 100000 --isz 0 --gpu 3 

echo $FAIL
