#!/bin/bash

FAIL=0

echo "starting"

python rrun.py --pt 100 --network rnn21 --save asuzjrnn100ptonly2 --isz 0 --end 100000 --epochs 60 --gpu 2 --eta 0 --etabin 2.4 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python rrun.py --pt 200 --network rnn21 --save asuzjrnn200ptonly2 --isz 0 --end 100000 --epochs 60 --gpu 2 --eta 0 --etabin 2.4 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 &
python rrun.py --pt 500 --network rnn21 --save asuzjrnn500ptonly2 --isz 0 --end 100000 --epochs 60 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python rrun.py --pt 1000 --network rnn21 --save asuzjrnn1000ptonly2 --isz 0 --end 100000 --epochs 60 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000  


for job in `jobs -p`
do
echo $job
    wait $job | let "FAIL+=1"
done

python getpred.py --save asuzjrnn200ptonly2 --pt 200 --end 100000 --isz 0 --gpu 2  &

python getpred.py --save asuzjrnn1000ptonly2 --pt 1000 --end 100000 --isz 0 --gpu 2 &  

python getpred.py --save asuzjrnn100ptonly2 --pt 100 --end 100000 --isz 0 --gpu 3 &
python getpred.py --save asuzjrnn500ptonly2 --pt 500 --end 100000 --isz 0 --gpu 3 


for job in `jobs -p`
do
echo $job
    wait $job | let "FAIL+=1"
done

python rrun.py --pt 100 --network rnn21 --save asuzjrnn100pteta2 --isz 0 --end 100000 --epochs 60 --gpu 2 --eta 0 --etabin 1 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python rrun.py --pt 200 --network rnn21 --save asuzjrnn200pteta2 --isz 0 --end 100000 --epochs 60 --gpu 2 --eta 0 --etabin 1 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 &
python rrun.py --pt 500 --network rnn21 --save asuzjrnn500pteta2 --isz 0 --end 100000 --epochs 60 --gpu 3 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python rrun.py --pt 1000 --network rnn21 --save asuzjrnn1000pteta2 --isz 0 --end 100000 --epochs 60 --gpu 3 --eta 0 --etabin 1 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000  

for job in `jobs -p`
do
echo $job
    wait $job | let "FAIL+=1"
done
python getpred.py --save asuzjrnn200pteta2 --pt 200 --end 100000 --isz 0 --gpu 2  &

python getpred.py --save asuzjrnn1000pteta2 --pt 1000 --end 100000 --isz 0 --gpu 2 &  

python getpred.py --save asuzjrnn100pteta2 --pt 100 --end 100000 --isz 0 --gpu 3 &
python getpred.py --save asuzjrnn500pteta2 --pt 500 --end 100000 --isz 0 --gpu 3 

echo $FAIL
