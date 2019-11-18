#!/bin/bash

FAIL=0

echo "starting"
python rrun.py --pt 100 --network rnn31 --save asuzjrnn100ptonly --isz 0 --end 100000 --epochs 60 --gpu 2 --eta 0 --etabin 2.4 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python rrun.py --pt 200 --network rnn31 --save asuzjrnn200ptonly --isz 0 --end 100000 --epochs 60 --gpu 2 --eta 0 --etabin 2.4 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 &
python rrun.py --pt 500 --network rnn31 --save asuzjrnn500ptonly --isz 0 --end 100000 --epochs 60 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python rrun.py --pt 1000 --network rnn31 --save asuzjrnn1000ptonly --isz 0 --end 100000 --epochs 60 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 & 

python modelrun.py --pt 500 --save asuzjcnn500ptonly2 --isz 0 --end 100000 --epochs 20 --gpu 0 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python modelrun.py --pt 1000 --save asuzjcnn1000ptonly2 --isz 0 --end 100000 --epochs 20 --gpu 0 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 &
python modelrun.py --pt 100 --save asuzjcnn100ptonly2 --isz 0 --end 100000 --epochs 20 --gpu 1 --eta 0 --etabin 2.4 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python modelrun.py --pt 200 --save asuzjcnn200ptonly2 --isz 0 --end 100000 --epochs 20 --gpu 1 --eta 0 --etabin 2.4 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 & 

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
python getpred.py --save asuzjrnn200ptonly --pt 200 --end 100000 --isz 0 --gpu 2  &

python getpred.py --save asuzjrnn1000ptonly --pt 1000 --end 100000 --isz 0 --gpu 2 &  

python getpred.py --save asuzjrnn100ptonly --pt 100 --end 100000 --isz 0 --gpu 3 &
python getpred.py --save asuzjrnn500ptonly --pt 500 --end 100000 --isz 0 --gpu 3 &

python getpred.py --save asuzjcnn1000ptonly2 --pt 1000 --end 100000 --isz 0 --gpu 0 & 
python getpred.py --save asuzjcnn500ptonly2 --pt 500 --end 100000 --isz 0 --gpu 0 &
python getpred.py --save asuzjcnn200ptonly2 --pt 200 --end 100000 --isz 0 --gpu 1 &
python getpred.py --save asuzjcnn100ptonly2 --pt 100 --end 100000 --isz 0 --gpu 1 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

echo "starting"
python rrun.py --pt 100 --network rnn31 --save asuzjrnn100pteta --isz 0 --end 100000 --epochs 60 --gpu 2 --eta 0 --etabin 1 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python rrun.py --pt 200 --network rnn31 --save asuzjrnn200pteta --isz 0 --end 100000 --epochs 60 --gpu 2 --eta 0 --etabin 1 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 &
python rrun.py --pt 500 --network rnn31 --save asuzjrnn500pteta --isz 0 --end 100000 --epochs 60 --gpu 3 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python rrun.py --pt 1000 --network rnn31 --save asuzjrnn1000pteta --isz 0 --end 100000 --epochs 60 --gpu 3 --eta 0 --etabin 1 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 & 

python modelrun.py --pt 500 --save asuzjcnn500pteta --isz 0 --end 100000 --epochs 20 --gpu 0 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python modelrun.py --pt 1000 --save asuzjcnn1000pteta --isz 0 --end 100000 --epochs 20 --gpu 0 --eta 0 --etabin 1 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 &
python modelrun.py --pt 100 --save asuzjcnn100pteta --isz 0 --end 100000 --epochs 20 --gpu 1 --eta 0 --etabin 1 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python modelrun.py --pt 200 --save asuzjcnn200pteta --isz 0 --end 100000 --epochs 20 --gpu 1 --eta 0 --etabin 1 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 & 

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
python getpred.py --save asuzjrnn200pteta --pt 200 --end 100000 --isz 0 --gpu 2  &

python getpred.py --save asuzjrnn1000pteta --pt 1000 --end 100000 --isz 0 --gpu 2 &  

python getpred.py --save asuzjrnn100pteta --pt 100 --end 100000 --isz 0 --gpu 3 &
python getpred.py --save asuzjrnn500pteta --pt 500 --end 100000 --isz 0 --gpu 3 &

python getpred.py --save asuzjcnn1000pteta --pt 1000 --end 100000 --isz 0 --gpu 0 & 
python getpred.py --save asuzjcnn500pteta --pt 500 --end 100000 --isz 0 --gpu 0 &
python getpred.py --save asuzjcnn200pteta --pt 200 --end 100000 --isz 0 --gpu 1 &
python getpred.py --save asuzjcnn100pteta --pt 100 --end 100000 --isz 0 --gpu 1 

echo $FAIL
