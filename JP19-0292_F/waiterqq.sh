#!/bin/bash

FAIL=0

echo "starting"
python saver.py --pt 500 --network rnn21 --save asuzjrnn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 0 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python saver.py --pt 100 --network rnn21 --save asuzjrnn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 1 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python saver.py --pt 200 --network rnn21 --save asuzjrnn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 2 --eta 0 --etabin 1 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 &
python saver.py --pt 1000 --network rnn21 --save asuzjrn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 1 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 &
python saver.py --pt 500 --network rnn21 --save asuzjrnn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 0 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python saver.py --pt 100 --network rnn21 --save asuzjrnn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 2.4 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python saver.py --pt 200 --network rnn21 --save asuzjrnn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 2 --eta 0 --etabin 2.4 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 &
python saver.py --pt 1000 --network rnn21 --save asuzjrn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python sgdrun.py --pt 100 --network rnn21 --save asuqqrnn100pteta21 --isz -1 --end 100000 --epochs 60 --gpu 6 --etabin 1 &
python sgdrun.py --pt 200 --network rnn21 --save asuqqrnn200pteta21 --isz -1 --end 100000 --epochs 60 --gpu 6 --etabin 1 &
python sgdrun.py --pt 500 --network rnn21 --save asuqqrnn500pteta21 --isz -1 --end 100000 --epochs 60 --gpu 7 --etabin 1 &
python sgdrun.py --pt 1000 --network rnn21 --save asuqqrnn1000pteta21 --isz -1 --end 100000 --epochs 60 --gpu 7 --etabin 1 &

python sgdrun.py --pt 100 --network rnn21 --save asuqqrnn100ptonly21 --isz -1 --end 100000 --epochs 60 --gpu 0 --etabin 2.4 &
python sgdrun.py --pt 200 --network rnn21 --save asuqqrnn200ptonly21 --isz -1 --end 100000 --epochs 60 --gpu 1 --etabin 2.4 &
python sgdrun.py --pt 500 --network rnn21 --save asuqqrnn500ptonly21 --isz -1 --end 100000 --epochs 60 --gpu 2 --etabin 2.4 &
python sgdrun.py --pt 1000 --network rnn21 --save asuqqrnn1000ptonly21 --isz -1 --end 100000 --epochs 60 --gpu 3 --etabin 2.4 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python npzpred.py --save asuqqrnn200pteta21 --pt 200 --end 100000 --isz -1 --gpu 6 &
python npzpred.py --save asuqqrnn1000pteta21 --pt 1000 --end 100000 --isz -1 --gpu 6 &
python npzpred.py --save asuqqrnn100pteta21 --pt 100 --end 100000 --isz -1 --gpu 7 &
python npzpred.py --save asuqqrnn500pteta21 --pt 500 --end 100000 --isz -1 --gpu 7 & 

python npzpred.py --save asuqqrnn200ptonly21 --pt 200 --end 100000 --isz -1 --gpu 6 &
python npzpred.py --save asuqqrnn1000ptonly21 --pt 1000 --end 100000 --isz -1 --gpu 6 &
python npzpred.py --save asuqqrnn100ptonly21 --pt 100 --end 100000 --isz -1 --gpu 7 &
python npzpred.py --save asuqqrnn500ptonly21 --pt 500 --end 100000 --isz -1 --gpu 7 & 
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
echo $FAIL
python getauc.py --save asuqqrnn200pteta21 --pt 200 --get ptcut &
python getauc.py --save asuqqrnn1000pteta21 --pt 1000 --get ptcut &
python getauc.py --save asuqqrnn500pteta21 --pt 500 --get ptcut &
python getauc.py --save asuqqrnn100pteta21 --pt 100 --get ptcut
python getauc.py --save asuqqrnn200ptonly21 --pt 200 --get ptcut &
python getauc.py --save asuqqrnn1000ptonly21 --pt 1000 --get ptcut &
python getauc.py --save asuqqrnn500ptonly21 --pt 500 --get ptcut &
python getauc.py --save asuqqrnn100ptonly21 --pt 100 --get ptcut

