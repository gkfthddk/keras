#!/bin/bash

FAIL=0

echo "starting"
python saver.py --pt 500 --network rnn21 --save asuzjrnn500ptonly3 --isz 1 --end 100000 --epochs 100 --gpu 0 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python saver.py --pt 100 --network rnn21 --save asuzjrnn500ptonly3 --isz 1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 1 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python saver.py --pt 200 --network rnn21 --save asuzjrnn500ptonly3 --isz 1 --end 100000 --epochs 100 --gpu 2 --eta 0 --etabin 1 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 &
python saver.py --pt 1000 --network rnn21 --save asuzjrn500ptonly3 --isz 1 --end 100000 --epochs 100 --gpu 3 --eta 0 --etabin 1 --ptmin 0.823 --ptmax 1.076 --batch_size 100000

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python sgdrun.py --pt 100 --network rnn21 --save asuzqrnn100pteta21 --isz 1 --end 100000 --epochs 60 --gpu 5 --etabin 1 &
python sgdrun.py --pt 200 --network rnn21 --save asuzqrnn200pteta21 --isz 1 --end 100000 --epochs 60 --gpu 5 --etabin 1 &
python sgdrun.py --pt 500 --network rnn21 --save asuzqrnn500pteta21 --isz 1 --end 100000 --epochs 60 --gpu 4 --etabin 1 &
python sgdrun.py --pt 1000 --network rnn21 --save asuzqrnn1000pteta21 --isz 1 --end 100000 --epochs 60 --gpu 4 --etabin 1 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python npzpred.py --save asuzqrnn200pteta21 --pt 200 --end 100000 --isz 1 --gpu 0 &
python npzpred.py --save asuzqrnn1000pteta21 --pt 1000 --end 100000 --isz 1 --gpu 1 &
python npzpred.py --save asuzqrnn100pteta21 --pt 100 --end 100000 --isz 1 --gpu 2 &
python npzpred.py --save asuzqrnn500pteta21 --pt 500 --end 100000 --isz 1 --gpu 3 & 
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
python getauc.py --save asuzqrnn200pteta21 --pt 200 --get ptcut &
python getauc.py --save asuzqrnn1000pteta21 --pt 1000 --get ptcut &
python getauc.py --save asuzqrnn500pteta21 --pt 500 --get ptcut &
python getauc.py --save asuzqrnn100pteta21 --pt 100 --get ptcut

echo $FAIL
