#!/bin/bash

FAIL=0

echo "starting"

python adamrrun.py --pt 100 --network rnn21 --save asuzjrnn100ptonly21adam --isz 0 --end 100000 --epochs 60 --gpu 4 --eta 0 --etabin 2.4 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python adamrrun.py --pt 200 --network rnn21 --save asuzjrnn200ptonly21adam --isz 0 --end 100000 --epochs 60 --gpu 4 --eta 0 --etabin 2.4 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 &
python adamrrun.py --pt 500 --network rnn21 --save asuzjrnn500ptonly21adam --isz 0 --end 100000 --epochs 60 --gpu 5 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python adamrrun.py --pt 1000 --network rnn21 --save asuzjrnn1000ptonly21adam --isz 0 --end 100000 --epochs 60 --gpu 5 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000  


for job in `jobs -p`
do
echo $job
    wait $job | let "FAIL+=1"
done

python getpred.py --save asuzjrnn200ptonly21adam --pt 200 --end 100000 --isz 0 --gpu 4  &

python getpred.py --save asuzjrnn1000ptonly21adam --pt 1000 --end 100000 --isz 0 --gpu 4 &  

python getpred.py --save asuzjrnn100ptonly21adam --pt 100 --end 100000 --isz 0 --gpu 5 &
python getpred.py --save asuzjrnn500ptonly21adam --pt 500 --end 100000 --isz 0 --gpu 5 


for job in `jobs -p`
do
echo $job
    wait $job | let "FAIL+=1"
done

python adamrrun.py --pt 100 --network rnn21 --save asuzjrnn100pteta21adam --isz 0 --end 100000 --epochs 60 --gpu 4 --eta 0 --etabin 1 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python adamrrun.py --pt 200 --network rnn21 --save asuzjrnn200pteta21adam --isz 0 --end 100000 --epochs 60 --gpu 4 --eta 0 --etabin 1 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 &
python adamrrun.py --pt 500 --network rnn21 --save asuzjrnn500pteta21adam --isz 0 --end 100000 --epochs 60 --gpu 5 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python adamrrun.py --pt 1000 --network rnn21 --save asuzjrnn1000pteta21adam --isz 0 --end 100000 --epochs 60 --gpu 5 --eta 0 --etabin 1 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000  

for job in `jobs -p`
do
echo $job
    wait $job | let "FAIL+=1"
done
python getpred.py --save asuzjrnn200pteta21adam --pt 200 --end 100000 --isz 0 --gpu 5  &

python getpred.py --save asuzjrnn1000pteta21adam --pt 1000 --end 100000 --isz 0 --gpu 5 &  

python getpred.py --save asuzjrnn100pteta21adam --pt 100 --end 100000 --isz 0 --gpu 4 &
python getpred.py --save asuzjrnn500pteta21adam --pt 500 --end 100000 --isz 0 --gpu 4 

echo $FAIL
