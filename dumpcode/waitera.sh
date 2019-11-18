#!/bin/bash

FAIL=0

echo "starting"

python adamcnnrun.py --pt 500 --save asuzjcnn500ptonlyadam --isz 0 --end 100000 --epochs 20 --gpu 0 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python adamcnnrun.py --pt 1000 --save asuzjcnn1000ptonlyadam --isz 0 --end 100000 --epochs 20 --gpu 0 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 &
python adamcnnrun.py --pt 100 --save asuzjcnn100ptonlyadam --isz 0 --end 100000 --epochs 20 --gpu 1 --eta 0 --etabin 2.4 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python adamcnnrun.py --pt 200 --save asuzjcnn200ptonlyadam --isz 0 --end 100000 --epochs 20 --gpu 1 --eta 0 --etabin 2.4 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 & 

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getpred.py --save asuzjcnn1000ptonlyadam --pt 1000 --end 100000 --isz 0 --gpu 0 & 
python getpred.py --save asuzjcnn500ptonlyadam --pt 500 --end 100000 --isz 0 --gpu 0 &
python getpred.py --save asuzjcnn200ptonlyadam --pt 200 --end 100000 --isz 0 --gpu 1 &
python getpred.py --save asuzjcnn100ptonlyadam --pt 100 --end 100000 --isz 0 --gpu 1 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

echo "starting"

python adamcnnrun.py --pt 500 --save asuzjcnn500ptetaadam --isz 0 --end 100000 --epochs 20 --gpu 0 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python adamcnnrun.py --pt 1000 --save asuzjcnn1000ptetaadam --isz 0 --end 100000 --epochs 20 --gpu 0 --eta 0 --etabin 1 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 &
python adamcnnrun.py --pt 100 --save asuzjcnn100ptetaadam --isz 0 --end 100000 --epochs 20 --gpu 1 --eta 0 --etabin 1 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python adamcnnrun.py --pt 200 --save asuzjcnn200ptetaadam --isz 0 --end 100000 --epochs 20 --gpu 1 --eta 0 --etabin 1 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 & 

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getpred.py --save asuzjcnn1000ptetaadam --pt 1000 --end 100000 --isz 0 --gpu 0 & 
python getpred.py --save asuzjcnn500ptetaadam --pt 500 --end 100000 --isz 0 --gpu 0 &
python getpred.py --save asuzjcnn200ptetaadam --pt 200 --end 100000 --isz 0 --gpu 1 &
python getpred.py --save asuzjcnn100ptetaadam --pt 100 --end 100000 --isz 0 --gpu 1 

echo $FAIL
