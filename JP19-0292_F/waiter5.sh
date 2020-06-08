#!/bin/bash

FAIL=0

echo "starting"

python modelrun.py --pt 500 --save asuzqcnn500pteta --isz 1 --end 100000 --epochs 20 --gpu 4 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python modelrun.py --pt 1000 --save asuzqcnn1000pteta --isz 1 --end 100000 --epochs 20 --gpu 4 --eta 0 --etabin 1 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 &
python modelrun.py --pt 100 --save asuzqcnn100pteta --isz 1 --end 100000 --epochs 20 --gpu 5 --eta 0 --etabin 1 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python modelrun.py --pt 200 --save asuzqcnn200pteta --isz 1 --end 100000 --epochs 20 --gpu 5 --eta 0 --etabin 1 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 & 
python modelrun.py --pt 500 --save asuqqcnn500pteta --isz -1 --end 100000 --epochs 20 --gpu 6 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 &
python modelrun.py --pt 1000 --save asuqqcnn1000pteta --isz -1 --end 100000 --epochs 20 --gpu 6 --eta 0 --etabin 1 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 &
python modelrun.py --pt 100 --save asuqqcnn100pteta --isz -1 --end 100000 --epochs 20 --gpu 7 --eta 0 --etabin 1 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 &
python modelrun.py --pt 200 --save asuqqcnn200pteta --isz -1 --end 100000 --epochs 20 --gpu 7 --eta 0 --etabin 1 --ptmin 0.819 --ptmax 1.123 --batch_size 100000  

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getpred.py --save asuzqcnn1000pteta --pt 1000 --end 100000 --isz 1 --gpu 4 & 
python getpred.py --save asuzqcnn500pteta --pt 500 --end 100000 --isz 1 --gpu 4 &
python getpred.py --save asuzqcnn200pteta --pt 200 --end 100000 --isz 1 --gpu 5 &
python getpred.py --save asuzqcnn100pteta --pt 100 --end 100000 --isz 1 --gpu 5 &
python getpred.py --save asuqqcnn1000pteta --pt 1000 --end 100000 --isz -1 --gpu 6 & 
python getpred.py --save asuqqcnn500pteta --pt 500 --end 100000 --isz -1 --gpu 6 &
python getpred.py --save asuqqcnn200pteta --pt 200 --end 100000 --isz -1 --gpu 7 &
python getpred.py --save asuqqcnn100pteta --pt 100 --end 100000 --isz -1 --gpu 7 

