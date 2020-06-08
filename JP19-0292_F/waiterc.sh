#!/bin/bash

FAIL=0

echo "starting"

python modelrun.py --pt 500 --save asuzjcnn500scalept --isz 0 --end 100000 --epochs 20 --gpu 3 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 --unscale 0&
python modelrun.py --pt 1000 --save asuzjcnn1000scalept --isz 0 --end 100000 --epochs 20 --gpu 4 --eta 0 --etabin 2.4 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 --unscale 0 &
python modelrun.py --pt 100 --save asuzjcnn100scalept --isz 0 --end 100000 --epochs 20 --gpu 5 --eta 0 --etabin 2.4 --ptmin 0.815 --ptmax 1.159 --batch_size 100000 --unscale 0 &
python modelrun.py --pt 200 --save asuzjcnn200scalept --isz 0 --end 100000 --epochs 20 --gpu 6 --eta 0 --etabin 2.4 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 --unscale 0 & 

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getpred.py --save asuzjcnn1000scalept --pt 1000 --end 100000 --isz 0 --gpu 3 --unscale 0 & 
python getpred.py --save asuzjcnn500scalept --pt 500 --end 100000 --isz 0 --gpu 4 --unscale 0 &
python getpred.py --save asuzjcnn200scalept --pt 200 --end 100000 --isz 0 --gpu 5 --unscale 0 &
python getpred.py --save asuzjcnn100scalept --pt 100 --end 100000 --isz 0 --gpu 6 --unscale 0 &

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

echo "starting"
python getauc.py --save asuzjcnn200scalept --pt 200 --get ptcut9
python getauc.py --save asuzjcnn1000scalept --pt 1000 --get ptcut9
python getauc.py --save asuzjcnn500scalept --pt 500 --get ptcut9
python getauc.py --save asuzjcnn100scalept --pt 100 --get ptcut9

python modelrun.py --pt 500 --save asuzjcnn500scalepteta --isz 0 --end 100000 --epochs 20 --gpu 3 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000 --unscale 0 &
python modelrun.py --pt 1000 --save asuzjcnn1000scalepteta --isz 0 --end 100000 --epochs 20 --gpu 4 --eta 0 --etabin 1 --ptmin 0.8235 --ptmax 1.076 --batch_size 100000 --unscale 0 &
python modelrun.py --pt 100 --save asuzjcnn100scalepteta --isz 0 --end 100000 --epochs 20 --gpu 5 --eta 0 --etabin 1 --ptmin 0.815 --ptmax 1.159 --batch_size 100000  --unscale 0&
python modelrun.py --pt 200 --save asuzjcnn200scalepteta --isz 0 --end 100000 --epochs 20 --gpu 6 --eta 0 --etabin 1 --ptmin 0.819 --ptmax 1.123 --batch_size 100000 --unscale 0 & 

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getpred.py --save asuzjcnn1000scalepteta --pt 1000 --end 100000 --isz 0 --gpu 3 --unscale 0 & 
python getpred.py --save asuzjcnn500scalepteta --pt 500 --end 100000 --isz 0 --gpu 4 --unscale 0 &
python getpred.py --save asuzjcnn200scalepteta --pt 200 --end 100000 --isz 0 --gpu 5 --unscale 0 &
python getpred.py --save asuzjcnn100scalepteta --pt 100 --end 100000 --isz 0 --gpu 6 --unscale 0 
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getauc.py --save asuzjcnn200scalepteta --pt 200 --get ptcut9
python getauc.py --save asuzjcnn1000scalepteta --pt 1000 --get ptcut9
python getauc.py --save asuzjcnn500scalepteta --pt 500 --get ptcut9
python getauc.py --save asuzjcnn100scalepteta --pt 100 --get ptcut9
echo $FAIL
