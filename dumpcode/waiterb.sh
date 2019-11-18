#python boost.py --pt 100 --gpu 0 --end 100000 --save asuzqbdt100pteta --isz 1 --ptmin 0.815 --ptmax 1.159 --etabin 1 &
#python boost.py --pt 200 --gpu 1 --end 100000 --save asuzqbdt200pteta --isz 1 --ptmin 0.819 --ptmax 1.123 --etabin 1 &
#python boost.py --pt 500 --gpu 2 --end 100000 --save asuzqbdt500pteta --isz 1 --ptmin 0.821 --ptmax 1.093 --etabin 1 &
#python boost.py --pt 1000 --gpu 3 --end 100000 --save asuzqbdt1000pteta --isz 1 --ptmin 0.8235 --ptmax 1.076 --etabin 1 &
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python xpred.py --pt 100 --gpu 0 --end 100000 --save asuzqbdt100pteta --isz 1 --ptmin 0.815 --ptmax 1.159  --etabin 1 &
python xpred.py --pt 200 --gpu 1 --end 100000 --save asuzqbdt200pteta --isz 1 --ptmin 0.819 --ptmax 1.123  --etabin 1 &
python xpred.py --pt 500 --gpu 2 --end 100000 --save asuzqbdt500pteta  --isz 1 --ptmin 0.821 --ptmax 1.093  --etabin 1 &
python xpred.py --pt 1000 --gpu 3 --end 100000 --save asuzqbdt1000pteta --isz 1 --ptmin 0.8235 --ptmax 1.076  --etabin 1


for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getauc.py --save asuzqbdt200pteta --pt 200 --get ptcut &
python getauc.py --save asuzqbdt1000pteta --pt 1000 --get ptcut &
python getauc.py --save asuzqbdt500pteta --pt 500 --get ptcut &
python getauc.py --save asuzqbdt100pteta --pt 100 --get ptcut  

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

#python boost.py --pt 100 --gpu 0 --end 100000 --save asuzqbdt100pt --isz 1 --ptmin 0.815 --ptmax 1.159 --etabin 2.4 &
#python boost.py --pt 200 --gpu 1 --end 100000 --save asuzqbdt200pt --isz 1 --ptmin 0.819 --ptmax 1.123 --etabin 2.4 
#python boost.py --pt 500 --gpu 2 --end 100000 --save asuzqbdt500pt --isz 1 --ptmin 0.821 --ptmax 1.093 --etabin 2.4 &
#python boost.py --pt 1000 --gpu 3 --end 100000 --save asuzqbdt1000pt --isz 1 --ptmin 0.8235 --ptmax 1.076 --etabin 2.4
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python xpred.py --pt 100 --gpu 0 --end 100000 --save asuzqbdt100pt --isz 1 --ptmin 0.815 --ptmax 1.159  --etabin 2.4 &
python xpred.py --pt 200 --gpu 1 --end 100000 --save asuzqbdt200pt --isz 1 --ptmin 0.819 --ptmax 1.123  --etabin 2.4 &
python xpred.py --pt 500 --gpu 2 --end 100000 --save asuzqbdt500pt  --isz 1 --ptmin 0.821 --ptmax 1.093  --etabin 2.4 &
python xpred.py --pt 1000 --gpu 3 --end 100000 --save asuzqbdt1000pt --isz 1 --ptmin 0.8235 --ptmax 1.076  --etabin 2.4


for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getauc.py --save asuzqbdt200pt --pt 200 --get ptcut &
python getauc.py --save asuzqbdt1000pt --pt 1000 --get ptcut &
python getauc.py --save asuzqbdt500pt --pt 500 --get ptcut &
python getauc.py --save asuzqbdt100pt --pt 100 --get ptcut  
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
#######################################
#python boost.py --pt 100 --gpu 0 --end 100000 --save asuqqbdt100pteta --isz -1 --ptmin 0.815 --ptmax 1.159 --etabin 1 &
#python boost.py --pt 200 --gpu 1 --end 100000 --save asuqqbdt200pteta --isz -1 --ptmin 0.819 --ptmax 1.123 --etabin 1 
#python boost.py --pt 500 --gpu 2 --end 100000 --save asuqqbdt500pteta --isz -1 --ptmin 0.821 --ptmax 1.093 --etabin 1 &
#python boost.py --pt 1000 --gpu 3 --end 100000 --save asuqqbdt1000pteta --isz -1 --ptmin 0.8235 --ptmax 1.076 --etabin 1
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python xpred.py --pt 100 --gpu 0 --end 100000 --save asuqqbdt100pteta --isz -1 --ptmin 0.815 --ptmax 1.159  --etabin 1 &
python xpred.py --pt 200 --gpu 1 --end 100000 --save asuqqbdt200pteta --isz -1 --ptmin 0.819 --ptmax 1.123  --etabin 1 &
python xpred.py --pt 500 --gpu 2 --end 100000 --save asuqqbdt500pteta  --isz -1 --ptmin 0.821 --ptmax 1.093  --etabin 1 &
python xpred.py --pt 1000 --gpu 3 --end 100000 --save asuqqbdt1000pteta --isz -1 --ptmin 0.8235 --ptmax 1.076  --etabin 1


for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getauc.py --save asuqqbdt200pteta --pt 200 --get ptcut &
python getauc.py --save asuqqbdt1000pteta --pt 1000 --get ptcut &
python getauc.py --save asuqqbdt500pteta --pt 500 --get ptcut &
python getauc.py --save asuqqbdt100pteta --pt 100 --get ptcut  

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

#python boost.py --pt 100 --gpu 0 --end 100000 --save asuqqbdt100pt --isz -1 --ptmin 0.815 --ptmax 1.159 --etabin 2.4 &
#python boost.py --pt 200 --gpu 1 --end 100000 --save asuqqbdt200pt --isz -1 --ptmin 0.819 --ptmax 1.123 --etabin 2.4 
#python boost.py --pt 500 --gpu 2 --end 100000 --save asuqqbdt500pt --isz -1 --ptmin 0.821 --ptmax 1.093 --etabin 2.4 &
#python boost.py --pt 1000 --gpu 3 --end 100000 --save asuqqbdt1000pt --isz -1 --ptmin 0.8235 --ptmax 1.076 --etabin 2.4
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python xpred.py --pt 100 --gpu 0 --end 100000 --save asuqqbdt100pt --isz -1 --ptmin 0.815 --ptmax 1.159  --etabin 2.4 &
python xpred.py --pt 200 --gpu 1 --end 100000 --save asuqqbdt200pt --isz -1 --ptmin 0.819 --ptmax 1.123  --etabin 2.4 &
python xpred.py --pt 500 --gpu 2 --end 100000 --save asuqqbdt500pt  --isz -1 --ptmin 0.821 --ptmax 1.093  --etabin 2.4 &
python xpred.py --pt 1000 --gpu 3 --end 100000 --save asuqqbdt1000pt --isz -1 --ptmin 0.8235 --ptmax 1.076  --etabin 2.4


for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

python getauc.py --save asuqqbdt200pt --pt 200 --get ptcut &
python getauc.py --save asuqqbdt1000pt --pt 1000 --get ptcut &
python getauc.py --save asuqqbdt500pt --pt 500 --get ptcut &
python getauc.py --save asuqqbdt100pt --pt 100 --get ptcut  
