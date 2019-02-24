
#python rcrun.py --network simple2 --save qg200simple2 --end 1. 
#python pred.py --save qg200simple2
#python rcrun.py --network naasym2 --save qg200naasym2 --end 1. 
#python rcrun.py --network nagru2 --save qg200nagru2 --end 1. 
#python pred.py --save qg200naasym2
#python pred.py --save qg200nagru2


#python ensrun.py --save1 slowlstm --save2 slowgru --right sdata/dijet_200_220/dijet_200_220_training --valright sdata/dijet_200_220/dijet_200_220_validation --end 1. --epoch 1

#python srun.py --network lstm --right sdata/dijet_200_220/dijet_200_220_training --valright sdata/dijet_200_220/dijet_200_220_validation --save slow220lstm --end 1. --epochs 10
#python srun.py --network simplecnn --right sdata/dijet_200_220/dijet_200_220_training --valright sdata/dijet_200_220/dijet_200_220_validation --save slow220cnn --end 1. --epochs 10
#python srun.py --network gru --right sdata/dijet_200_220/dijet_200_220_training --valright sdata/dijet_200_220/dijet_200_220_validation --save slow220gru --end 1. --epochs 10
#python srun.py --network naasym --right sdata/dijet_200_220/dijet_200_220_training --valright sdata/dijet_200_220/dijet_200_220_validation --save slow220asym --end 1. --epochs 10
#python srun.py --network simple --right  sdata/dijet_200_220/dijet_200_220_training --valright sdata/dijet_200_220/dijet_200_220_validation --save slow220simple --end 1. --epochs 10
#python srun.py --network asym --right  sdata/dijet_200_220/dijet_200_220_training --valright sdata/dijet_200_220/dijet_200_220_validation --save slow220asym --end 1. --epochs 10
#python srun.py --network lstmasym --right  sdata/dijet_200_220/dijet_200_220_training --valright sdata/dijet_200_220/dijet_200_220_validation --save slow220lstmasym --end 1. --epochs 10

#declare -a arr=("gru" "asym" "simplecnn" "rnncnn" "rnn" "lstm" "simple" "lstmcnn")
#for i in "${arr[@]}"
#do
#    python srun.py --network ${i} --pt 100 --save slow100${i} --end 0.5 --epochs 10 --gpu 1
    #python srun.py --network ${i} --pt 200 --save slow200${i} --end 0.5 --epochs 10 --gpu 1
#    python srun.py --network ${i} --pt 500 --save slow500${i} --end 0.5 --epochs 10 --gpu 1
    #python srun.py --network ${i} --pt 1000 --save slow1000${i} --end 0.5 --epochs 10 --gpu 1
#    done
#declare -a arr=("gru" "simplecnn" "rnncnn" "gru2" "simplecnn2")
#declare -a arr=("cnn")
#declare -a pt=("500" "1000")
#for j in "${pt[@]}"
#do
#for i in "${arr[@]}"
#do
#python srun.py --network ${i} --pt ${j} --save ten${j}${i}4 --end 1. --epochs 10 --gpu 1
#python tpred.py --save ten${j}${i}4 --pt ${j}
#done
#done
#for j in "${pt[@]}"
#do
#python epred.py --net1 gru3 --net2 lstm4 --pt ${j}
#python epred.py --net1 simplecnn2 --net2 cnn4 --pt ${j}
#python epred.py --net1 gru --net2 lstm4 --pt ${j}
#python epred.py --net1 simplecnn3 --net2 cnn4 --pt ${j}
#done

#python run.py --network rnn2 --pt 200 --save testrnn2 --end 1 --epochs 100 --gpu 0
#python runadam.py --network rnn2 --pt 200 --save testadamrnn2 --end 1 --epochs 100 --gpu 0

#20190219
#python testrun.py --network cnn --pt 100 --save testcnn128 --end 0.3 --epochs 20 --batch_size 128 --gpu 1
#python testrun.py --network cnn --pt 100 --save testcnn256 --end 0.3 --epochs 20 --batch_size 256 --gpu 1
#python testrun.py --network cnn --pt 100 --save testcnn512 --end 0.3 --epochs 20 --batch_size 512 --gpu 1
#python testrun.py --network cnn --pt 100 --save testcnn1024 --end 0.3 --epochs 20 --batch_size 1024 --gpu 1

#python testrun.py --network cnn --pt 100 --save testcnn --end 0.3 --epochs 20 --gpu 1

#20190220
python zrun.py --network rnn --pt 100 --save paperzqrnn1002 --isz 1 --end 100000 --epochs 100 --gpu 1
declare -a pt=("200")
for j in "${pt[@]}"
do
python zrun.py --network rnn --pt ${j} --save paperzjrnn${j} --end 100000 --epochs 100 --gpu 1
python zrun.py --network rnn --pt ${j} --save paperqqrnn${j} --isz -1 --end 100000 --epochs 100 --gpu 1
python zrun.py --network rnn --pt ${j} --save paperzqrnn${j}2 --isz 1 --end 100000 --epochs 100 --gpu 1
done
