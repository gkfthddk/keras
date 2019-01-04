
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

declare -a arr=("gru" "asym" "simplecnn" "rnncnn" "rnn" "lstm" "simple" "lstmcnn")
for i in "${arr[@]}"
  do
    python srun.py --network ${i} --pt 100 --save slow100${i} --end 0.5 --epochs 10 --gpu 1
    #python srun.py --network ${i} --pt 200 --save slow200${i} --end 0.5 --epochs 10 --gpu 1
    python srun.py --network ${i} --pt 500 --save slow500${i} --end 0.5 --epochs 10 --gpu 1
    #python srun.py --network ${i} --pt 1000 --save slow1000${i} --end 0.5 --epochs 10 --gpu 1
    done
