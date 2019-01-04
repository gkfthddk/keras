#20181017
#python rcrun.py --network rnncnn --save qg100rnncnn --end 0.1
#python rcrun.py --network lstm --save qg100lstm --end 0.1
#python rcrun.py --network asym --save qg100asym --end 0.1
#python rcrun.py --network gru --save qg100gru --end 0.1
#python pred.py --save qg100asym0.1
#python pred.py --save qg100lstmasym0.1
#python pred.py --save qg100gruasym0.1
#python pred.py --save qg100gru0.1
#python pred.py --save qg100lstm0.1
#python rcrun.py --network lstmasym --save qg100lstmasym --end 1.
#python rcrun.py --network gruasym --save qg100gruasym --end 1.
#python rcrun.py --network asym --save qg100asym --end 1.
#python rcrun.py --network gru --save qg100gru --end 1.
#20181018
#python rcrun.py --network lstm --save qg100lstm --end 1.
#python rcrun.py --network rnn --save qg100rnn --end 1.
#python rcrun.py --network vgg --save qg100vgg --end 1.
#python pred.py --save qg100asym
#python pred.py --save qg100vgg
#python pred.py --save qg100rnn
#python pred.py --save qg100lstmasym
#python pred.py --save qg100gruasym
#python pred.py --save qg100gru
#python pred.py --save qg100lstm
#python pred.py --save qg100rnnasym
#20181022
#python jrun.py --network gruasym --save zj100gruasym --end 1. 
#python jrun.py --network asym --save zj100asym --end 1. 
#python jrun.py --network rnn --save zj100rnn --end 1. 
#python jpred.py --save zj100asym
#python jpred.py --save zj100rnn
#20181023
#python rcrun.py --network simple --save qg100simple --end 1. 
#python jrun.py --network simple --save zj100simple --end 1. 

#python rcrun.py --network naasym --save qg100naasym --end 1. 
#python rcrun.py --network nagru --save qg100nagru --end 1. 
#python pred.py --save qg100naasym
#python pred.py --save qg100nagru

#20181109
#python rcrun.py --network simplecnn --save goocnn1 --end .3 --epochs 5
#python rcrun.py --network simplernn --save goornn1 --end .3 --epochs 5
#python rcrun.py --network asym --save gooasym1 --end .3 --epochs 5
#python rcrun.py --network gru --save googru1 --end .3 --epochs 5
#python rcrun.py --network lstm --save goolstm1 --end .3 --epochs 5
#python rcrun.py --network simplecnn --save goocnn2 --end .3 --epochs 5
#python rcrun.py --network simplernn --save goornn2 --end .3 --epochs 5
#python rcrun.py --network asym --save gooasym2 --end .3 --epochs 5
#python rcrun.py --network gru --save googru2 --end .3 --epochs 5
#python rcrun.py --network lstm --save goolstm2 --end .3 --epochs 5


#python srun.py --network simple --right  sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --save slowsimple --end 1. --epochs 10
#python srun.py --network lstmasym --right  sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --save slowlstmasym --end 1. --epochs 10

#20181224
#python srun.py --network lstm --right sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --save slowlstm2 --end 1. --epochs 10
#python srun.py --network simplecnn --right sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --save slowcnn2 --end 1. --epochs 10
#python srun.py --network asym --right  sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --save slowasym --end 1. --epochs 10
#python srun.py --network gru --right sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --save slowgru2 --end 1. --epochs 10
#python srun.py --network naasym --right sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --save slownaasym --end 1. --epochs 10
#python srun.py --network asym --right  sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --save slowasym2 --end 1. --epochs 10

#20181226
#python ensrun.py --save1 slowcnn --save2 slowcnn2 --right sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --end 1. --epoch 1
#python ensrun.py --save1 slowcnn --save2 slowasym2 --right sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --end 1. --epoch 1
#python ensrun.py --save1 slowgru --save2 slowlstm --right sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --end 1. --epoch 1
#python srun.py --network asym --right  sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --save slowasym --end 1. --epochs 10 --gpu 1
#python srun.py --network simplecnn --right  sdata/dijet_100_110/dijet_100_110_training --valright sdata/dijet_100_110/dijet_100_110_validation --save slowcnn --end 1. --epochs 10 --gpu 1

#python srun.py --network simplecnn --right  sdata/dijet_200_220/dijet_200_220_training --valright sdata/dijet_200_220/dijet_200_220_validation --save slow220cnn2 --end 1. --epochs 10 --gpu 1
#python srun.py --network gru --right  sdata/dijet_200_220/dijet_200_220_training --valright sdata/dijet_200_220/dijet_200_220_validation --save slow220gru2 --end 1. --epochs 10 --gpu 1

#python srun.py --network simplecnn --right  sdata/dijet_500_550/dijet_500_550_training --valright sdata/dijet_500_550/dijet_500_550_validation --save slow550cnn2 --end 1. --epochs 10 --gpu 1
#python srun.py --network simplelstm --right  sdata/dijet_500_550/dijet_500_550_training --valright sdata/dijet_500_550/dijet_500_550_validation --save slow550lstm --end 1. --epochs 10 --gpu 1
#python srun.py --network simplecnn --right  sdata/dijet_500_550/dijet_500_550_training --valright sdata/dijet_500_550/dijet_500_550_validation --save slow550cnn --end 1. --epochs 10 --gpu 1
#python srun.py --network simplegru --right  sdata/dijet_500_550/dijet_500_550_training --valright sdata/dijet_500_550/dijet_500_550_validation --save slow550gru --end 1. --epochs 10 --gpu 1
#python srun.py --network simple --right  sdata/dijet_500_550/dijet_500_550_training --valright sdata/dijet_500_550/dijet_500_550_validation --save slow550simple --end 1. --epochs 10 --gpu 1
#python srun.py --network lstmasym --right  sdata/dijet_500_550/dijet_500_550_training --valright sdata/dijet_500_550/dijet_500_550_validation --save slow550lstmasym --end 1. --epochs 10 --gpu 1

#20190102

declare -a arr=("gru" "simplecnn" "simplernn" "rnn" "lstm")
#declare -a arr=("gru" "asym" "simplecnn" "rnncnn" "rnn" "lstm" "simple" "lstmcnn")
for i in "${arr[@]}"
do
python srun.py --network ${i} --pt 100 --save slow100${i}2 --end 0.5 --epochs 10 --gpu 1
python srun.py --network ${i} --pt 200 --save slow200${i}2 --end 0.5 --epochs 10 --gpu 1
python srun.py --network ${i} --pt 500 --save slow500${i}2 --end 0.5 --epochs 10 --gpu 1
python srun.py --network ${i} --pt 1000 --save slow1000${i}2 --end 0.5 --epochs 10 --gpu 1
done

#20190103
python srun.py --network lstmcnn --pt 1000 --save slow1000lstmcnn --end 0.5 --epochs 10 --gpu 1
python srun.py --network lstmcnn --pt 500 --save slow500lstmcnn --end 0.5 --epochs 10 --gpu 1
python srun.py --network lstmcnn --pt 200 --save slow200lstmcnn --end 0.5 --epochs 10 --gpu 1
python srun.py --network lstmcnn --pt 100 --save slow100lstmcnn --end 0.5 --epochs 10 --gpu 1
