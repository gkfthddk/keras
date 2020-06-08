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

#declare -a arr=("gru" "simplecnn" "simplernn" "rnn" "lstm")
#declare -a arr=("gru" "asym" "simplecnn" "rnncnn" "rnn" "lstm" "simple" "lstmcnn")
#for i in "${arr[@]}"
#do
#python srun.py --network ${i} --pt 100 --save slow100${i}2 --end 0.5 --epochs 10 --gpu 1
#python srun.py --network ${i} --pt 200 --save slow200${i}2 --end 0.5 --epochs 10 --gpu 1
#python srun.py --network ${i} --pt 500 --save slow500${i}2 --end 0.5 --epochs 10 --gpu 1
#python srun.py --network ${i} --pt 1000 --save slow1000${i}2 --end 0.5 --epochs 10 --gpu 1
#done

#20190103
#python srun.py --network lstmcnn --pt 1000 --save slow1000lstmcnn --end 0.5 --epochs 10 --gpu 1
#python srun.py --network lstmcnn --pt 500 --save slow500lstmcnn --end 0.5 --epochs 10 --gpu 1
#python srun.py --network lstmcnn --pt 200 --save slow200lstmcnn --end 0.5 --epochs 10 --gpu 1
#python srun.py --network lstmcnn --pt 100 --save slow100lstmcnn --end 0.5 --epochs 10 --gpu 1

#20190104
#python srun.py --network simple --pt 100 --save ten100simple --end 0.3 --epochs 10 --gpu 1
#declare -a arr=("grucnn")
#declare -a pt=("200" "500" "1000")
#for i in "${arr[@]}"
#do
#for j in "${pt[@]}"
#do
#python srun.py --network ${i} --pt ${j} --save ten${j}${i} --end 1. --epochs 10 --gpu 1
#python tpred.py --save ten${j}${i} --pt ${j}
#done
#done

#20190106
#declare -a pt=("1000" "500")
#for j in "${pt[@]}"
#do
#python epred.py --net1 gru --net2 simplecnn --pt ${j} 
#done
#declare -a arr=("cnn")
#declare -a pt=("100" "200")
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
#python epred.py --net1 gru --net2 lstm4 --pt ${j}
#python epred.py --net1 simplecnn3 --net2 cnn4 --pt ${j}
#done

#python srun.py --network egru --pt 200 --save test1 --end 1 --epochs 10 --gpu 1
#python srun.py --network cgru --pt 200 --save test2 --end 1 --epochs 10 --gpu 1

#python srun.py --network lcnn --pt 200 --save local1 --end 1 --epochs 10 --gpu 1
#python srun.py --network lcnn2 --pt 200 --save local2 --end 1 --epochs 10 --gpu 1

#20190207
#python run.py --network rnn --pt 200 --save test --end 1 --epochs 10 --gpu 1
#python srun.py --network lcnn3 --pt 200 --save local3 --end 1 --epochs 10 --gpu 1
#python run.py --network lcnn --pt 200 --save lcnn --end 1 --epochs 10 --gpu 1
#python run.py --network lcnn3 --pt 200 --save lcnn3 --end 1 --epochs 10 --gpu 1
#python run.py --network rnn --pt 200 --save testrnn --end 1 --epochs 10 --gpu 1
#python run.py --network rnn2 --pt 200 --save testrnn2 --end 1 --epochs 10 --gpu 1
#python run.py --network rnn4 --pt 200 --save testrnn4 --end 1 --epochs 10 --gpu 1
#python run.py --network ernn --pt 200 --save testernn --end 1 --epochs 10 --gpu 0

#python testrun.py --network rnn --pt 100 --save testrnn1024 --end 0.3 --epochs 20 --batch_size 1024 --gpu 2
#python testrun.py --network rnn --pt 100 --save testrnn512 --end 0.3 --epochs 20 --batch_size 512 --gpu 2

#python run.py --network rnn3 --pt 100 --save testrnn3 --end 0.3 --epochs 30 --gpu 2
#python run.py --network rnn2 --pt 100 --save testrnn2 --end 0.3 --epochs 30 --gpu 2
#python run.py --network rnn --pt 100 --save testrnn --end 0.3 --epochs 30 --gpu 2

#declare -a pt=("100" "200")
#for j in "${pt[@]}"
#do
#python pred.py --save paperzjcnn${j} --pt ${j} --end 100000 --isz 0
#python pred.py --save paperzqcnn${j} --pt ${j} --end 100000 --isz 1
#python pred.py --save paperqqcnn${j} --pt ${j} --end 100000 --isz -1
#done
#python pred.py --save testrms --pt 1000 --end 10000 --isz 0

#python pred.py --save pepzjada1000 --pt 1000 --end 100000 --isz 0 --gpu 0
#python pred.py --save pepzjrnn500sgd --pt 500 --end 100000 --isz 0 --gpu 1
#python pred.py --save pepzjrnn200sgd --pt 200 --end 100000 --isz 0 --gpu 1
#python pred.py --save pepzjrnn100sgd --pt 100 --end 100000 --isz 0 --gpu 1
#python pred.py --save pepzjcnn200 --pt 200 --end 100000 --isz 0 --gpu 1

#declare -a etas=("0" "0.4" "0.8" "1.2" "1.6" "2.")
#for eta in "${etas[@]}"
#do
#python etapred.py --save pepzjcnn100model --pt 100 --end 100000 --isz 0 --gpu 1 --eta ${eta} --etabin 0.2 &
#python etapred.py --save pepzjcnn200model --pt 200 --end 100000 --isz 0 --gpu 2 --eta ${eta} --etabin 0.2 &
#python etapred.py --save pepzjcnn500model --pt 500 --end 100000 --isz 0 --gpu 4 --eta ${eta} --etabin 0.2 &
#python etapred.py --save pepzjcnn1000model --pt 1000 --end 100000 --isz 0 --gpu 3 --eta ${eta} --etabin 0.2
#done
#python getpred.py --save pepzjcnn100model --pt 100 --end 100000 --isz 0 --gpu 1 --eta 0 --etabin 2.4 &
#python getpred.py --save pepzjcnn200model --pt 200 --end 100000 --isz 0 --gpu 2 --eta 0 --etabin 2.4 &
#python getpred.py --save pepzjcnn500model --pt 500 --end 100000 --isz 0 --gpu 4 --eta 0 --etabin 2.4 &
#python getpred.py --save pepzjcnn1000model --pt 1000 --end 100000 --isz 0 --gpu 3 --eta 0 --etabin 2.4
#python modelrun.py --pt 500 --save asuzjcnn500pt1 --isz 0 --end 100000 --epochs 30 --gpu 2 --eta 0 --etabin 1 --ptmin 0. --ptmax 0.95 &
#python modelrun.py --pt 500 --save asuzjcnn500pt2 --isz 0 --end 100000 --epochs 30 --gpu 3 --eta 0 --etabin 1 --ptmin 0.95 --ptmax 1.05 &
#python modelrun.py --pt 500 --save asuzjcnn500pt3 --isz 0 --end 100000 --epochs 30 --gpu 4 --eta 0 --etabin 1 --ptmin 1.05 --ptmax 2.0 &
#python modelrun.py --pt 1000 --save asuzjcnn1000pt1 --isz 0 --end 100000 --epochs 30 --gpu 4 --eta 0 --etabin 1 --ptmin 0. --ptmax 0.95 &
#python modelrun.py --pt 1000 --save asuzjcnn1000pt2 --isz 0 --end 100000 --epochs 30 --gpu 3 --eta 0 --etabin 1 --ptmin 0.95 --ptmax 1.05 
#python modelrun.py --pt 1000 --save asuzjcnn1000pt3 --isz 0 --end 100000 --epochs 30 --gpu 2 --eta 0 --etabin 1 --ptmin 1.05 --ptmax 2.0
#python getpred.py --save asuzjcnn500pt1 --pt 500 --end 100000 --isz 0 --gpu 2 &
#python getpred.py --save asuzjcnn500pt2 --pt 500 --end 100000 --isz 0 --gpu 3 &
#python getpred.py --save asuzjcnn500pt3 --pt 500 --end 100000 --isz 0 --gpu 4 &
#python getpred.py --save asuzjcnn1000pt1 --pt 1000 --end 100000 --isz 0 --gpu 4 &
#python getpred.py --save asuzjcnn1000pt2 --pt 1000 --end 100000 --isz 0 --gpu 3 &
#python getpred.py --save asuzjcnn1000pt3 --pt 1000 --end 100000 --isz 0 --gpu 2 

#python pred.py --save pepqqcnn100model --pt 100 --end 100000 --isz -1 --gpu 0 &
#python pred.py --save pepqqcnn200model --pt 200 --end 100000 --isz -1 --gpu 1 &
#python pred.py --save pepqqcnn500model --pt 500 --end 100000 --isz -1 --gpu 2 &
#python pred.py --save pepqqcnn1000model --pt 1000 --end 100000 --isz -1 --gpu 3
#python vpred.py --save pepzjcnn100model --pt 100 --end 100000 --isz 0 --gpu 3 &
#python vpred.py --save pepzjcnn200model --pt 200 --end 100000 --isz 0 --gpu 1 &
#python vpred.py --save pepzjcnn500model --pt 500 --end 100000 --isz 0 --gpu 0 &
#python vpred.py --save pepzjcnn1000model --pt 1000 --end 100000 --isz 0 --gpu 2
#python vpred.py --save pepzjrnn1000sgd --pt 1000 --end 100000 --isz 0 --gpu 0 &
#python vpred.py --save pepzjrnn500sgd --pt 500 --end 100000 --isz 0 --gpu 1 &
#python vpred.py --save pepzjrnn100sgd --pt 100 --end 100000 --isz 0 --gpu 2 
#python vpred.py --save pepzjrnn200sgd --pt 200 --end 100000 --isz 0 --gpu 3 

#python gausrun.py --pt 1000 --save asuzjcnn1000ptgaus --isz 0 --end 100000 --epochs 30 --gpu 5 --eta 0 --etabin 1 &
#python gausrun.py --pt 500 --save asuzjcnn500ptgaus --isz 0 --end 100000 --epochs 30 --gpu 6 --eta 0 --etabin 1 &
#python gausrun.py --pt 200 --save asuzjcnn200ptgaus --isz 0 --end 100000 --epochs 30 --gpu 6 --eta 0 --etabin 1 
#python gausrun.py --pt 100 --save asuzjcnn100ptgaus --isz 0 --end 100000 --epochs 30 --gpu 5 --eta 0 --etabin 1
#python getpred.py --save asuzjcnn1000ptgaus --pt 1000 --end 100000 --isz 0 --gpu 5 & 
#python getpred.py --save asuzjcnn500ptgaus --pt 500 --end 100000 --isz 0 --gpu 5 &
#python getpred.py --save asuzjcnn200ptgaus --pt 200 --end 100000 --isz 0 --gpu 6 &
#python getpred.py --save asuzjcnn100ptgaus --pt 100 --end 100000 --isz 0 --gpu 6 
#python modelrun.py --pt 1000 --save asuzqcnn1000noeta --isz 1 --end 100000 --epochs 20 --gpu 5 --eta 0 --etabin 2.4 &
#python modelrun.py --pt 500 --save asuzqcnn500noeta --isz 1 --end 100000 --epochs 20 --gpu 6 --eta 0 --etabin 2.4 &
#python modelrun.py --pt 200 --save asuzqcnn200noeta --isz 1 --end 100000 --epochs 20 --gpu 7 --eta 0 --etabin 2.4 
#python modelrun.py --pt 100 --save asuzqcnn100noeta --isz 1 --end 100000 --epochs 20 --gpu 5 --eta 0 --etabin 2.4 &
#python modelrun.py --pt 1000 --save asuqqcnn1000noeta --isz -1 --end 100000 --epochs 20 --gpu 7 --eta 0 --etabin 2.4 &
#python modelrun.py --pt 500 --save asuqqcnn500noeta --isz -1 --end 100000 --epochs 20 --gpu 6 --eta 0 --etabin 2.4 
#python modelrun.py --pt 200 --save asuqqcnn200noeta --isz -1 --end 100000 --epochs 20 --gpu 7 --eta 0 --etabin 2.4 &
#python modelrun.py --pt 100 --save asuqqcnn100noeta --isz -1 --end 100000 --epochs 20 --gpu 5 --eta 0 --etabin 2.4
#python getpred.py --save asuzqcnn1000noeta --pt 1000 --end 100000 --isz 1 --gpu 5 & 
#python getpred.py --save asuzqcnn500noeta --pt 500 --end 100000 --isz 1 --gpu 6 &
#python getpred.py --save asuzqcnn200noeta --pt 200 --end 100000 --isz 1 --gpu 5 &
#python getpred.py --save asuzqcnn100noeta --pt 100 --end 100000 --isz 1 --gpu 6 
#python getpred.py --save asuqqcnn1000noeta --pt 1000 --end 100000 --isz -1 --gpu 5 & 
#python getpred.py --save asuqqcnn500noeta --pt 500 --end 100000 --isz -1 --gpu 6 &
#python getpred.py --save asuqqcnn200noeta --pt 200 --end 100000 --isz -1 --gpu 5 &
#python getpred.py --save asuqqcnn100noeta --pt 100 --end 100000 --isz -1 --gpu 6 

#python checkhistory.py --save asuzjcnn100noeta
#python checkhistory.py --save asuzjcnn200noeta
#python checkhistory.py --save asuzjcnn500noeta
#python checkhistory.py --save asuzjcnn1000noeta
#python checkhistory.py --save asuzjcnn100
#python checkhistory.py --save asuzjcnn200
#python checkhistory.py --save asuzjcnn500
#python checkhistory.py --save asuzjcnn1000
#python checkhistory.py --save asuzjcnn100pt
#python checkhistory.py --save asuzjcnn200pt
#python checkhistory.py --save asuzjcnn500pt
#python checkhistory.py --save asuzjcnn1000pt
#python checkhistory.py --save asuzjcnn100ptgaus
#python checkhistory.py --save asuzjcnn200ptgaus
#python checkhistory.py --save asuzjcnn500ptgaus
#python checkhistory.py --save asuzjcnn1000ptgaus
#python saver.py --pt 500 --network rnn31 --save asuzjrnn500ptonly3 --isz 0 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000
#python saver.py --pt 100 --network rnn31 --save asuzjrnn500ptonly3 --isz 0 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000
#python saver.py --pt 200 --network rnn31 --save asuzjrnn500ptonly3 --isz 0 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000
#python saver.py --pt 1000 --network rnn31 --save asuzjrnn500ptonly3 --isz 0 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000

#python saver.py --pt 500 --network rnn31 --save asuzjrnn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000
#python saver.py --pt 100 --network rnn31 --save asuzjrnn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000
#python saver.py --pt 200 --network rnn31 --save asuzjrnn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000
#python saver.py --pt 1000 --network rnn31 --save asuzjrnn500ptonly3 --isz -1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 2.4 --ptmin 0.821 --ptmax 1.093 --batch_size 100000
#python saver.py --pt 500 --network rnn31 --save asuzjrnn500ptonly3 --isz 1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000
#python saver.py --pt 100 --network rnn31 --save asuzjrnn500ptonly3 --isz 1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000
#python saver.py --pt 200 --network rnn31 --save asuzjrnn500ptonly3 --isz 1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000
#python saver.py --pt 1000 --network rnn31 --save asuzjrnn500ptonly3 --isz 1 --end 100000 --epochs 100 --gpu 1 --eta 0 --etabin 1 --ptmin 0.821 --ptmax 1.093 --batch_size 100000

#20190910
#python jetdual.py --save dualnb500con --network nnb --pt 500 --epoch 20 --stride 2 --gpu 0 --pred 1 --seed con &
#python jetdual.py --save dualnk200add --network nnk --pt 200 --epoch 30 --stride 2 --gpu 2 --pred 1 --seed add & 
#python jetdual.py --save dualnk200sub --network nnk --pt 200 --epoch 30 --stride 2 --gpu 3 --pred 1 --seed sub &
#python jetdual.py --save dualnk200ave --network nnk --pt 200 --epoch 30 --stride 2 --gpu 4 --pred 1 --seed ave 
#python jetdual.py --save dualnk200non --network nnk --pt 200 --epoch 30 --stride 2 --gpu 3 --pred 1 --seed non &
#python jetdual.py --save dualnk200con --network nnk --pt 200 --epoch 30 --stride 2 --gpu 2 --pred 1 --seed con &
#python jetdual.py --save dualnk200min --network nnk --pt 200 --epoch 30 --stride 2 --gpu 4 --pred 1 --seed min &
#python jetdual.py --save dualnk200max --network nnk --pt 200 --epoch 30 --stride 2 --gpu 5 --pred 1 --seed max 
#python jetdual.py --save dualnk200mul --network nnk --pt 200 --epoch 30 --stride 2 --gpu 6 --pred 1 --seed mul &
##python jetdual.py --save dualnb500dot --network nnb --pt 500 --epoch 20 --stride 2 --gpu 7 --pred 1 --seed dot
#python jetdual.py --save dualna200 --network nna3 --pt 200 --epoch 30 --stride 2 --gpu 7 --pred 1
#python jetdual.py --save dualc500con --network cnn --pt 500 --epoch 30 --stride 2 --gpu 4 --pred 1 --seed con --opt adam &
#python jetdual.py --save dualc500non --network cnn --pt 500 --epoch 30 --stride 2 --gpu 5 --pred 1 --seed non --opt adam 
#python jetdual.py --save dualc500min --network cnn --pt 500 --epoch 30 --stride 2 --gpu 4 --pred 1 --seed min --opt adam &
#python jetdual.py --save dualc500max --network cnn --pt 500 --epoch 30 --stride 2 --gpu 5 --pred 1 --seed max --opt adam 
#python jetdual.py --save dualc500non --network cnn --pt 500 --epoch 30 --stride 2 --gpu 1 --pred 1 --seed non 
#python pfrun.py --save pfr --network pfr --pt 200 --epoch 30 --stride 2 --gpu 1 --pred 1 --seed non &
#python pfrun.py --save pfk --network pfk --pt 200 --epoch 30 --stride 2 --gpu 2 --pred 1 --seed non

#20191021
#python jetboost.py --pt 200 --gpu 1 --save bdt1pt2003 --stride 3 --pred 1
#python genboost.py --pt 200 --gpu 1 --save genbdt1pt2001 --stride 1 --pred 1 --jet 0
#python genboost.py --pt 200 --gpu 1 --save genbdt2pt2001 --stride 1 --pred 1 --jet 1
#python genboost.py --pt 200 --gpu 1 --save gen1pt2002 --stride 2 --pred 1
#python jetboost.py --pt 500 --gpu 1 --save bdt1pt5003 --stride 3 --pred 1
#python genboost.py --pt 500 --gpu 1 --save genbdt1pt5001 --stride 1 --pred 1 --jet 0
#python genboost.py --pt 500 --gpu 1 --save genbdt2pt5001 --stride 1 --pred 1 --jet 1
#python genboost.py --pt 500 --gpu 1 --save gen1pt5002 --stride 2 --pred 1
#python jetxpred.py --pt 200 --save bdt2pt2001 --stride 1
#python jetxpred.py --pt 200 --save bdt2pt2002 --stride 2
#python jetxpred.py --pt 200 --save bdt2pt2003 --stride 3
#python jetxpred.py --pt 500 --save bdt2pt5001 --stride 1
#python jetxpred.py --pt 500 --save bdt2pt5002 --stride 2
#python jetxpred.py --pt 500 --save bdt2pt5003 --stride 3

#20200421
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5011 --channel 1
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5021 --channel 2
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5031 --channel 3
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5041 --channel 4
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5051 --channel 5
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5061 --channel 6
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5071 --channel 7
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5081 --channel 8
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5012 --channel 1
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5022 --channel 2
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5032 --channel 3
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5042 --channel 4
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5052 --channel 5
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5062 --channel 6
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5072 --channel 7
#python drrun.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg5082 --channel 8
#for i in {1..100}
#do
#python drrun2.py --gpu 2 --epochs 50 --pt 50 --opt adam --memo qg50
#python drrun.py --gpu 1 --epochs 50 --pt 50 --opt adam --memo qg50 --channel 4
#done
