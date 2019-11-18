
from symbols import symbols
#model=symbols.jetcon("nnn3",2) 
#model.summary()
#model=symbols.jetcon("rnn",2,"con") 
#model.summary()
#model=symbols.pfcon("pfc",2,"non")
#model=symbols.jetcnn(stride=2,seed="con")
model=symbols.jetcon("enn",stride=1,seed="non")
model.summary()
#model=symbols.modelss()
#model.summary()
#model3=symbols.jetcon(3) 
#model=symbols.get_symbol("jet1nnn") 
