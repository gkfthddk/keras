import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import scipy.stats as sts
import xgboost as xgb
import pandas as pd
import argparse
from datetime import datetime
#python jetboost.py --pt 500 --gpu 4 --save jettest
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
parser=argparse.ArgumentParser()
parser.add_argument("--end",type=float,default=100000.,help='end ratio')
parser.add_argument("--save",type=str,default="test_",help='save name')
parser.add_argument("--network",type=str,default="rnn",help='network name on symbols/')
parser.add_argument("--right",type=str,default="/scratch/yjdata/gluon100_img",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--pt",type=int,default=200,help='pt range pt~pt*1.1')
parser.add_argument("--ptmin",type=float,default=0.,help='pt range pt~pt*1.1')
parser.add_argument("--ptmax",type=float,default=2.,help='pt range pt~pt*1.1')
parser.add_argument("--epochs",type=int,default=10,help='num epochs')
parser.add_argument("--batch_size",type=int,default=100000,help='batch_size')
parser.add_argument("--loss",type=str,default="categorical_crossentropy",help='network name on symbols/')
parser.add_argument("--gpu",type=int,default=0,help='gpu number')
parser.add_argument("--isz",type=int,default=0,help='0 or z or not')
parser.add_argument("--eta",type=float,default=0.,help='end ratio')
parser.add_argument("--etabin",type=float,default=1,help='end ratio')
parser.add_argument("--stride",type=int,default=2,help='stride')
parser.add_argument("--pred",type=int,default=0,help='stride')

args=parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
batch_size=args.batch_size

params = {
        'max_depth': sts.randint(1,6),
        'learning_rate': sts.uniform(0.0010,0.500),
        'n_estimators': sts.randint(10,101)
        }
model=xgb.XGBClassifier(objective='binary:logistic',tree_method="gpu_hist")
loaded=np.load("jjtt{}.npz".format(args.pt))
X=loaded["bdtset"][:,:5*args.stride]
Y=loaded["labelset"]
X=X[:int(90000*0.7)]
Y=Y[0,:int(90000*0.7),0]
folds = 3
param_comb = 100

skf = KFold(n_splits=folds, shuffle = True, random_state = 173)
#skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=6, cv=skf.split(X,Y), verbose=3, random_state=173 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, Y)
timer(start_time) 

#print(random_search.predict(X[:10]))

#print('\n All results:')
#print(random_search.cv_results_)
#print('\n Best estimator:')
#print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
#print('\n Best hyperparameters:')
#print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb/{}-{}.csv'.format(args.save,args.pt), index=False)

#import pickle
#pickle.dump(random_search.best_estimator_,open("xgb/bdt5500pickle-{}.dat".format(args.pt),'wb'))
#random_search.best_estimator_.save_model("bdt-{}.dat".format(args.pt))

if(args.pred==1):os.system("python jetxpred.py --pt {} --save {} --stride {}".format(args.pt,args.save,args.stride))
