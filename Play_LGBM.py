from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import joblib
import pandas as pd
import os
from numpy import mean,std

RESP_SORT_LS=[]
TREE_EST=[200] #20,50,75,100,150,200,300,500,750,1000
K_fold_splite=[3] #3,4,6,8,12
LGBM_SOCRE=[]
result_df=pd.DataFrame()
# Load the ML feature and Response for prediction
FEAT=pd.read_csv('./TRAIN_TBL.csv',index_col=False)
RESP=pd.read_csv('./REPONSE.csv',index_col=False)
# drop the feature column - AQ_ST name
GET_AQ_site=FEAT['AQ_ST']
FEAT.drop('AQ_ST',axis=1,inplace=True)
FEATURES=FEAT.drop('Unnamed: 0',axis=1)
# Pick the response value according based on the correct site name 
for X_AQ in GET_AQ_site:
    IND=RESP[RESP.AQ_ST==X_AQ].index
    RESP_SORT_LS.append(RESP['WIN_AVG'][IND[0]])
# convert the ls to dataframe
AQ_df=pd.DataFrame()
AQ_df['AQ_WINTER']=RESP_SORT_LS
# define model
for X_FOLD in K_fold_splite:
    for X_TREE in TREE_EST:
        model=LGBMRegressor(n_estimators=X_TREE,learning_rate=0.05)
        cv=RepeatedKFold(n_splits=X_FOLD,n_repeats=3)
        n_scores=cross_val_score(model,FEATURES,AQ_df,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1,error_score='raise')
        print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
        LGBM_SOCRE.append(mean(n_scores))
        model.fit(FEATURES,AQ_df)
    result_df['TreeNUM']=TREE_EST
    result_df['neg_MAE']=LGBM_SOCRE
    result_df.to_csv('./'+'LGBM_'+str(X_FOLD)+'.csv')
    LGBM_SOCRE=[]
    result_df=pd.DataFrame()
SAVE_ML_MODEL='/MODELS/'
joblib.dump(model,SAVE_ML_MODEL+'model.sav')
