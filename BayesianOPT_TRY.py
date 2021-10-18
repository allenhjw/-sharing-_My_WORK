    import joblib
    import os
    #from lightgbm import LGBMRegressor
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.model_selection import RepeatedKFold
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern,RationalQuadratic,WhiteKernel
    from sklearn.metrics import explained_variance_score 
    from sklearn.svm import SVR
    from sklearn import preprocessing as PREPRO
    import pandas as pd
    import numpy as np
    #from sklearn.preprocessing import normalizer

    #import xgboost
    #import lightgbm
    from bayes_opt import BayesianOptimization
    from statistics import mean

    Grid_SCORE=[]
    time_start=time.time()
    RESP_SORT_LS=[]

    # Load the ML feature and Response for prediction
    FEAT=pd.read_csv('./WORK_PH'+'Predictor.csv',index_col=False)
    RESP=pd.read_csv('./WORK_PH'+'Response.csv',index_col=False)
    # drop the feature column - AQ_ST name
    GET_AQ_site=FEAT['Name']
    FEAT.drop('Name',axis=1,inplace=True)
    FEATURES=FEAT.drop('Unnamed: 0',axis=1)
    # Pick the response value according based on the correct site name 
    for X_AQ in GET_AQ_site:
        IND=RESP[RESP.AQ_ST==X_AQ].index
        RESP_SORT_LS.append(RESP['Value'][IND[0]])
    # convert the ls to dataframe
    AQ_df=pd.DataFrame()
    AQ_df['Value']=RESP_SORT_LS
    TRAIN_X,TEST_Y,Resp_X, Resp_Y=train_test_split(FEATURES,AQ_df,test_size=0.25,random_state=0)
    # standardize the data [ONLY for GPR]
    #TRAIN_X_PRO=PREPRO.scale(TRAIN_X)

    def PROCESS_DATA_NORM_COL(IN_DATA):
        #COL_HEAD=['MjTR','MinTR','HSE']
        PREPRO_df=pd.DataFrame()
        buffer_LS=[]
        for X_COL in IN_DATA.columns:
            #IN_DATA_PROC=StandardScaler().fit(IN_DATA[X_COL])
            #IN_DATA_PROC=PREPRO.scale(IN_DATA[X_COL])
            #IN_DATA_PROC=PREPRO.normalize(IN_DATA[X_COL])
            #a=PREPRO.normalize(IN_DATA[X_COL].values.reshape(-1,1))
            NORM=np.linalg.norm(IN_DATA[X_COL])
            a=IN_DATA[X_COL]/NORM
            for i in a:
                buffer_LS.append(i)
            #IN_DATA_PROC=IN_DATA_PROC.reshape(len(IN_DATA_PROC),1)
            PREPRO_df[X_COL]=a
            buffer_LS=[]
        return PREPRO_df

    def XGBOOST_R(n_estimators,max_depth,reg_alpha,reg_lambda,min_child_weight,num_boost_round,gamma):

        params={"booster": 'gbtree',
                "objective" : "reg:squarederror",
                "eval_metric" : "auc", 
                "is_unbalance":True,
                "n_estimators":int(n_estimators),
                "max_depth" :int(max_depth),
                "reg_alpha" :reg_alpha,
                "reg_lambda" :reg_lambda,
                "gamma": gamma,
                "num_threads" :20,
                "min_child_weight" : int(min_child_weight),
                "learning_rate" :0.01,
                "subsample_freq" :5,
                "seed" :42,
                "verbosity" :0,
                "num_boost_round":int(num_boost_round),
                "eval_metric":'mae'}

        Train_DATA=xgboost.DMatrix(TRAIN_X,Resp_X)
        CV_result=xgboost.cv(params,Train_DATA,1000,nfold=3,early_stopping_rounds=100,stratified=False,)
        a=(CV_result['test-mae-mean'].iloc[-1])*-1
        return a

    def lIGHTGBM_R(num_leaves,learning_rate,n_estimators,min_split_gain,
                    reg_alpha,reg_lambda,random_state):

        params = {"boosting_type": 'gbdt',
                "num_leaves" : int(num_leaves),
                "max_depth":-1,
                "learning_rate": learning_rate, 
                "n_estimators": int(n_estimators),
                "min_split_gain": float(min_split_gain),
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "random_state": int(random_state)}

        #train_data = lightgbm.Dataset(TRAIN_X, Resp_X)
        #cv_result = lightgbm.cv(params,
        #                   train_data,
        #                   1000,
        #                   early_stopping_rounds=100,
        #                   stratified=False,
        #                   nfold=3)
        ML_model=LGBMRegressor(**params)
        cv_result=cross_val_score(ML_model,TRAIN_X,Resp_X['AQ_AVG'].ravel(),cv=3,scoring='neg_mean_absolute_error',error_score='raise',verbose=0)
        MAE=mean(cv_result)
        return MAE

    def RF_R(n_estimators,max_depth,min_samples_split,min_samples_leaf,random_state):

        params = {'n_estimators':int(n_estimators),
                'criterion':'mae',
                'max_depth':int(max_depth),
                'min_samples_split':int(min_samples_split),
                'min_samples_leaf':int(min_samples_leaf),
                'min_weight_fraction_leaf':0,
                'max_features':'auto',
                'n_jobs':0,
                'random_state':int(random_state),
                'verbose':0}

        ML_model=RandomForestRegressor(**params)
        cv_result=cross_val_score(ML_model,TRAIN_X,Resp_X['AQ_AVG'].ravel(),cv=8,scoring='neg_mean_absolute_error',error_score='raise',verbose=0)
        MAE=mean(cv_result)
        return MAE

    def GradBoost_R(n_estimators,learning_rate,max_depth,min_samples_split,min_samples_leaf,random_state,alpha):

        params = {'n_estimators':int(n_estimators),
                'loss':'quantile',
                'learning_rate':float(learning_rate),
                'criterion':'friedman_mse',
                'max_depth':int(max_depth),
                'min_samples_split':int(min_samples_split),
                'min_samples_leaf':int(min_samples_leaf),
                'min_weight_fraction_leaf':0,
                'max_features':'sqrt',
                'random_state':int(random_state),
                'alpha':float(alpha),
                'verbose':0}
        ML_model=GradientBoostingRegressor(**params)
        cv_result=cross_val_score(ML_model,TRAIN_X,Resp_X['AQ_AVG'].ravel(),cv=8,scoring='neg_mean_absolute_error',error_score='raise',verbose=0) #neg_mean_absolute_error
        MAE=mean(cv_result)
        return MAE

    def GPR (alpha,C_length_scale,KM_length_scale,W_length_scale,nu,n_restarts_optimizer,random_state):
        KERNEL=Matern(KM_length_scale,(1e-5,1e5),nu)
        #KERNEL=C(C_length_scale,(1e-4,1e4))
        #KERNEL= C(C_length_scale,(1e-4,1e4))*RBF(KM_length_scale,(1e-5,1e6))
        #KERNEL=C(C_length_scale,(1e-3,1e4))*RationalQuadratic(KM_length_scale,alpha_bounds=(1e-5,1e5))
        #KERNEL= C(C_length_scale,(1e-2,1e4))*Matern(KM_length_scale,(1e-5,1e5),nu) 
        #+ WhiteKernel(noise_level=W_length_scale, noise_level_bounds=(1e-5, 1e+5))
        params = {'kernel':KERNEL,
                'n_restarts_optimizer':int(n_restarts_optimizer),
                'alpha':alpha,
                'random_state':int(random_state)}
        ML_model=GaussianProcessRegressor(**params)
        cv_result=cross_val_score(ML_model,TRAIN_X,Resp_X['AQ_AVG'].ravel(),cv=8,scoring='neg_mean_absolute_error',error_score='raise',verbose=0) #neg_mean_absolute_error
        MAE=mean(cv_result)
        return MAE

    # ONLY applicable for GPR. DO NOT SCALE THE VALUE FOR TREE BASED MODEL
    #FEATURES_PROCE=PROCESS_DATA_NORM_COL(FEATURES)
    #FEATURES_PROCE=FEATURES_PROCE.fillna(0)
    #AQ_PROCE=PROCESS_DATA_NORM_COL(AQ_df)

    #TRAIN_X,TEST_Y,Resp_X, Resp_Y=train_test_split(FEATURES_PROCE,AQ_df,test_size=0.25,random_state=0)
    ########################################

    #n_estimators_seq=[X for X in range (10,100,10)]
    n_estimators_seq=(10,20,30,40,50,60,70,80,90,100)
    max_depth_seq=[X for X in range (3,40,1)]
    reg_alpha_seq=[X for X in np.logspace(1,2,num=10,base=10)]
    nu=[X for X in np.logspace(0,1,num=10,base=10)]
    reg_lambda_seq=[X for X in np.logspace(-2,-1,num=10,base=10)]
    min_child_weight_seq=[X for X in range (1,10,1)]
    num_boost_round_seq=[X for X in range (100,1000,100)]
    gamma_seq=[X for X in range (0,10,1)]

    # Mask out those you don't need. 

    #OPMODEL = BayesianOptimization(XGBOOST_R, {"n_estimators": (10,1000),
    #                                        'max_depth': (3,50),
    #                                        'reg_alpha': (0.01,0.1),
    #                                        'reg_lambda': (0.01,0.1),
    #                                        'min_child_weight': (1,10),
    #                                        'num_boost_round': (100,1000),
    #                                        "gamma": (0,10)})

    #OPMODEL = BayesianOptimization(lIGHTGBM_R, {'num_leaves':(10,50),
    #                                        'learning_rate': (0.001,0.1),
    #                                        'n_estimators': (10,1000),
    #                                        'min_split_gain': (0,1),
    #                                        'reg_alpha': (0.01,1),
    #                                        'reg_lambda': (0.01,1),
    #                                        "random_state": (3,50)})

    OPMODEL=BayesianOptimization(RF_R, {'n_estimators':(20,20000),
                                    'max_depth':(3,60),
                                    'min_samples_split':(2,30),
                                    'min_samples_leaf':(1,30),
                                    'random_state':(0,60)})                              

    #OPMODEL=BayesianOptimization(GPR,{'alpha':(1e-4,1),
    #                                'C_length_scale':(1e-3,1e-1),
    #                                'KM_length_scale':(1e-5,1e-1),
    #                                'W_length_scale':(1,10),
    #                                'nu':(1e-3,1),
    #                                'n_restarts_optimizer':(1,15),
    #                                'random_state':(3,50)})


    #OPMODEL=BayesianOptimization(GradBoost_R, {'n_estimators':(20,20000),
    #                                        'learning_rate':(0.001,0.1),
    #                                        'max_depth':(3,60),
    #                                        'min_samples_split':(2,30),
    #                                        'min_samples_leaf':(1,30),
    #                                        'random_state':(0,60),
    #                                        'alpha':(0.001,0.99)})        

    OPMODEL.maximize(n_iter=70, init_points=5) 
    Best_PARAMS=OPMODEL.max['params']
    print (Best_PARAMS)
    # fit the model with the best parameters
    ML_model=RandomForestRegressor(n_estimators=int(Best_PARAMS['n_estimators']),max_depth=int(Best_PARAMS['max_depth']),
        min_samples_split=int(Best_PARAMS['min_samples_split']),min_samples_leaf=int(Best_PARAMS['min_samples_leaf']),
        random_state=int(Best_PARAMS['random_state']))
    
    # Mask out those you don't  need.

    #ML_model=GaussianProcessRegressor(alpha=float(Best_PARAMS['alpha']),kernel=KERNEL,n_restarts_optimizer=int(Best_PARAMS['n_restarts_optimizer']),
    #    random_state=int(Best_PARAMS['random_state']))

    #ML_model=GradientBoostingRegressor(n_estimators=int(Best_PARAMS['n_estimators']),max_depth=int(Best_PARAMS['max_depth']),
    #    min_samples_split=int(Best_PARAMS['min_samples_split']),min_samples_leaf=int(Best_PARAMS['min_samples_leaf']),
    #    random_state=int(Best_PARAMS['random_state']),alpha=float(Best_PARAMS['alpha']))    

    #ML_model=LGBMRegressor(learning_rate=float(Best_PARAMS['learning_rate']),num_leaves=int(Best_PARAMS['num_leaves']),
    #    n_estimators=int(Best_PARAMS['n_estimators']),min_split_gain=float(Best_PARAMS['min_split_gain']),
    #    reg_alpha=float(Best_PARAMS['reg_alpha']),reg_lambda=float(Best_PARAMS['reg_lambda']),random_state=int(Best_PARAMS['random_state']))    


    # for others models----
    ML_model.fit(TRAIN_X,Resp_X)
    Y_TRUE,Y_PRED=Resp_Y,ML_model.predict(TEST_Y)

    AE=abs(Y_TRUE-Y_PRED.reshape(6,1))
    MAE=mean(AE['AQ_AVG'])
    print(MAE)

    SAVE_ML_MODEL='./WORK_PH/MODELS/'
    joblib.dump(ML_model,SAVE_ML_MODEL+'[BAYOPT]MODEL.sav')
