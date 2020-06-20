# In this archive we have to define the dictionary ml_info. This is a dictionary of dictionaries, that for each of the ML models we want
# assigns a dictionary that contains:
#
# clf: a scikit-learn classifier, or any object that implements the functions fit, and predict_proba or decision_function in the same way.
# formal_name: name to be used in plots and report
#
# In this archive we provide 4 examples:
# RF for Random Forest
# BT for Boosted Trees
# LR for Logistic Regression
# RF_pipeline for a Random Forest with hyperparameter tuning including the choice of feature selection strategy

import sklearn.model_selection as sk_ms
import sklearn.ensemble as sk_en
import sklearn.linear_model as sk_lm
import sklearn.pipeline as sk_pl
import sklearn.svm as sk_svm
import sklearn.preprocessing as sk_pp
from utils.featureselecter import FeatureSelecter

ML_info ={}

pipeline_svm = sk_pl.Pipeline(steps=[('std_scaler', sk_pp.StandardScaler()),("svm",sk_svm.SVC())])

pipeline_rf = sk_pl.Pipeline(steps=[('std_scaler', sk_pp.StandardScaler()),("rf",sk_en.RandomForestClassifier())])

pipeline_lr = sk_pl.Pipeline(steps=[('std_scaler', sk_pp.StandardScaler()),("lr",sk_lm.LogisticRegression())])


grid_params_svm=[{'svm__C':[0.1, 1, 10],
                  'svm__kernel':['linear']
#                   'svm__class_weight':['balanced'],
#                   'svm__probability':[True]
                 }]

grid_params_rf=[{'rf__max_features':['auto', 1],
#                  'rf__criterion':['entropy'],
                 'rf__max_depth':[1, 2, 5, None]}]

grid_params_lr=[{'lr__penalty':['elasticnet'],
                 'lr__l1_ratio':[0,0.5,1],
                 'lr__C':[0.1, 1, 10],
                 'lr__solver':['saga']}]


tuned_svm=sk_ms.GridSearchCV(pipeline_svm,grid_params_svm, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

tuned_rf=sk_ms.GridSearchCV(pipeline_rf,grid_params_rf, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

tuned_lr=sk_ms.GridSearchCV(pipeline_lr,grid_params_lr, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)


ML_info['SVM_pipeline'] = {'formal_name': 'Support Vector Machine',
                          'clf': tuned_svm}
ML_info['RF_pipeline'] = {'formal_name': 'Random Forest',
					'clf': tuned_rf}
ML_info['LR_pipeline'] = {'formal_name': 'Logistic Regression',
					'clf': tuned_lr}


# ML_info['BT'] = {'formal_name': 'XGBoost',
# 					'clf': xgb.XGBClassifier(n_estimators=1000)}
# pipeline_rf = sk_pl.Pipeline(steps=[("fs",FeatureSelecter()),("rf",sk_en.RandomForestClassifier(n_estimators = 1000,  max_features = 'auto'))])
# grid_params_rf=[{'fs__method':['eq','sfm_rf', 'skb_10'],'rf__n_estimators':[100,1000],
# 					'rf__max_features':[1,'auto'], 'rf__criterion':['gini','entropy'], 'rf__max_depth':[None, 1,2,5]}]
# tuned_rf=sk_ms.GridSearchCV(pipeline_rf,grid_params_rf, cv=10,scoring ='roc_auc', return_train_score=False, verbose=1)

# ML_info['RF_pipeline'] = {'formal_name': 'Random Forest (Hyperparameter Tuning)',
# 						  'clf': tuned_rf}
