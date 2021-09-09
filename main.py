from explanations.explainer import Explanation_reg,Explanation_cls
from target_models.model import loadDataset,train_model,train_test_split
from evaluation.metrics import metrics_cls,metrics_reg
import shap
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def Main_reg(dataset):
    X,y = loadDataset(dataset)
    sc = StandardScaler()
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    X_scaled = sc.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    X_trainS,X_testS,y_trainS,y_testS = train_test_split(X_scaled,y)

    print("--------------------------------------------------------------")
    print("Training Logistic regression, Random Forest classifier and SVM classifier ...")
    LR_model = train_model("Linear Regression",X_train,y_train)
    RF_model = train_model("Random Forest Regressor",X_train,y_train)
    SVR_model = train_model("SVR",X_trainS,y_trainS)
    print("Done Training")
    print("--------------------------------------------------------------")
    print("Model Evaluation")
    print("")
    print("Linear Regression")
    print("R2 for Train", LR_model.score( X_train, y_train ))
    print('R2 for Test (cross validation)', LR_model.score(X_test, y_test))
   
    print("Random Forest Regression")
    print('R2 for Train', RF_model.score( X_train, y_train ))
    print('R2 for Test (cross validation)', RF_model.score(X_test, y_test))
   
    print("Random Forest Regression")
    print('R2 for Train', SVR_model.score( X_trainS, y_trainS))
    # print('R2 for Test (cross validation)', r2_score(y_testS, sc.inverse_transform(SVC_model.predict(X_testS))))

    X100 = shap.maskers.Independent(X, max_samples=100)
    X100_ = shap.utils.sample(X, 100)
    X100S = shap.maskers.Independent(X_trainS, max_samples=100)
    X100_S = shap.utils.sample(X_trainS, 100)
    print("--------------------------------------------------------------")
    print("Building Explanation ...")
    LR_shap,LR_baseVal = Explanation_reg("SHAP",LR_model,X_test.iloc[100:,],X100)
    LR_shap_k,RF_expected_val_k = Explanation_reg("Kernel SHAP",LR_model,X_test.iloc[100:,],X100_)
    # LR_lime1 = Explanation_reg("LIME",LR_model,X_test,X100)
    LR_lime = Explanation_reg("LIME-SHAP",LR_model,X_test,X100)

    RF_shap, RF_baseVal = Explanation_reg("SHAP",RF_model,X_test.iloc[100:,],X100)
    RF_shap_k,RF_expected_val_k = Explanation_reg("Kernel SHAP",RF_model,X_test.iloc[100:,],X100_)
    # RF_lime1 = Explanation_reg("LIME",RF_model,X_test,X100)
    RF_lime = Explanation_reg("LIME-SHAP",RF_model,X_test,X100)

    SVR_shap, SVR_baseVal = Explanation_reg("SHAP",SVR_model,X_testS[100:,],X100)
    SVR_shap_k,SVR_expected_val_k = Explanation_reg("Kernel SHAP",SVR_model,X_testS[100:,],X100_)
    # RF_lime1 = Explanation_reg("LIME",SVR_model,X_test,X100)
    SVR_lime = Explanation_reg("LIME-SHAP",SVR_model,X_testS,X100)
    print("Done building Explanation")


    faithfulness_LR_shap = metrics_reg(model=LR_model,X=X_test.iloc[100:,],shap_val=LR_shap,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
    # faithfulness_LR_lime = metrics_reg(model=LR_model,X=X_test.iloc[100:,],shap_val=LR_lime,explainer_type="lime",metrics_type="faithfulness",dataset=dataset)
    faithfulness_RF_shap = metrics_reg(model=RF_model,X=X_test.iloc[100:,],shap_val=RF_shap,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
    # faithfulness_RF_lime = metrics_reg(model=RF_model,X=X_test.iloc[100:,],shap_val=RF_lime,explainer_type="lime",metrics_type="faithfulness",dataset=dataset)
    faithfulness_LR_shap_k = metrics_reg(model=LR_model,X=X_test.iloc[100:,],shap_val=LR_shap_k,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)
    faithfulness_RF_shap_k = metrics_reg(model=RF_model,X=X_test.iloc[100:,],shap_val=RF_shap_k,explainer_type="shap",metrics_type="faithfulness",dataset=dataset)

    monotonicity_LR_shap = metrics_reg(model=LR_model,X=X_test.iloc[100:,],shap_val=LR_shap,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)
    # monotonicity_LR_lime = metrics_reg(model=LR_model,X=X_test.iloc[100:,],shap_val=LR_lime,explainer_type="lime",metrics_type="monotonicity",dataset=dataset)
    monotonicity_RF_shap = metrics_reg(model=RF_model,X=X_test.iloc[100:,],shap_val=RF_shap,explainer_type="shap",metrics_type="monotonicity",dataset=dataset)
    # monotonicity_RF_lime = metrics_reg(model=RF_model,X=X_test.iloc[100:,],shap_val=RF_lime,explainer_type="lime",metrics_type="monotonicity",dataset=dataset)


    print("====================================================================")
    print("faithfulness for SHAP explainer for Linear regression",faithfulness_LR_shap)
    print("====================================================================")
    # print("faithfulness for lime explainer for Linear regression",faithfulness_LR_lime)
    print("====================================================================")
    print("faithfulness for SHAP explainer for Linear regression",faithfulness_RF_shap)
    print("====================================================================")
    # print("faithfulness for lime explainer for Linear regression",faithfulness_RF_lime)
    print("====================================================================")
    print(monotonicity_LR_shap)
    print("====================================================================")
    # print(monotonicity_LR_lime.any())
    print("====================================================================")
    print(monotonicity_RF_shap)
    print("====================================================================")
    # print(monotonicity_RF_lime.any())
    print("====================================================================")
    #monotonicity_RF_lime,faithfulness_LR_lime,faithfulness_RF_lime,monotonicity_LR_lime
    return faithfulness_RF_shap,monotonicity_LR_shap,monotonicity_RF_shap,faithfulness_LR_shap


def Main_cls(dataset):
    if dataset == "wine":
        X,y,target_names,feature_names = loadDataset(dataset)
    else:
        X,y = loadDataset(dataset)
    
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    X_trainS,X_testS,y_trainS,y_testS = train_test_split(X_scaled,y)

    print("--------------------------------------------------------------")
    print("Training Logistic regression, Random Forest classifier and SVM classifier ...")
    LR_model = train_model("Logistic Regression",X_train,y_train)
    RF_model = train_model("Random Forest Classifier",X_train,y_train)
    SVC_model = train_model("SVC",X_trainS,y_trainS)
    print("Done Training")
    print("--------------------------------------------------------------")
    print("Model Evaluation")
    print("Logistic regression")
    print('R2 for Train', RF_model.score( X_train, y_train ))
    print('R2 for Test (cross validation)', RF_model.score(X_test, y_test))
    print("Random forest classifier")
    print('R2 for Train: ', LR_model.score( X_train, y_train ))
    print('R2 for Test (cross validation)', LR_model.score(X_test, y_test))
    print("SVM classifier")
    print('R2 for Train)', SVC_model.score( X_trainS, y_trainS))
    
    predict_fnLR = lambda x:LR_model.predict_proba(x)[:,1]
    predict_fnRF = lambda x:RF_model.predict_proba(x)[:,1]
    predict_fnSVC = SVC_model.decision_function 

    X100 = shap.maskers.Independent(X, max_samples=100)
    X100_ = shap.utils.sample(X, 100)
    
    print("--------------------------------------------------------------")
    print("Building Explanation ...")
    LR_shap,LR_baseVal = Explanation_cls("SHAP",predict_fnLR,X_test[:50,],X100)
    LR_shap_k,LR_expected_val_k = Explanation_cls("Kernel SHAP",predict_fnLR,X_test[:50,],X100_)
    LR_lime1 = Explanation_cls("LIME",LR_model.predict_proba,X_test,X100)
    
    RF_shap, RF_baseVal = Explanation_cls("SHAP",predict_fnRF,X_test[:50,],X100)
    RF_shap_k,RF_expected_val_k = Explanation_cls("Kernel SHAP",predict_fnRF,X_test[:50,],X100_)
    RF_lime1 = Explanation_cls("LIME",RF_model.predict_proba,X_test,X100)
    
    SVC_shap, SVC_baseVal = Explanation_cls("SHAP",predict_fnSVC,X_testS[:50,],X100)
    SVC_shap_k,SVC_expected_val_k = Explanation_cls("Kernel SHAP",predict_fnSVC,X_testS[:50,],X100_)
    SVC_lime1 = Explanation_cls("LIME",SVC_model.predict_proba,X_testS,X100)
    
    print("Done building Explanation")
    ################### evaluation#####################
    
    #faithfulness
    faithfulness_LR_shap= metrics_cls(model=LR_model,X=X_test[:50,],shap_val=LR_shap,explainer_type="shap",metrics_type="faithfulness",dataset="dataset")
    print("Mean Faithfulness for shap Logistic reg:",np.mean(np.array(faithfulness_LR_shap)))
    faithfulness_LR_shap_k = metrics_cls(model=LR_model,X=X_test[:50,],shap_val=LR_shap_k,explainer_type="kernel shap",metrics_type="faithfulness",dataset="dataset")
    print("Mean Faithfulness for kernel shap Logistic reg:",np.mean(np.array(faithfulness_LR_shap_k)))
    faithfulness_LR_lime = metrics_cls(model=LR_model,X=X_test[:50,],shap_val=LR_lime1,explainer_type="lime",metrics_type="faithfulness",dataset="dataset")
    print("Mean Faithfulness for lime Logistic Reg:",np.mean(np.array(faithfulness_LR_lime)))
    faithfulness_RF_shap = metrics_cls(model=RF_model,X=X_test[:50,],shap_val=RF_shap,explainer_type="shap",metrics_type="faithfulness",dataset="dataset")
    print("Mean Faithfulness for shap RF Classification:",np.mean(np.array(faithfulness_RF_shap)))
    faithfulness_RF_shap_k = metrics_cls(model=RF_model,X=X_test[:50,],shap_val=RF_shap_k,explainer_type="kernel shap",metrics_type="faithfulness",dataset="dataset")
    print("Mean Faithfulness for kernel shap RF Classification:",np.mean(np.array(faithfulness_RF_shap_k)))
    faithfulness_RF_lime = metrics_cls(model=LR_model,X=X_test[:50,],shap_val=RF_lime1,explainer_type="lime",metrics_type="faithfulness",dataset="dataset")
    print("Mean Faithfulness for lime RF Classification:",np.mean(np.array(faithfulness_RF_lime)))
    #TODO svc
    
dataset1_cls = "wine"
dataset2_cls = "breast cancer"
Main_cls(dataset1_cls)
Main_cls(dataset2_cls)

dataset1_reg = "boston"
dataset2_reg = "superconductivity"
dataset3_reg = "diabetes"

Main_reg(dataset1_reg)
Main_reg(dataset2_reg)
Main_reg(dataset3_reg)
