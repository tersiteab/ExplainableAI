from explanations.explainer import Explanation_reg
from target_models.model import loadDataset,train_model,train_test_split
from evaluation.metrics import metrics_cls,metrics_reg
import shap
from sklearn.preprocessing import StandardScaler

def Main_ref(dataset):
    X,y = loadDataset(dataset)
    sc = StandardScaler()
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    X_scaled = sc.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    X_trainS,X_testS,y_trainS,y_testS = train_test_split(X_scaled,y)

    LR_model = train_model("Linear Regression",X_train,y_train)
    RF_model = train_model("Random Forest Regressor",X_train,y_train)
    SVR_model = train_model("SVR",X_trainS,y_trainS)

    print("========================================================")
    print("Linear Regression")
    print('R2 for Train)', LR_model.score( X_train, y_train ))
    print('R2 for Test (cross validation)', LR_model.score(X_test, y_test))
    print("========================================================")
    print("Random Forest Regression")
    print('R2 for Train)', RF_model.score( X_train, y_train ))
    print('R2 for Test (cross validation)', RF_model.score(X_test, y_test))
    print("========================================================")
    print("Random Forest Regression")
    print('R2 for Train)', SVR_model.score( X_trainS, y_trainS))
    # print('R2 for Test (cross validation)', r2_score(y_testS, sc.inverse_transform(SVC_model.predict(X_testS))))

    X100 = shap.maskers.Independent(X, max_samples=100)
    X100_ = shap.utils.sample(X, 100)
    X100S = shap.maskers.Independent(X_trainS, max_samples=100)
    X100_S = shap.utils.sample(X_trainS, 100)
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