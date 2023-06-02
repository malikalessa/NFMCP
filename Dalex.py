import pandas as pd
import dalex as dx


import numpy as np


class DalexDatasets():
    def __init__(self, path_results):
        self.path_results = path_results



    def createDalex_ranking(self,dataset,y_labels, model, dataset_type):
        # To compute the feature relevance in the models
        print('Dalex is computed using : ', dataset_type)
        explainer = dx.Explainer(model, dataset, y_labels)
        explanation = explainer.model_parts(random_state=42)
        variable_importance = pd.DataFrame(explanation.result)

        variable_importance = variable_importance.sort_values(by=['dropout_loss'], ascending=False)

        variable_importance.drop(['label'], axis=1, inplace=True)

        variable_importance = variable_importance[variable_importance.variable != '_full_model_']
        variable_importance = variable_importance[variable_importance.variable != '_baseline_']


        print("Dalex File Shape  : ", variable_importance.shape)

        variable_importance.to_csv(path_or_buf=self.path_results + dataset_type+'_Dalex_ranking.csv', index=False)


    def loadDalexDatasets(self, dataset_type):
        path =  self.path_results + dataset_type +'_Dalex_ranking.csv'
        df = pd.read_csv(path)
        print('The Dalex File has been uploaded')
        return df

    def Dalex_Ranking_C_offensive(self,x_test,y_test,model):

        new_test = []
        new_Ytest = []
        Y_predicted = model.predict(x_test)
        Y_predicted = np.argmax(Y_predicted, axis=1)

        for i in range(Y_predicted.shape[0]):
            if Y_predicted[i] == y_test[i]:
                new_test.append(x_test.iloc[i])
                new_Ytest.append(y_test[i])

        new_test = pd.DataFrame(new_test)
        new_Ytest = pd.DataFrame(new_Ytest)
        print('TestC shap : ', new_test.shape)
        explainer = dx.Explainer(model, new_test, new_Ytest)
        explanation = explainer.model_parts(random_state=42)
        variable_importance = pd.DataFrame(explanation.result)

        variable_importance = variable_importance.sort_values(by=['dropout_loss'], ascending=False)

        variable_importance.drop(['label'], axis=1, inplace=True)

        variable_importance = variable_importance[variable_importance.variable != '_full_model_']
        variable_importance = variable_importance[variable_importance.variable != '_baseline_']

        print("Dalex File Shape for TestC  : ", variable_importance.shape)

        variable_importance.to_csv(path_or_buf=self.path_results  + 'Offensive_TestC_Dalex_ranking.csv', index=False)
        da_rank = np.zeros((1, x_test.shape[1]))
        columns = list(variable_importance['variable'])
        da_rank = pd.DataFrame(da_rank, columns=columns)
        da_rank.iloc[0] = variable_importance['dropout_loss']
        da_rank = da_rank.T

        print(da_rank.shape)
        return da_rank, new_test, new_Ytest