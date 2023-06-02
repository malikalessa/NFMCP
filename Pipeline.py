import Create_Adv_Samples as adv
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import metrics as mt


class Pipeline():

    def __init__(self,path_models):
        self.path_models = path_models


    def FGSM(self,TestC,y_TestC, advDataset_path,eps,model, feature_importance, label,N, explanation_type,datatype):

        # This Function to creat FGSM adversarial Samples
        execution = adv.Create_Adv_Samples(path = advDataset_path)

        for i in eps :
            for j in range(N,len(feature_importance),N):
                adv_dataset,y_Advlabels = execution.Adv_Samples(TestC,y_TestC,model,i,feature_importance[:j],label,
                                                        advDataset_path)
                file_name = datatype + '_eps_'+str(i)+'_'+explanation_type +'_step_'+str(j)
                mt.metrics(model,adv_dataset,y_Advlabels, file_name,self.path_models)

            adv_dataset, y_Advlabels = execution.Adv_Samples(TestC, y_TestC, model, i, feature_importance, label,
                                                             advDataset_path)
            file_name = explanation_type+ datatype +'_eps_' + str(i) + '_ALL'
            mt.metrics(model, adv_dataset, y_Advlabels, file_name, self.path_models)




    def mutual_info(self,x_train,y_train,x_test,y_test,model, results_path,dataset_type,N):

        if dataset_type == 'Train':
            feature_names = list(x_train.columns.values)
            filter_KBest_MI = SelectKBest(mutual_info_classif, k=x_train.shape[1])
            SelectedX = filter_KBest_MI.fit_transform(x_train, y_train)
            MI_Features = x_train.columns[filter_KBest_MI.get_support()]
            MI_Features = pd.DataFrame(MI_Features)
            MI_Features['Score'] = filter_KBest_MI.scores_
            MI_Features = MI_Features.sort_values(by = ['Score'], ascending = False)
            MI_Features = MI_Features.rename(columns = {0: 'Features'})
            print(MI_Features)

            MI_Features.to_csv(path_or_buf = results_path + dataset_type +'_MI_Ranking.csv', index = False)

            return MI_Features


        elif  dataset_type == 'Test':
            feature_names = list(x_train.columns.values)
            filter_KBest_MI = SelectKBest(mutual_info_classif, k=x_test.shape[1])
            SelectedX = filter_KBest_MI.fit_transform(x_test, y_test)
            MI_Features = x_test.columns[filter_KBest_MI.get_support()]
            MI_Features = pd.DataFrame(MI_Features)
            MI_Features['Score'] = filter_KBest_MI.scores_
            MI_Features = MI_Features.sort_values(by = ['Score'], ascending = False)
            print(MI_Features)
            MI_Features.to_csv(path_or_buf=results_path + dataset_type + '_MI_Ranking.csv', index=False)

        elif  dataset_type == 'TestC':

            new_test = []
            new_Ytest = []
            Y_predicted = model.predict(x_test)
            Y_predicted = np.argmax(Y_predicted, axis=1)

            for i in range(Y_predicted.shape[0]):
                if Y_predicted[i] == y_test[i]:
                    new_test.append(x_test.iloc[i])
                    new_Ytest.append(y_test[i])
            new_test = pd.DataFrame(new_test)
            new_Ytest = pd.DataFrame(new_Ytest, columns=['Class'])

            feature_names = list(x_train.columns.values)
            filter_KBest_MI = SelectKBest(mutual_info_classif, k=x_test.shape[1])
            SelectedX = filter_KBest_MI.fit_transform(new_test, new_Ytest)
            MI_Features = x_test.columns[filter_KBest_MI.get_support()]
            MI_Features = pd.DataFrame(MI_Features)
            MI_Features['Score'] = filter_KBest_MI.scores_
            MI_Features = MI_Features.sort_values(by = ['Score'], ascending = False)
            MI_Features.rename(columns = {0:'Feature'}, inplace = True)
            MI_rank = np.zeros((1, x_train.shape[1]))

            columns = list(MI_Features['Feature'])
            MI_rank = pd.DataFrame(MI_rank, columns=columns)
            MI_rank.iloc[0] = MI_Features['Score']
            MI_rank = MI_rank.T
            print(MI_rank)

            return MI_rank, new_test, new_Ytest


