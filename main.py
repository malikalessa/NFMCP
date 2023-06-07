
import os
import numpy as np
import readDataset as read_dataset
import configparser
import sys
import pandas as pd
import Train_Baseline as train_base
import Pipeline as pipeline
import metrics
import Dalex as dx

np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def main():

    dataset = sys.argv[1]
    config = configparser.ConfigParser()
    config.read('Conf.conf')
    dsConf = config[dataset]
    configuration = config['setting']
    pd.set_option('display.expand_frame_repr', False)
    dataset = read_dataset.readDataset(dsConf.get('pathDataset'))
    x_train, y_train = dataset.read(dsConf.get('nameTrain'), dsConf.get('label'))
    print('X_train Shape : ', x_train.shape)
    x_test, y_test = dataset.read(dsConf.get('nameTest'), dsConf.get('label'))
    print('X_Test Shape : ', x_test.shape)
    dataset_columns = x_train.columns

    # The step to be selected
    N = int(configuration.get('N'))

    ### Train the Baseline Model
    execution = train_base.Train_Baseline(path=dsConf.get('pathModels'))
    model = execution.train_baseline(x_train, y_train, train_model=int(configuration.get('Train_Baseline')),

                                     save_model=configuration.get('save_model'))
    metrics.metrics(model,x_test,y_test,'Accuracy_Score_Test_DNN_',dsConf.get('results_path'))


    ###################  Defensive Part ##############################

    if (int(configuration.get('defensive'))):
        # Dalex

        dalex_values = dx.DalexDatasets(dsConf.get('results_path'))
        if (int(configuration.get('Dalex'))):
            dalex_values.createDalex_ranking(x_train, y_train, model, 'Train')

            dalex_ranking = dalex_values.loadDalexDatasets('Train')
        else:
            dalex_ranking = dalex_values.loadDalexDatasets('Train')
        
        # change the name to " Dalex_feature_selection
        if (int(configuration.get('Rank_Dalex_features_Train'))):
            dalex_ranking_features =  list(dalex_ranking['variable'])

            model = execution.train_model_based_feature(x_train, y_train, x_test, y_test,
                                                    int(configuration.get('Train_model_features')),
                                                    configuration.get('save_model'), dalex_ranking_features,
                                                    dsConf.get('results_path'), 'dalex')


        ##### Mutual Info

        mutual_info = pipeline.Pipeline(dsConf.get('pathModels'))
        MI = mutual_info.mutual_info(x_train, y_train, x_test, y_test, model, dsConf.get('results_path'),
                                     'Train', N)

        # change the name to " MI_feature_selection

        if (int(configuration.get('MI_Train'))):
            MI = list(MI['Features'])

            model = execution.train_model_based_feature(x_train, y_train, x_test, y_test,
                                                        int(configuration.get('Train_model_features')),
                                                        configuration.get('save_model'), MI,
                                                        dsConf.get('results_path'), 'MI')

       


    elif(int(configurations.get('defensive')== 0) :
        ########################## Offensive Part    ############################

        ##Compute feature importance based on the correctly classified test samples

        #### This Part is based ion the Testing Dataset
        
        
         #### Add if statement to choose between dalex and MI
        eps = [0.001,0.01]
        execution_pipeline = pipeline.Pipeline(path_models=dsConf.get('pathModels'))

        dalex_values = dx.DalexDatasets(dsConf.get('results_path'))

        feature_importance_Dalex, TestC_Dalex, y_TestC_Dalex = dalex_values.Dalex_Ranking_C_offensive(x_test, y_test,
         model)

        execution_pipeline.FGSM(TestC_Dalex, y_TestC_Dalex, dsConf.get('AdvDataset_path'), eps, model,
                                feature_importance_Dalex,
                                dsConf.get('Label'), N,'Dalex', 'TestC')
        # Add a comment to explain TestC
        feature_importance_MI, TestC_MI, y_TestC_MI = execution_pipeline.mutual_info(x_train,y_train,
                                                        x_test, y_test, model, dsConf.get('results_path'),'TestC',N)
        execution_pipeline.FGSM(TestC_MI, y_TestC_MI, dsConf.get('AdvDataset_path'), eps, model,
                                feature_importance_MI,
                                dsConf.get('Label'), N, 'MI', 'TestC')



if __name__ == "__main__":
    main()
