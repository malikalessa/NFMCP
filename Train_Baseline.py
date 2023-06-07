import numpy as np
import Baseline_hyperopt
from keras.models import load_model
from sklearn.model_selection import  StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report

class Train_Baseline():

    def __init__(self, path):
        self.path = path

    def train_baseline(self, x_train, y_train, train_model,save_model):

      if train_model:
          print('Train DNN Model_....')
          model = Baseline_hyperopt.hypersearch(x_train, y_train)
          if save_model:
                  model.save(self.path + 'Model_DNN.h5')

      else:
          model = load_model(self.path + 'Model_DNN.h5')
          print('Loaded DNN Model...')


      return model


    def train_model_based_feature(self, x_train, y_train,x_test,y_test, train_model, save_model,feature_importance,
                                  results_path,explainer,N):

        print(feature_importance)
        if train_model:
            print('Train DNN Model_....')

            for i in range(N,x_train.shape[1],N):
                # To remove the range and add a parameters
                train = x_train[feature_importance[:i]]
                test = x_test[feature_importance[:i]]

                print(train.shape, test.shape)
                model = Baseline_hyperopt.hypersearch(train, y_train)

                Y_predicted = model.predict(test)
                Y_predicted = np.argmax(Y_predicted, axis=1)

                cm = confusion_matrix(y_test, Y_predicted)

                print(cm)
                print("***********Classification Report  set")

                print(classification_report(y_test, Y_predicted))

                with open(results_path +'Model_Accuracy_'+str(i)+explainer+'.txt', "w") as f:
                    print(cm, file=f)
                    print(classification_report(y_test, Y_predicted), file=f)

                if save_model:
                   model.save(self.path + 'Model_DNN_features_'+explainer +str(i)+'.h5')


