import tensorflow as tf
import numpy as np
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent

import pandas as pd

class Create_Adv_Samples():

    def __init__(self, path):
        self.path = path

    ####### Create FastGradientMethod ##########

    def Adv_Samples(self,x_dataset, y_labels, model,eps,feature_importance,class_name,advDataset_path):

        columns = list(x_dataset.columns)

        feature_importance = list(feature_importance.index)
        print(feature_importance)
        mask_array = np.zeros(x_dataset.shape[1])

        for i in (feature_importance):
            mask_array[columns.index(i)] = 1
        print(mask_array)

        classifier = TensorFlowV2Classifier(model, nb_classes= len(np.unique(y_labels)),input_shape=(1,x_dataset.shape[1]),
                                        loss_object = tf.keras.losses.CategoricalCrossentropy())

                ##### Create FGSM Samples #######

        attack = FastGradientMethod(estimator=classifier, eps=eps)
        x_dataset = np.asarray(x_dataset)

        adversarial_samples = attack.generate(x=x_dataset, mask = mask_array)


        return  adversarial_samples,y_labels

