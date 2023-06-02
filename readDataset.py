import pandas as pd


class readDataset():

    def __init__(self, path):
        # Default parameters for the datasets included in the implementation
        self.path = path

    def read(self, name, label):
        ''''
        Dataset_names already embeded are:
        1- Maldroid20
        2- CICIDS17
        3- NSL-KDD
        '''

        dataset= pd.read_csv(self.path + name )

        y_label = dataset[label]
        try:
                dataset.drop([label], axis=1, inplace=True)

        except IOError:
                print(IOError)


        return dataset, y_label


