import pandas as pd

class reader():
    def get_x_train(self, task):
        if task == 'classification':
            return pd.read_csv("src/data/splits/classification/x_train.csv")
        elif task == 'regression':
            return pd.read_csv("src/data/splits/regression/x_train.csv")
        else:
            raise NameError('Chosen task '+task+' does not exist. Choose between regression and classification.')
    
    def get_x_test(self, task):
        if task == 'classification':
            return pd.read_csv("src/data/splits/classification/x_test.csv")
        elif task == 'regression':
            return pd.read_csv("src/data/splits/regression/x_test.csv")
        else:
            raise NameError('Chosen task '+task+' does not exist. Choose between regression and classification.')

    def get_y_train(self, task):
        if task == 'classification':
            return pd.read_csv("src/data/splits/classification/y_train.csv")
        elif task == 'regression':
            return pd.read_csv("src/data/splits/regression/y_train.csv")
        else:
            raise NameError('Chosen task '+task+' does not exist. Choose between regression and classification.')

    def get_y_test(self, task):
        if task == 'classification':
            return pd.read_csv("src/data/splits/classification/y_test.csv")
        elif task == 'regression':
            return pd.read_csv("src/data/splits/regression/y_test.csv")
        else:
            raise NameError('Chosen task '+task+' does not exist. Choose between regression and classification.')

    def get_all(self, task):
        return (self.get_x_train(task), self.get_y_train(task), self.get_x_test(task), self.get_y_test(task))
