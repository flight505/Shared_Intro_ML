import pandas as pd

class reader():
    class_col = 'Baselinehistological_staging'
    regr_col = 'Baselinehistological_grading'

    def get_col(self, task):
        if task == 'regression':
            return 'Baselinehistological_grading'
        elif task == 'classification':
            return 'Baselinehistological_staging'
        else:
            raise NameError('Chosen task '+task+' does not exist. Choose between regression and classification.')

    def get_x_train(self, task):
        return pd.read_csv("src/data/splits/"+task+"/x_train.csv")
    
    def get_x_test(self, task):
        return pd.read_csv("src/data/splits/"+task+"/x_test.csv")

    def get_y_train(self, task):
        return pd.read_csv("src/data/splits/"+task+"/y_train.csv")[self.get_col(task)]

    def get_y_test(self, task):
        return pd.read_csv("src/data/splits/"+task+"/y_test.csv")[self.get_col(task)]

    def get_all(self, task):
        return (self.get_x_train(task), self.get_y_train(task), self.get_x_test(task), self.get_y_test(task))
