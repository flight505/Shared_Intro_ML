import pandas as pd
from numpy import genfromtxt

class reader():

    def get_x_train(self, task):
        return genfromtxt("src/data/splits/"+task+"/x_train.csv", delimiter=',')
    
    def get_x_test(self, task):
        return genfromtxt("src/data/splits/"+task+"/x_test.csv", delimiter=',')

    def get_y_train(self, task):
        return genfromtxt("src/data/splits/"+task+"/y_train.csv", delimiter=',')

    def get_y_test(self, task):
        return genfromtxt("src/data/splits/"+task+"/y_test.csv", delimiter=',')

    def get_all(self, task):
        return (self.get_x_train(task), self.get_y_train(task), self.get_x_test(task), self.get_y_test(task))
