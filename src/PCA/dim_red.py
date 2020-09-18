import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

class PCA():
    # different versions of datasets used in different stages
    raw_dataset, adj_dataset, pca_ready_dataset = [], [], []
    cat_cols_idx = []#[1,3,4,5,6,7,8,9] #columns that need to be 1-of-K encoded
    lookup_dict = {}


    def __init__(self, filename):
        self.read_data(filename)
        self.prep_dataset()
        self.sub_mean()
        self.apply_PCA()

    def read_data(self, filename):
        # simply loads the dataset into raw_dataset
        with open(filename) as infile:
            next(infile)
            for line in infile:
                line = line.strip().split(",")
                line = list(map(float, line))
                self.raw_dataset.append(line)
        #self.raw_dataset = np.array(self.raw_dataset)

    def prep_dataset(self):
        # prepare the dataset - a bit convoluted for allowing 1-of-K encoding.
        # ignore for now
        for col in range(len(self.raw_dataset[0])):
            if col in self.cat_cols_idx:
                old_col = [row[col] for row in self.raw_dataset]
                classes = max(old_col)
                new_col = np.zeros((old_col.size, classes))
                for i, value in enumerate(old_col):
                    new_col[i][int(value)-1] = 1

                self.lookup_dict[col] = new_col
            else:
                old_col = [row[col] for row in self.raw_dataset]
                self.lookup_dict[col] = sum(old_col)/len(old_col)

        
        for row in range(len(self.raw_dataset)):
            row_vector = []
            for col in range(len(self.raw_dataset[row])):
                if col in self.cat_cols_idx:
                    row_vector.append(self.lookup_dict[col][row])
                else:
                    row_vector.append(self.raw_dataset[row][col])
                   
            self.adj_dataset.append(row_vector)
        
        self.adj_dataset = list(map(list, zip(*self.adj_dataset)))
        

    def sub_mean(self):
        # for columns that are not 1-of-K encoded we subtract the mean
        for col in range(len(self.adj_dataset)):
            if col not in self.cat_cols_idx:
                self.lookup_dict[col] = [self.lookup_dict[col]]*len(self.adj_dataset)
                column = [row[col] for row in self.adj_dataset]
                new_col = []
                for idx in range(len(column)):
                    new_col.append(column[idx] - self.lookup_dict[col][0])
                self.pca_ready_dataset.append(new_col)
            else:
                column = [row[col] for row in self.adj_dataset]
                self.pca_ready_dataset.append(column)
        

    def apply_PCA(self):
        # this is complaining. Not sure why. I am inputting a np array (1385,29) just like ex2_1_5.py
        # but this returns errors. Need to fix this.
        """lengths = []
        for r in range(self.pca_ready_dataset.shape[0]):
            lengths.append(self.pca_ready_dataset[r].shape[0])
            for c in range(self.pca_ready_dataset[r].shape[0]):
                print()
                lengths.append(self.pca_ready_dataset[r][c].shape)

        print(lengths)"""
        self.pca_ready_dataset = np.array(self.pca_ready_dataset)
        U,S,Vh = svd(self.pca_ready_dataset, full_matrices=True)

        rho = (S*S) / (S*S).sum() 

        threshold = 0.9

        # Plot variance explained
        plt.figure()
        plt.plot(range(1,len(rho)+1),rho,'x-')
        plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
        plt.plot([1,len(rho)],[threshold, threshold],'k--')
        plt.title('Variance explained by principal components');
        plt.xlabel('Principal component');
        plt.ylabel('Variance explained');
        plt.legend(['Individual','Cumulative','Threshold'])
        plt.grid()
        plt.show()

    


if __name__ == "__main__":
    pca = PCA("src/data/HCV-Egy-Data.csv")
    