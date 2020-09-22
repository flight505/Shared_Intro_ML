import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

class PCA():
    """
        At this stage we initialize the variables needed in the various methods below.
    """
    raw_dataset, pca_ready_dataset = [], []
    targets = []
    # this array contains the index of the columns that are classes and need 1-of-K encoding
    cat_cols_idx = [1,3,4,5,6,7,8,9]
    lookup_dict_mean, lookup_dict_std = {}, {} # dict used for standardization


    def __init__(self, filename):
        """Initialization method - acts as a controller

        Args:
            filename (string): path to the file to process
        """
        self.read_data(filename)
        self.prep_dataset()
        self.sub_mean()
        self.apply_PCA()

    def read_data(self, filename):
        """ Loads in the specified file. Creates features vector and targets vector

        Args:
            filename (string): path to the file to process
        """
        with open(filename) as infile:
            next(infile)
            for line in infile:
                line = line.strip().split(",")
                features = line[:-2]
                features = list(map(float, features))
                self.targets.append(line[-1])
                self.raw_dataset.append(features)


    def prep_dataset(self):
        """
            We iterate through the columns and do different things depending on
            whether the column is 1-of-K or not.
            old_col contains all the measurements for that particular feature.
            new_col is the 1-of-K encoded array.
            if the column we are looking at is not a column to 1-of-K encode,
            we also compute mean and std for later standardization.
        """
        for col in range(len(self.raw_dataset[0])):
            if col in self.cat_cols_idx:
                old_col = [row[col] for row in self.raw_dataset]
                classes = int(max(old_col))
                new_col = np.zeros((len(old_col), classes))
                for i, value in enumerate(old_col):
                    new_col[i][int(value)-1] = 1

            else:
                old_col = [row[col] for row in self.raw_dataset]
                self.lookup_dict_mean[col] = sum(old_col)/len(old_col)
                self.lookup_dict_std[col] = np.std(old_col)


    def sub_mean(self):
        """
            We iterate through the columns once more in order to create an array that is ready for svd.
            Again, we need to perform to distinct operations depending on weather the column analyzed
            is 1-of-K encoded or not.
            If it is, we simply append the 1-of-K encoded vector.
            Otherswise we standardize by dubtracting the mean and dividing by std - previously calculated -
            and append the result
            Finally we transpose the array because the way we have have done it, created a column vector. We
            want a row vector instead.
        """
        for col in range(len(self.raw_dataset[0])):
            if col not in self.cat_cols_idx:
                column = [row[col] for row in self.raw_dataset]
                new_col = []
                for idx in range(len(column)):
                    new_col.append((column[idx] - self.lookup_dict_mean[col]) / self.lookup_dict_std[col])
                self.pca_ready_dataset.append(new_col)
            else:
                column = [row[col] for row in self.raw_dataset]
                self.pca_ready_dataset.append(column)

        self.pca_ready_dataset = list(map(list, zip(*self.pca_ready_dataset)))

        

    def apply_PCA(self):
        """
            Shamelessly copied from ex2_1_3.py
        """
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
    