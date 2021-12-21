import numpy as np
import datetime
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import feature_generator
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#  Creates a support vector machine to predict sap flux from environmental variables

class SupportVectorMachines:
    def __init__(self):
        self.X = np.empty([1, 1])
        self.Y = np.empty([1, 1])

    def data_import(self, features, files):
        # Select and import data
        feature_generator(features, files)
        data = np.loadtxt('data/modeling_data/working_data.csv', skiprows=1, delimiter=',')
        self.X = data[:, 0:-1]
        self.Y = data[:, -1]
        print(f'The number of data points is {len(self.X)}.')
        return self.X, self.Y

    def build_models(self):
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        #  Split to training/validation sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, self.Y, test_size=0.2, random_state=42)
        self.X_test, self.X_val, self.Y_test, self.Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
        self.svr = SVR().fit(X_train, Y_train)

    def score_models(self):
        Y_pred = self.svr.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, Y_pred)
        r2 = self.svr.score(self.X_test, self.Y_test)
        print(f'MSE was {mse} and R2 was {r2}')

if __name__ == "__main__":
    begin_time = datetime.datetime.now()
    svm_creator = SupportVectorMachines()

    #  Define data categories
    creator = ClusterCreator()
    creator.k_means_clusters()
    creator.func_type_clusters()
    creator.biome_clusters()
    k_clusters = creator.k_cluster_dict
    func_clusters = creator.func_cluster_dict
    biome_clusters = creator.biome_cluster_dict

    # Define which features and files to use
    my_features = ['ta', 'rh', 'vpd', 'ppfd_in', 'swc_shallow', 'precip']  # Specify desired features
    my_files = biome_clusters['Tropical rain forest']  # Can choose by biome, functional type, K-Means, or include all using []

    # Import data, build model and score
    svm_creator.data_import(my_features, my_files)
    print(f'The runtime for importing data was {datetime.datetime.now() - begin_time}.')
    begin_time = datetime.datetime.now()
    svm_creator.build_models()
    print(f'The runtime to fit the ML model was {datetime.datetime.now() - begin_time}.')
    begin_time = datetime.datetime.now()
    svm_creator.score_models()
    print(f'The runtime to score the ML model was {datetime.datetime.now() - begin_time}.')
