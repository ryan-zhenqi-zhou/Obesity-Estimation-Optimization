# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:56:53 2022

@author: ryanz
"""

# Packages
import pandas as pd
import numpy as np
import geopandas as gpd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statsmodels.api as sm

from scipy.spatial import distance

# Geographical RandomForest
class GeographicalRandomForest:
    # this is the initialization function
    # param local_model_num controls how many local models will participate in prediction; default is 1 !!!New
    def __init__(self, ntree, mtry, band_width, local_weight, local_model_num=1, bootstrap=False, random_seed=42):
        self.ntree = ntree
        self.mtry = mtry
        self.band_width = band_width
        self.local_weight = local_weight
        self.local_model_num = local_model_num
        self.bootstrap=bootstrap
        self.random_seed = random_seed
        self.global_model = None
        self.local_models = None
        self.train_data_coords = None
        self.distance_matrix = None
        self.train_data_index = None
        self.train_data_columns = None
       
    
    # param X_train contains a data frame of the the training indepdent variables 
    # param y_train contains a data series of the target dependent variable
    # param coords contains a data frame of the two-dimensional coordinates
    # param record_index contains a data series of the indices of the data for helping store local models
    def fit(self, X_train, y_train, coords, record_index):
        
        # save the index of the training data
        self.train_data_index = record_index
        self.train_data_columns = X_train.columns
        
        # get Global RF model and importance information, and save global RF model
        rf_global = RandomForestRegressor(bootstrap = self.bootstrap, n_estimators = self.ntree, max_features = self.mtry, random_state = self.random_seed) 
        rf_global.fit(X_train, y_train)
        self.global_model = rf_global
        
        
        # create an empty dictionary for local models
        self.local_models = {}
        
        # get the distance matrix between the training geographic features
        coords_array = np.array(coords, dtype = np.float64) # translate (x,y) to array type
        self.train_data_coords = coords_array
        self.distance_matrix = distance.cdist(coords_array,coords_array, 'euclidean') # calculate Euclidean Distance
        
        # train local models
        for i in range(len(X_train)):
            distance_array = self.distance_matrix[i]
            idx = np.argpartition(distance_array, self.band_width)  # Get the index of the geographic features that are the nearest to the target geographic feature
            idx = idx[:self.band_width]  # only those indices within the band_width are valid 
            
            local_X_train = X_train.iloc[idx]
            local_y_train = y_train.iloc[idx]
            
            # make local tree size smaller, because there is no sufficient data to train a big tree !!!New
            local_tree_size = int(self.ntree * (self.band_width*1.0/len(X_train)))
            if local_tree_size < 1:
                local_tree_size = 1  # local tree size should be at least 1
             
            # get local model
            rf_local = RandomForestRegressor(bootstrap = self.bootstrap, n_estimators = local_tree_size, max_features = self.mtry, random_state = self.random_seed) # input
            rf_local.fit(local_X_train, local_y_train.values.ravel())
            
            # key for storing local rf model in a dictionary
            rf_local_key = str(record_index.iloc[i])+"|"+ str(int(coords_array[i][0]))+"|"+str(int(coords_array[i][1]))
            self.local_models[rf_local_key] = rf_local
            
    
    # the function for making predictions using the GRF model
    # param X_test contains a data frame of the independent variables in the test dataset
    # param coords contains a data frame of the two-dimensional coordinates
    def predict(self, X_test, coords_test): 
        
        # first, make prediction using the global RF model 
        predict_global = self.global_model.predict(X_test).flatten() # get the global predict y first
        
        # Second, make prediction using the local RF model 
        coords_test_array = np.array(coords_test, dtype = np.float64)
        distance_matrix_test_to_train = distance.cdist(coords_test_array, self.train_data_coords, 'euclidean')
        predict_local = []
        
        for i in range(len(X_test)):
            distance_array = distance_matrix_test_to_train[i]
            idx = np.argpartition(distance_array, self.local_model_num)  # Get the index of the geographic features that are the nearest to the target geographic feature
            idx = idx[:self.local_model_num]
            
            this_local_prediction = 0
            for this_idx in idx:
                local_model_key = str(self.train_data_index.iloc[this_idx])+"|"+ str(int(self.train_data_coords[this_idx][0]))+"|"+str(int(self.train_data_coords[this_idx][1]))
                local_model = self.local_models[local_model_key]
                this_local_prediction += local_model.predict(X_test[i:i+1]).flatten()[0]
            
            this_local_prediction = this_local_prediction*1.0 / self.local_model_num  # average local predictions
            predict_local.append(this_local_prediction)
          
        
        # Third, combine global and local predictions
        predict_combined = []
        for i in range(len(predict_global)):
            this_combined_prediction = predict_local[i]*self.local_weight + predict_global[i]*(1-self.local_weight) 
            predict_combined.append(this_combined_prediction)
        
        
        return predict_combined, predict_global, predict_local   # return three types of predictions
    
    
    # this function outputs the local feature importance based on the local models
    def get_local_feature_importance(self):
        if self.local_models == None:
            print("The model has not been trained yet...")
            return None
        
        column_list = [self.train_data_index.name] 
        for column_name in self.train_data_columns: 
            column_list.append(column_name) 
            
        feature_importance_df = pd.DataFrame(columns = column_list) 
        
        for model_key in self.local_models.keys():
            model_info = model_key.split("|")
            this_local_model = self.local_models[model_key]
            this_row = {}
            this_row[self.train_data_index.name] = model_info[0] # the index of a row
            for feature_index in range(0, len(self.train_data_columns)):
                this_row[self.train_data_columns[feature_index]]=this_local_model.feature_importances_[feature_index]
            
            feature_importance_df = feature_importance_df.append(this_row, ignore_index = True) # TypeError: Can only append a dict if ignore_index=True
            
            
        return feature_importance_df
    
# main 
if __name__ == '__main__':
    # Read the files
    X_socio_2 = pd.read_csv("../Data/06 Data for Optimal GRF Search/01 NYC/X_sociodemo_train.csv", index_col='GEOID') # input
    y_2 = X_socio_2.pop('obesity_cr')
    
    local_weight_list = []
    local_model_num_list = []
    r_square_list = []
    rmse_list = []
    search_result = pd.DataFrame(columns = ['local_weight','local_model_num','r2','rmse']) 
    
    local_weight = [0.25, 0.5, 0.75, 1]
    
    print('All numbers of try: 160')
    number = 1
    
    def standarize_data(data, stats):
        return (data - stats['mean'])/ stats['std']
    
    for local_w in local_weight:
        for local_model_n in range(1,41,1):
            y_rf_socio_predict = []
            y_true = []
            
            ten_fold = KFold(n_splits=10, shuffle=True, random_state=42)
            
            for train_index, test_index in ten_fold.split(X_socio_2):
                # print("TEST:", test_index)
            
                X_train_1, X_test_1 = X_socio_2.iloc[train_index], X_socio_2.iloc[test_index]
                y_train, y_test = y_2.iloc[train_index], y_2.iloc[test_index]
                X_train = X_train_1[['% Black','% Ame Indi and AK Native','% Asian','% Nati Hawa and Paci Island','% Hispanic or Latino','% male','% married','% age 18-29','% age 30-39','% age 40-49','% age 50-59','% age >=60','% <highschool','median income','% unemployment','% below poverty line','% food stamp/SNAP','median value units built','median year units built','% renter-occupied housing units','population density']]
                X_test = X_test_1[['% Black','% Ame Indi and AK Native','% Asian','% Nati Hawa and Paci Island','% Hispanic or Latino','% male','% married','% age 18-29','% age 30-39','% age 40-49','% age 50-59','% age >=60','% <highschool','median income','% unemployment','% below poverty line','% food stamp/SNAP','median value units built','median year units built','% renter-occupied housing units','population density']]
                xy_coord = X_train_1[["Lonpro","Latpro"]]
                train_index_1 = X_train.index
                train_index = pd.Series(train_index_1)
                coords_test = X_test_1[["Lonpro","Latpro"]]
                
                training_stat = X_train.describe().transpose()
                scaled_X_train = standarize_data(X_train, training_stat)
                scaled_X_test = standarize_data(X_test, training_stat)
                
                grf = GeographicalRandomForest(560, 7, 220, local_w, local_model_n) 
                grf.fit(scaled_X_train, y_train, xy_coord, train_index)
                
                predict_combined, predict_global, predict_local = grf.predict(scaled_X_test,coords_test)
                y_rf_socio_predict = y_rf_socio_predict + predict_combined
                y_true = y_true + y_test.tolist()
            
            rf_socio_rmse = mean_squared_error(y_true , y_rf_socio_predict, squared=False) #False means return RMSE value
            rf_socio_r2 = r2_score(y_true, y_rf_socio_predict)
            
            local_weight_list.append(local_w)
            local_model_num_list.append(local_model_n)
            r_square_list.append(rf_socio_r2)
            rmse_list.append(rf_socio_rmse)
            print('now： ' + str(number))
            number = number + 1
    
    search_result['local_weight'] = local_weight_list
    search_result['local_model_num'] = local_model_num_list
    search_result['r2'] = r_square_list
    search_result['rmse'] = rmse_list
    search_result.to_csv("../Data/06 Data for Optimal GRF Search/01 NYC/search_result.csv") # input
    
    
    