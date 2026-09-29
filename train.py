from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    avocado_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    avocado_model.fit(X_train, y_train)
    prediction = avocado_model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    return(mae)

avocado_filename = "avocado.csv"
avocado_dataset = pd.read_csv(avocado_filename)

avocado_dataset = avocado_dataset.dropna(axis=0) # drops N/A values

# print(avocado_dataset.describe())
# print(avocado_dataset.columns)

# target
y = avocado_dataset.AveragePrice
# features
features = ["Total Volume", "Total Bags", "Small Bags", "Large Bags", "XLarge Bags", "year"]
X = avocado_dataset[features]


# make model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# compare MAE with differing values of max_leaf_nodes
best_mae = 0
best_leaf_nodes = 0
leaf_nodes_canidates = [5, 50, 500, 5000]
for max_leaf_nodes in leaf_nodes_canidates: # 500 is most accurate
    mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    if((leaf_nodes_canidates[0] == max_leaf_nodes) or (best_mae > mae)):
        best_mae = mae
        best_leaf_nodes = max_leaf_nodes


# # DecisionTreeRegressor
# print("DecisionTreeRegressor")
# final_avocado_model = DecisionTreeRegressor(max_leaf_nodes = best_leaf_nodes, random_state=1)
# final_avocado_model.fit(X,y)

# RandomForestRegressor (SO MUCH MORE ACCURATE!!!)
print("RandomForestRegressor")
final_avocado_model = RandomForestRegressor(random_state=1)
final_avocado_model.fit(X,y)

prediction = final_avocado_model.predict(X_test)
print(prediction)
print(y_test)
mae = mean_absolute_error(y_test, prediction)
print(mae)





