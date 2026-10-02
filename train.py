from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
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
# print("\n==== DecisionTreeRegressor ====")
# final_avocado_model = DecisionTreeRegressor(max_leaf_nodes = best_leaf_nodes, random_state=1)
# final_avocado_model.fit(X_train,y_train)

# RandomForestRegressor
print("\n==== RandomForestRegressor ====")
final_avocado_model = RandomForestRegressor(random_state=1)
final_avocado_model.fit(X_train,y_train)


prediction = final_avocado_model.predict(X_test)
print("\n==== Predictions ====")
print(prediction)
print("\n==== Actual Results ====")
print(y_test)
mae = mean_absolute_error(y_test, prediction)

print("\n==== Metrics ====")
print(f"The mean absolute error of the prediction is ${mae:.4f}/lb")

# desision tree model metrics
# mae = $0.1897/lb

# random forest regressor model metrics (MOST ACCURATE!)
# mae = $0.1613/lb
