import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

SHOW_DATA_INFO = False
SHOW_SHAPE = True


# 1. Read data
train = pd.read_csv("covid.train.csv")
test = pd.read_csv("covid.test.csv")
sample = pd.read_csv("sampleSubmission.csv")


# 2. Optional data inspection
if SHOW_DATA_INFO:
    print("train shape:", train.shape)
    print("test shape:", test.shape)
    print("sample shape:", sample.shape)

    print(train.head())
    print(train.columns)

    print("train columns:", train.columns.tolist())
    print("test columns:", test.columns.tolist())
    print("sample columns:", sample.columns.tolist())

    print("train last column:", train.columns[-1])
    print("test last column:", test.columns[-1])

    print(train["tested_positive.2"].describe())


# 3. Split features and target
target = "tested_positive.2"

selected_features = list(train.columns[1:41]) + [
    "tested_positive",
    "tested_positive.1"
]

print("selected feature count:", len(selected_features))
print(selected_features)

X = train[selected_features]
y = train[target]

X_test = test[selected_features]

if SHOW_SHAPE:
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X_test shape:", X_test.shape)


# 4. 切資料
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. 標準化
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)


if SHOW_SHAPE:
    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("y_train:", y_train.shape)
    print("y_valid:", y_valid.shape)



# 5. Try Ridge with different alpha values
for alpha in [0.01, 0.1, 1, 10, 100]:
    model = Ridge(alpha=alpha)

    model.fit(X_train_scaled, y_train)

    valid_pred = model.predict(X_valid_scaled)

    rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

    print("Scaled Ridge alpha =", alpha, "RMSE =", rmse)


# 6. Try RandomForest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_valid_pred = rf_model.predict(X_valid)

rf_rmse = np.sqrt(mean_squared_error(y_valid, rf_valid_pred))

print("RandomForest RMSE =", rf_rmse)

# 7. Train final model with all training data
final_model = Ridge(alpha=10)

final_model.fit(X, y)

test_pred = final_model.predict(X_test)


# 8. Create submission file
submission = sample.copy()

submission["tested_positive"] = test_pred

submission.to_csv("submission.csv", index=False)

print("Saved submission.csv")
print(submission.head())


# avarage
baseline_pred = np.full_like(y_valid, y_train.mean())

baseline_rmse = np.sqrt(mean_squared_error(y_valid, baseline_pred))

print("Mean baseline RMSE =", baseline_rmse)
