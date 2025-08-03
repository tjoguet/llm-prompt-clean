from xgboost import XGBRegressor, callback
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_estimators = 10
pbar = tqdm(total=n_estimators, desc="Training XGBoost")

class TQDMCallback(callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        pbar.update(1)
        return False  # continue training

model = XGBRegressor(n_estimators=n_estimators)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[TQDMCallback()],
)

pbar.close()