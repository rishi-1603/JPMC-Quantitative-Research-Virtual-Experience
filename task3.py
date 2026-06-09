import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

target_col = "default"

if target_col not in df.columns:
    target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

X = X.select_dtypes(include=["number"])

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

RECOVERY_RATE = 0.10

def predict_probability(loan_features):
    loan_df = pd.DataFrame([loan_features])
    loan_df = loan_df.reindex(columns=X.columns, fill_value=0)
    return float(model.predict_proba(loan_df)[0][1])

def expected_loss(loan_features, loan_amount):
    pd_value = predict_probability(loan_features)
    lgd = 1 - RECOVERY_RATE
    return pd_value * lgd * loan_amount

sample_borrower = X.iloc[0].to_dict()

pd_estimate = predict_probability(sample_borrower)
loss_estimate = expected_loss(sample_borrower, 100000)

print("\nProbability of Default:", round(pd_estimate, 4))
print("Expected Loss on $100,000 Loan:", round(loss_estimate, 2))