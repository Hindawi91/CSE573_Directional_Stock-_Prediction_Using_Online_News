import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

dataset = "Amzn"
method = "tfidf"

df = pd.read_pickle(f"../2_features_extraction/{dataset}_{method}_features.pkl")

X = df.iloc[: , 6:]
y = df.iloc[:, 4]

# Assuming you have your data in variables x and y
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize models
logreg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(random_state=42)
adaboost = AdaBoostClassifier(random_state=42)
svc = SVC(probability=True, random_state=42)
knn = KNeighborsClassifier()
voting_clf = VotingClassifier(estimators=[('logreg', logreg), ('rf', rf), ('adaboost', adaboost), ('svc', svc), ('knn', knn)], voting='soft')

# Train models
models = {'Logistic Regression': logreg, 'Random Forest': rf, 'AdaBoost': adaboost, 'SVC': svc, 'KNN': knn, 'Voting Ensemble': voting_clf}

results = pd.DataFrame(columns=['Model', 'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'])

confusion_matrices = {}
all_predictions = pd.DataFrame({'True Label': y_test}) 

for name, model in models.items():
    # Train the model
    model.fit(x_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(x_test_scaled)
    
    # Save predictions to a DataFrame
    all_predictions[f'Predicted Label ({name})'] = y_pred
    
    # Save metrics to results DataFrame
    results.loc[len(results)] = [name,
                                  accuracy_score(y_test, y_pred),
                                  balanced_accuracy_score(y_test, y_pred),
                                  precision_score(y_test, y_pred),
                                  recall_score(y_test, y_pred),
                                  f1_score(y_test, y_pred),
                                  roc_auc_score(y_test, y_pred)]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    
# Save results to CSV
results.to_csv(f'../results/{dataset}_{method}_model_results.csv', index=False)
print(results)

# Plot confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(16, 12))

for i, (name, cm) in enumerate(confusion_matrices.items()):
    row = i // 3
    col = i % 3
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
    axes[row, col].set_title(name)

plt.tight_layout()
plt.savefig(f'../results/{dataset}_{method}_CM.png')
plt.show()

# Save all predictions to a single CSV file
all_predictions.to_csv(f'../results/{dataset}_{method}_all_predictions.csv', index=False)