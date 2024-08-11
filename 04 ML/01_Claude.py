import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler


# Load the data
df = pd.read_csv(r'03 Clean Data\windows_rules_filter.csv')

# Extract the features and labels
X = df['Detection']
y = df['Tags']

# Convert the string-based detection rules to a numerical feature matrix
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Handle class imbalance using oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_vectorized, y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define the feature selection pipeline
selector = SelectKBest(chi2, k=100)
pipeline = Pipeline([('selector', selector)])

# Fit the pipeline on the training data and transform the data
X_train_transformed = pipeline.fit(X_train, y_train).transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Try different models and perform cross-validation
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Random Forest', RandomForestClassifier())
]

for name, model in models:
    print(f'Evaluating {name}:')
    
    # Fit the model and evaluate on the test set
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)
    
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1-score: {f1:.3f}')
    print()