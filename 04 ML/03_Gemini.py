import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from ast import literal_eval
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

# Read the CSV file into a DataFrame
df = pd.read_csv('windows_rules_filter.csv')

# Convert `Tags` column to string
df['Tags'] = df['Tags'].astype(str)

# Convert string representation of dictionaries to actual dictionaries
df['Detection'] = df['Detection'].apply(literal_eval)

# Extract values from the dictionaries and create new columns
df['Keywords'] = df['Detection'].apply(lambda x: x.get('keywords', ''))
df['Filter'] = df['Detection'].apply(lambda x: x.get('filter', ''))
df['Selection'] = df['Detection'].apply(lambda x: x.get('selection', ''))
df['Condition'] = df['Detection'].apply(lambda x: x.get('condition', ''))

# Fill NaN values with empty strings
df['Keywords'].fillna('', inplace=True)
df['Filter'].fillna('', inplace=True)
df['Selection'].fillna('', inplace=True)
df['Condition'].fillna('', inplace=True)

# Combine text from multiple columns
df['Combined_Text'] = df['Keywords'].astype(str) + ' ' + df['Filter'].astype(str) + ' ' + df['Selection'].astype(str) + ' ' + df['Condition'].astype(str)

# Create a new dataframe with `Combined_Text` and `Tags`
df_processed = df[['Combined_Text', 'Tags']]

# Oversampling minority classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df_processed[['Combined_Text']], df_processed['Tags'])
df_processed = pd.concat([X_resampled, y_resampled], axis=1)

# Split the data into training and test sets with stratified sampling
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(df_processed, df_processed['Tags']):
    train_set = df_processed.loc[train_index]
    test_set = df_processed.loc[test_index]

# TF-IDF Vectorizer for text feature extraction
tfidf_vectorizer = TfidfVectorizer()

# SelectKBest for feature selection
selector = SelectKBest(chi2)

# Pipeline for preprocessing and classification
pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('selector', selector),
    ('clf', None)  # Placeholder for the classifier
])

# Define different models (excluding XGBClassifier)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier()
}

# Function to evaluate a model using cross-validation
def evaluate_model(model):
    pipeline.set_params(clf=model)
    scores = cross_val_score(pipeline, train_set['Combined_Text'], train_set['Tags'], cv=5, scoring='f1_weighted')
    return scores.mean()

# Store evaluation results
results = {}

# Iterate over models and k values
for model_name, model in models.items():
    for k in [500, 1000, 1500, 2000]:
        pipeline.set_params(clf=model, selector__k=k)
        f1_score = evaluate_model(model)
        results[(model_name, k)] = f1_score

# Find the best model and k
best_model, best_k = max(results, key=results.get)
best_f1_score = results[(best_model, best_k)]

# Train the best model on the full training set
pipeline.set_params(clf=models[best_model], selector__k=best_k)
pipeline.fit(train_set['Combined_Text'], train_set['Tags'])

# Make predictions on the test set
y_pred = pipeline.predict(test_set['Combined_Text'])

# Evaluate the model on the test set
report = classification_report(test_set['Tags'], y_pred)

# Print the results
print(f"Best Model: {best_model}, Best k: {best_k}, Best F1-Score: {best_f1_score}")
print(report)
