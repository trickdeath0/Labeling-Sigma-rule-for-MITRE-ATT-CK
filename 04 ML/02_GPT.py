# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import precision_recall_fscore_support
# from imblearn.over_sampling import RandomOverSampler
# import json

# # Load the data
# data = pd.read_csv(r'03 Clean Data\windows_rules_filter.csv')

# def clean_and_parse_detection(detection_str):
#     # Replace single quotes with double quotes
#     detection_str = detection_str.replace("'", "\"")
#     try:
#         # Load the JSON string
#         detection_json = json.loads(detection_str)
#         # Extract keywords if available
#         keywords = detection_json.get('keywords', [])
#         # If no keywords, extract selections as a single string
#         if not keywords:
#             selections = detection_json.get('selection', {})
#             if isinstance(selections, dict):
#                 keywords = ' '.join([f"{k}:{v}" for k, v in selections.items()])
#             elif isinstance(selections, list):
#                 keywords = ' '.join(selections)
#         return ' '.join(keywords) if isinstance(keywords, list) else keywords
#     except json.JSONDecodeError:
#         return ""

# # Apply the cleaning and parsing function
# data['Detection'] = data['Detection'].apply(clean_and_parse_detection)

# # Preprocess the 'Tags' column
# data['Tags'] = data['Tags'].apply(lambda x: x.strip("[]").replace("'", "").split(', '))

# # Binarize the techniques
# all_techniques = set(tech for sublist in data['Tags'] for tech in sublist)
# for tech in all_techniques:
#     data[tech] = data['Tags'].apply(lambda x: 1 if tech in x else 0)

# # Extract features and labels
# X = data['Detection']
# y = data[list(all_techniques)]

# # Vectorize the text data using TF-IDF
# vectorizer = TfidfVectorizer()
# X_tfidf = vectorizer.fit_transform(X)

# # Feature selection
# selector = SelectKBest(chi2, k='all')
# X_selected = selector.fit_transform(X_tfidf, y.values.argmax(axis=1))

# # Oversample the training data
# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(X_selected, y.values.argmax(axis=1))

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# # Train several classification models
# models = {
#     'RandomForest': RandomForestClassifier(),
#     'LogisticRegression': LogisticRegression(max_iter=1000),
#     'SVM': SVC()
# }

# # Perform cross-validation and evaluate models
# results = {}
# for model_name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)
#     results[model_name] = {'precision': precision, 'recall': recall, 'f1': f1}

# # Display the results
# results
