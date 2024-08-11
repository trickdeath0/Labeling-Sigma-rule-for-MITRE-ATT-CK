import csv
import os
import yaml
import re

# Define headers and their order
headers = [
    #'Title',
    'Sid',
    #'Status',
    #'Description',
    'Detection',
    #'False Positives',
    'Level',
    'Tags'
]

yaml_dir = r'02 Windows Rules\windows'  # Define the directory containing the YAML files
dst_dir = r'03 Clean Data'
csv_file = os.path.join(dst_dir, 'windows_rules_filter.csv')  # Full path for CSV file

# Function to filter tags to only include unique TXXX numbers with a capital 'T'
def filter_tags(tags):
    # Use regex to find all TXXXX patterns
    pattern = r'\bt\d{4}\b'
    # Find all matches
    matches = re.findall(pattern, tags, re.IGNORECASE)
    # Remove duplicates by converting the list to a set and back to a list
    unique_matches = sorted(set(matches), key=lambda x: (int(x[1:]), x))
    # Convert matches to uppercase
    unique_matches = [match.upper() for match in unique_matches]
    # Return as a list
    return unique_matches

# Create the CSV file and write the headers
try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    print(f"CSV file created: {csv_file}")
except Exception as e:
    print(f"Error creating CSV file: {e}")

# Walk through the YAML files and extract the information
for root, dirs, files in os.walk(yaml_dir):
    for file in files:
        if file.endswith('.yaml') or file.endswith('.yml'):
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as yaml_file:
                    data = yaml.safe_load(yaml_file)
                    
                    # Check if YAML data is correctly loaded
                    if data is None:
                        print(f"Warning: No data in file {file_path}")
                        continue
                    
                    # Extract data according to headers
                    tags = ', '.join(data.get('tags', []))
                    filtered_tags = filter_tags(tags)
                    
                    # Skip rows with empty Tags
                    if not filtered_tags:
                        continue
                    
                    # Create the row with filtered Tags
                    row = [
                        #data.get('title', ''),
                        data.get('id', ''),
                        #data.get('status', ''),
                        #data.get('description', ''),
                        str(data.get('detection', {})),
                        #', '.join(data.get('falsepositives', [])),
                        data.get('level', ''),
                        filtered_tags
                    ]
                    
                    # Append rows to CSV file
                    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(row)
                    #print(f"Row added for file: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
