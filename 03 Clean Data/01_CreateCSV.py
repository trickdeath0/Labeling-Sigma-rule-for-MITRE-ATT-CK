import os
import csv
import yaml

yaml_dir = r'02 Windows Rules\windows'  # Define the directory containing the YAML files
dst_dir = r'03 Clean Data'
csv_file = os.path.join(dst_dir, 'windows_rules_filter.csv')  # Full path for CSV file


headers = [
    'Title',
    'ID',
    'Related IDs',
    'Status',
    'Description',
    'References',
    'Author',
    'Date',
    'Modified',
    'Tags',
    'Log Source',
    'Detection',
    'False Positives',
    'Level'
]




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
                    
                    
                    row = [
                        data.get('title', ''),
                        data.get('id', ''),
                        ', '.join([related.get('id', '') for related in data.get('related', [])]),
                        data.get('status', ''),
                        data.get('description', ''),
                        ', '.join(data.get('references', [])),
                        data.get('author', ''),
                        data.get('date', ''),
                        data.get('modified', ''),
                        ', '.join(data.get('tags', [])),
                        data.get('logsource', {}).get('product', ''),
                        str(data.get('detection', {})),
                        ', '.join(data.get('falsepositives', [])),
                        data.get('level', '')
                    ]
                    
                    
                    # Append rows to CSV file
                    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(row)
                    #print(f"Row added for file: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
