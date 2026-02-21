import csv

def remove_newlines_from_csv(input_file, output_file):
    """
    Remove newlines from all text fields in a CSV file
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    
    rows = []
    
    # Read the CSV file
    with open(input_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Get headers
        
        for row in reader:
            # Remove newlines from each cell
            cleaned_row = [cell.replace('\n', ' ').replace('\r', ' ') for cell in row]
            rows.append(cleaned_row)
    
    # Write the cleaned data
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Successfully processed {len(rows)} rows")
    print(f"Output saved to: {output_file}")

# Main execution
if __name__ == "__main__":
    input_csv = "data/user_interpretations.csv"
    output_csv = "data/user_interpretations.csv"  
    
    try:
        remove_newlines_from_csv(input_csv, output_csv)
    except FileNotFoundError:
        print(f"Error: File '{input_csv}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")