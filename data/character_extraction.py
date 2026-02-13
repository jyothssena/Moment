import re
import csv
import PyPDF2

def extract_character_data(pdf_path):
    """Extract character data from PDF and return as list of dictionaries"""
    
    # Read PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    characters = []
    
    # Split by character entries (numbered entries)
    entries = re.split(r'\n\d+\.\s+', text)
    
    for entry in entries[1:]:  # Skip first split (before first number)
        char_data = {}
        
        # Extract name and category
        name_match = re.search(r'^(.+?)\s+\[(.+?)\]', entry)
        if name_match:
            char_data['Name'] = name_match.group(1).strip()
            char_data['Distribution_Category'] = name_match.group(2).strip()
        
        # Extract demographics line
        demo_match = re.search(r'(Male|Female)\s+\|\s+(\d+)\s+\|\s+(.+?)\s+\|\s+(.+?)\s+\|\s+(.+?)\s+\|\s+(.+?)\s+\((\d+)\)\s+\|\s+(.+?)\s+\((\d+)\)', entry)
        if demo_match:
            char_data['Gender'] = demo_match.group(1)
            char_data['Age'] = demo_match.group(2)
            char_data['Profession'] = demo_match.group(3).strip()
            char_data['Personality'] = demo_match.group(4).strip()
            char_data['Interest'] = demo_match.group(5).strip()
            char_data['Reading_Intensity'] = demo_match.group(6).strip()
            char_data['Reading_Count'] = demo_match.group(7)
            char_data['Experience_Level'] = demo_match.group(8).strip()
            char_data['Experience_Count'] = demo_match.group(9)
        
        # Extract journey
        journey_match = re.search(r'Journey:\s+"(.+?)"', entry, re.DOTALL)
        if journey_match:
            char_data['Journey'] = journey_match.group(1).strip().replace('\n', ' ')
        
        # Extract style
        style_match = re.search(r'Style:\s+(.+?)(?:\n|$)', entry)
        if style_match:
            styles = style_match.group(1).strip().split('|')
            char_data['Style_1'] = styles[0].strip() if len(styles) > 0 else ''
            char_data['Style_2'] = styles[1].strip() if len(styles) > 1 else ''
            char_data['Style_3'] = styles[2].strip() if len(styles) > 2 else ''
            char_data['Style_4'] = styles[3].strip() if len(styles) > 3 else ''
        
        if char_data:
            characters.append(char_data)
    
    return characters

def save_to_csv(characters, output_path):
    """Save character data to CSV file"""
    
    if not characters:
        print("No character data found")
        return
    
    # Define CSV headers
    headers = [
        'Name', 'Distribution_Category', 'Gender', 'Age', 'Profession',
        'Personality', 'Interest', 'Reading_Intensity', 'Reading_Count',
        'Experience_Level', 'Experience_Count', 'Journey',
        'Style_1', 'Style_2', 'Style_3', 'Style_4'
    ]
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(characters)
    
    print(f"Successfully saved {len(characters)} characters to {output_path}")

# Main execution
if __name__ == "__main__":
    # Replace with your PDF file path
    pdf_path = "data/0.Character traits - 50.pdf"
    output_csv = "characters.csv"
    
    try:
        characters = extract_character_data(pdf_path)
        save_to_csv(characters, output_csv)
    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found")
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")