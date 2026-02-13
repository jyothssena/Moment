import PyPDF2
import re
import pandas as pd
import json

def parse_pdf_to_datasets(pdf_path):
    """
    Parse PDF containing passages and character interpretations into two datasets.
    Handles multiple format variations including detailed demographic sections.
    """
    
    # Read PDF
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()
    
    # Initialize data structures
    passages = []
    interpretations = []
    
    # Try to extract book info from headers
    book_match = re.search(r'^([^-\n]+?)\s*-\s*PASSAGE', full_text, re.MULTILINE)
    default_book = book_match.group(1).strip() if book_match else "Unknown"
    
    # Split by passages
    passage_pattern = r'(?:([^-\n]+?)\s*-\s*)?PASSAGE\s+(\d+):\s+([^\n]+?)(?:\s*\((?:Chapter|Letter)\s+([^)]+)\))?(?:\s*-\s*)?(?:All\s+(\d+)\s+Character\s+Interpretations)?'
    
    matches = list(re.finditer(passage_pattern, full_text))
    
    for idx, match in enumerate(matches):
        book_title = match.group(1).strip() if match.group(1) else default_book
        passage_id = int(match.group(2))
        passage_title = match.group(3).strip()
        chapter_number = match.group(4).strip() if match.group(4) else "Unknown"
        num_interpretations = int(match.group(5)) if match.group(5) else None
        
        # Get content between this match and next match
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        section_content = full_text[start_pos:end_pos]
        
        # Extract book info if present
        book_info_match = re.search(r'Book:\s*([^\n]+)', section_content)
        if book_info_match:
            book_title = book_info_match.group(1).strip()
        
        book_author = "Unknown"
        
        # Extract passage location if present
        location_match = re.search(r'Passage Location:\s*([^\n]+)', section_content)
        if location_match:
            chapter_number = location_match.group(1).strip()
        
        # Extract passage text - try multiple patterns
        passage_text = None
        
        # Try with various quote styles
        for quote_pattern in [r'Passage Text:\s*["\"](.+?)["\"]', 
                             r'Passage:\s*["\"](.+?)["\"]',
                             r'Passage Text:\s*\n(.+?)(?:\n---|CHARACTER|Character)']:
            pattern_match = re.search(quote_pattern, section_content, re.DOTALL)
            if pattern_match:
                passage_text = pattern_match.group(1).strip()
                break
        
        if passage_text:
            passages.append({
                'passage_id': passage_id,
                'book_title': book_title,
                'book_author': book_author,
                'chapter_number': chapter_number,
                'passage_title': passage_title,
                'passage_text': passage_text,
                'num_interpretations': num_interpretations
            })
        
        # Extract character interpretations
        # Split by "---" or "CHARACTER X:" patterns
        char_splits = re.split(r'(?:---|(?:CHARACTER|Character)\s+\d+:)\s*([^\n]+)', section_content)
        
        # Alternative: find all character blocks
        char_pattern = r'(?:---|(?:CHARACTER|Character)\s+\d+:)\s*([^\n]+?)(?=\n)((?:(?!---|(?:CHARACTER|Character)\s+\d+:).)+)'
        char_matches = list(re.finditer(char_pattern, section_content, re.DOTALL))
        
        for char_match in char_matches:
            character_name = char_match.group(1).strip()
            char_content = char_match.group(2)
            
            # Extract demographics section
            gender = age = profession = None
            demo_match = re.search(r'Gender:\s*([^|]+)\s*\|\s*Age:\s*(\d+)\s*\|\s*Profession:\s*([^\n]+)', char_content)
            if demo_match:
                gender = demo_match.group(1).strip()
                age = int(demo_match.group(2).strip())
                profession = demo_match.group(3).strip()
            else:
                # Try condensed format: Female | 28 | Data Scientist
                profile_match = re.search(r'(?:Profile:)?\s*([^|\n]+)\s*\|\s*(\d+)\s*\|\s*([^|\n]+)', char_content)
                if profile_match:
                    gender = profile_match.group(1).strip()
                    age = int(profile_match.group(2).strip())
                    profession = profile_match.group(3).strip()
            
            # Extract reading profile details
            reading_style_match = re.search(r'Reading Style:\s*([^|\n]+?)(?:\s*\||$)', char_content)
            reading_style = reading_style_match.group(1).strip() if reading_style_match else ""
            
            genre_match = re.search(r'Genre(?:\s+Taste)?:\s*([^\n]+)', char_content)
            genre = genre_match.group(1).strip() if genre_match else ""
            
            volume_match = re.search(r'Reading Volume:\s*([^\n]+)', char_content)
            volume = volume_match.group(1).strip() if volume_match else ""
            
            classics_match = re.search(r'Classics Familiarity:\s*([^\n]+)', char_content)
            classics = classics_match.group(1).strip() if classics_match else ""
            
            favorites_match = re.search(r'Favorite Books:\s*([^\n]+)', char_content)
            favorites = favorites_match.group(1).strip() if favorites_match else ""
            
            journey_match = re.search(r'Reading Journey:\s*["\"]([^""]+)["\"]', char_content)
            reading_journey = journey_match.group(1).strip() if journey_match else ""
            
            # Extract interpretation style
            interp_style_match = re.search(r'Interpretation Style:\s*\n(.+?)(?=\n\nInterpretation:|\nInterpretation:)', char_content, re.DOTALL)
            interpretation_style = ""
            if interp_style_match:
                style_text = interp_style_match.group(1).strip()
                interpretation_style = style_text.replace('\n', ' | ')
            
            # Extract writing/focus/depth/length if available
            writing_match = re.search(r'Writing:\s*([^|\n]+)', char_content)
            focus_match = re.search(r'Focus:\s*([^|\n]+)', char_content)
            depth_match = re.search(r'Depth:\s*([^|\n]+)', char_content)
            length_match = re.search(r'Length:\s*([^\n]+)', char_content)
            
            if any([writing_match, focus_match, depth_match, length_match]):
                style_parts = []
                if writing_match: style_parts.append(f"Writing: {writing_match.group(1).strip()}")
                if focus_match: style_parts.append(f"Focus: {focus_match.group(1).strip()}")
                if depth_match: style_parts.append(f"Depth: {depth_match.group(1).strip()}")
                if length_match: style_parts.append(f"Length: {length_match.group(1).strip()}")
                interpretation_style = " | ".join(style_parts)
            
            # Extract interpretation text (without word count if present)
            interp_match = re.search(r'Interpretation(?:\s+\((\d+)\s+words\))?:\s*(.+?)(?=(?:---|CHARACTER|Character)\s+\d+:|$)', 
                                    char_content, re.DOTALL)
            
            if interp_match:
                word_count = int(interp_match.group(1)) if interp_match.group(1) else None
                interpretation_text = interp_match.group(2).strip()
                
                # Calculate word count if not provided
                if not word_count:
                    word_count = len(interpretation_text.split())
                
                interpretations.append({
                    'passage_id': passage_id,
                    'character_name': character_name,
                    'gender': gender,
                    'age': age,
                    'profession': profession,
                    'reading_style': reading_style,
                    'genre': genre,
                    'volume': volume,
                    'classics': classics,
                    'favorites': favorites,
                    'reading_journey': reading_journey,
                    'interpretation_style': interpretation_style,
                    'interpretation_text': interpretation_text,
                    'word_count': word_count
                })
    
    # Create DataFrames
    passages_df = pd.DataFrame(passages)
    interpretations_df = pd.DataFrame(interpretations)
    
    return passages_df, interpretations_df


# Example usage
if __name__ == "__main__":
    pdf_paths=['data/1.Frankenstein/Frankenstein_Passage_1_SELECTABLE.pdf',
                'data/1.Frankenstein/Frankenstein_Passage_2_SELECTABLE.pdf',
                'data/1.Frankenstein/Frankenstein_Passage_3_SELECTABLE.pdf',
                'data/2.Pride and Prejudice/Pride_and_Prejudice_Passage_1_SELECTABLE.pdf',
                'data/2.Pride and Prejudice/Pride_and_Prejudice_Passage_2_SELECTABLE.pdf',
                'data/2.Pride and Prejudice/Pride_and_Prejudice_Passage_3_SELECTABLE.pdf',
                'data/3.The Great Gatsby/Gatsby_Passage_1_SELECTABLE.pdf',
                'data/3.The Great Gatsby/Gatsby_Passage_2_SELECTABLE.pdf',
                'data/3.The Great Gatsby/Gatsby_Passage_3_SELECTABLE.pdf',
                
                ]   # Replace with your PDF path
    '''
        pdf_paths=['data/1.Frankenstein/Frankenstein_Passage_1_SELECTABLE.pdf',
                'data/1.Frankenstein/Frankenstein_Passage_2_SELECTABLE.pdf',
                'data/1.Frankenstein/Frankenstein_Passage_3_SELECTABLE.pdf',
                'data/2.Pride and Prejudice/Pride_and_Prejudice_Passage_1_SELECTABLE.pdf',
                'data/2.Pride and Prejudice/Pride_and_Prejudice_Passage_2_SELECTABLE.pdf',
                'data/2.Pride and Prejudice/Pride_and_Prejudice_Passage_3_SELECTABLE.pdf',
                'data/3.The Great Gatsby/Gatsby_Passage_1_SELECTABLE.pdf',
                'data/3.The Great Gatsby/Gatsby_Passage_2_SELECTABLE.pdf',
                'data/3.The Great Gatsby/Gatsby_Passage_3_SELECTABLE.pdf',
                
                ] 
    '''
    # Parse the PDF
    passages_df=pd.DataFrame()
    interpretations_df=pd.DataFrame()
    for i in pdf_paths:
        passages_temp, interpretations_temp = parse_pdf_to_datasets(i)
        passages_df = pd.concat([passages_df, passages_temp], ignore_index=True)
        interpretations_df = pd.concat([interpretations_df, interpretations_temp], ignore_index=True)
        # Display results
        print("PASSAGES DATASET:")
        print(passages_df)
        print(f"\nTotal passages: {len(passages_df)}")
    
        print("\n" + "="*80 + "\n")
        
        print("INTERPRETATIONS DATASET:")
        print(interpretations_df)
        print(f"\nTotal interpretations: {len(interpretations_df)}")
        
    passages_df.to_csv('passages.csv', index=False)
    interpretations_df.to_csv('interpretations.csv', index=False)
        
    # Save to JSON (alternative format)
    passages_df.to_json('passages.json', orient='records', indent=2)
    interpretations_df.to_json('interpretations.json', orient='records', indent=2)
        
    print("\n✓ Datasets saved to CSV and JSON files")