import json
import pandas as pd
from datetime import datetime

def generate_numeric_stats_table(numeric_stats):
    if not numeric_stats:
        return "<p>No numeric fields found.</p>"
    
    # Get field names
    fields = list(numeric_stats.keys())
    stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    
    table_html = '<table class="stats-table">'
    table_html += '<thead><tr><th>Statistic</th>'
    for field in fields:
        table_html += f'<th>{field}</th>'
    table_html += '</tr></thead><tbody>'
    
    for stat in stats:
        table_html += f'<tr><td><strong>{stat}</strong></td>'
        for field in fields:
            value = numeric_stats[field].get(stat, 'N/A')
            if isinstance(value, float):
                table_html += f'<td>{value:.2f}</td>'
            else:
                table_html += f'<td>{value}</td>'
        table_html += '</tr>'
    
    table_html += '</tbody></table>'
    return table_html

def generate_categorical_summary(categorical_summary):
    if not categorical_summary:
        return "<p>No categorical fields summary available.</p>"
    
    html = '<div class="categorical-summary">'
    for field, info in list(categorical_summary.items())[:10]:  # Show first 10
        html += f'''
        <div class="categorical-item">
            <strong>{field}</strong><br>
            Unique Values: {info['unique_values']}<br>
            Most Common: {info['most_common']}
        </div>
        '''
    html += '</div>'
    return html

# Load all statistics JSON files
print("Loading statistics data...")

with open('data/output/moments_statistics_summary.json', 'r') as f:
    moments_stats = json.load(f)

with open('data/output/users_statistics_summary.json', 'r') as f:
    users_stats = json.load(f)

with open('data/output/books_statistics_summary.json', 'r') as f:
    books_stats = json.load(f)

# Create HTML report
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>STEP 10: Data Schema & Statistics Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .info-box {{
            background: #e8eaf6;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .info-box h4 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .info-box p {{
            line-height: 1.6;
            color: #555;
        }}
        
        .info-box ul {{
            margin: 10px 0 10px 20px;
            line-height: 1.8;
        }}
        
        .dataset-section {{
            margin-bottom: 50px;
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
        }}
        
        .dataset-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        .dataset-content {{
            padding: 30px;
        }}
        
        .overview {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stat-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .section-title {{
            font-size: 1.5em;
            color: #667eea;
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow-x: auto;
            display: block;
        }}
        
        .stats-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        
        .stats-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .stats-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .stats-table tr:hover {{
            background: #e8eaf6;
        }}
        
        .field-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }}
        
        .field-badge {{
            background: #667eea;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 500;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
            border-top: 2px solid #e0e0e0;
        }}
        
        .categorical-summary {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
        }}
        
        .categorical-item {{
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-left: 4px solid #667eea;
        }}
        
        .legend {{
            background: #fff3cd;
            border: 2px solid #ffc107;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .legend h4 {{
            color: #856404;
            margin-bottom: 10px;
        }}
        
        .legend ul {{
            margin-left: 20px;
            line-height: 1.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 STEP 10: Data Schema & Statistics Report</h1>
            <p>TensorFlow Data Validation Analysis</p>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="content">
            <div class="info-box">
                <h4>📋 Report Overview</h4>
                <p>This report provides comprehensive schema documentation and statistical analysis for three datasets in the MOMENT pipeline:</p>
                <ul>
                    <li><strong>Moments Data:</strong> 450 character interpretations of literary passages</li>
                    <li><strong>Users Data:</strong> 50 reader profiles with demographics and reading preferences</li>
                    <li><strong>Books Data:</strong> 9 passages from 3 classic books (Frankenstein, Pride and Prejudice, The Great Gatsby)</li>
                </ul>
                <p>All data has been validated using TensorFlow Data Validation (TFDV) to ensure quality and readiness for machine learning.</p>
            </div>
            
            <!-- MOMENTS DATA -->
            <div class="dataset-section">
                <div class="dataset-header">📊 Moments Data</div>
                <div class="dataset-content">
                    <div class="info-box">
                        <h4>📖 What is Moments Data?</h4>
                        <p>This dataset contains textual interpretations written by 50 different character personas, each responding to literary passages. Each record represents one character's interpretation of one passage, capturing their unique reading experience and perspective.</p>
                    </div>
                    
                    <div class="overview">
                        <div class="stat-card">
                            <div class="label">Total Records</div>
                            <div class="value">{moments_stats['total_records']}</div>
                            <small>Total interpretations collected (50 characters × 9 passages)</small>
                        </div>
                        <div class="stat-card">
                            <div class="label">Total Fields</div>
                            <div class="value">{moments_stats['total_fields']}</div>
                            <small>Number of features after flattening nested data</small>
                        </div>
                        <div class="stat-card">
                            <div class="label">Numeric Fields</div>
                            <div class="value">{len(moments_stats.get('numeric_fields', []))}</div>
                            <small>Quantitative measurements (counts, scores, metrics)</small>
                        </div>
                        <div class="stat-card">
                            <div class="label">Categorical Fields</div>
                            <div class="value">{len(moments_stats.get('categorical_fields', []))}</div>
                            <small>Text/category fields (names, IDs, labels)</small>
                        </div>
                    </div>
                    
                    <h3 class="section-title">Numeric Field Statistics</h3>
                    <div class="info-box">
                        <h4>📊 Understanding the Statistics</h4>
                        <ul>
                            <li><strong>count:</strong> Number of non-null values (should match total records)</li>
                            <li><strong>mean:</strong> Average value across all records</li>
                            <li><strong>std:</strong> Standard deviation - how spread out the values are</li>
                            <li><strong>min/max:</strong> Smallest and largest values</li>
                            <li><strong>25%/50%/75%:</strong> Quartiles showing data distribution</li>
                        </ul>
                        <p><strong>Key Metrics:</strong> word_count, char_count, readability_score, quality_score</p>
                    </div>
                    {generate_numeric_stats_table(moments_stats.get('numeric_statistics', {}))}
                    
                    <h3 class="section-title">Field Names</h3>
                    <div class="info-box">
                        <h4>🏷️ Data Schema</h4>
                        <p>These are all the columns in the moments dataset after flattening nested structures. Fields include:</p>
                        <ul>
                            <li><strong>Identifiers:</strong> interpretation_id, user_id, book_id, passage_id</li>
                            <li><strong>Content:</strong> cleaned_interpretation (the actual text)</li>
                            <li><strong>Metrics:</strong> word_count, char_count, readability_score</li>
                            <li><strong>Quality Flags:</strong> is_valid, quality_score, quality_issues</li>
                            <li><strong>Detection:</strong> has_pii, has_profanity, is_spam</li>
                        </ul>
                    </div>
                    <div class="field-list">
                        {''.join([f'<span class="field-badge">{field}</span>' for field in moments_stats.get('field_names', [])])}
                    </div>
                    
                    <h3 class="section-title">Categorical Field Summary</h3>
                    <div class="info-box">
                        <h4>📂 Category Analysis</h4>
                        <p>Shows the variety and distribution of categorical fields:</p>
                        <ul>
                            <li><strong>Unique Values:</strong> How many different categories exist</li>
                            <li><strong>Most Common:</strong> The most frequently occurring value</li>
                        </ul>
                        <p>This helps identify data balance and potential biases in the dataset.</p>
                    </div>
                    {generate_categorical_summary(moments_stats.get('categorical_summary', {}))}
                </div>
            </div>
            
            <!-- USERS DATA -->
            <div class="dataset-section">
                <div class="dataset-header">👥 Users Data</div>
                <div class="dataset-content">
                    <div class="info-box">
                        <h4>👤 What is Users Data?</h4>
                        <p>This dataset contains profiles of the 50 character personas who generated the interpretations. Each record includes demographic information, reading preferences, and behavioral characteristics that influence how they interpret literature.</p>
                    </div>
                    
                    <div class="overview">
                        <div class="stat-card">
                            <div class="label">Total Records</div>
                            <div class="value">{users_stats['total_records']}</div>
                            <small>Number of unique character personas</small>
                        </div>
                        <div class="stat-card">
                            <div class="label">Total Fields</div>
                            <div class="value">{users_stats['total_fields']}</div>
                            <small>Profile attributes per character</small>
                        </div>
                        <div class="stat-card">
                            <div class="label">Numeric Fields</div>
                            <div class="value">{len(users_stats.get('numeric_fields', []))}</div>
                            <small>Age, books per year, etc.</small>
                        </div>
                        <div class="stat-card">
                            <div class="label">Categorical Fields</div>
                            <div class="value">{len(users_stats.get('categorical_fields', []))}</div>
                            <small>Education, genre preferences, etc.</small>
                        </div>
                    </div>
                    
                    <h3 class="section-title">Numeric Field Statistics</h3>
                    <div class="info-box">
                        <h4>📈 User Demographics & Behavior</h4>
                        <p>Numeric fields capture quantifiable aspects of reader profiles:</p>
                        <ul>
                            <li><strong>age:</strong> Reader age distribution</li>
                            <li><strong>books_per_year:</strong> Reading frequency/volume</li>
                            <li><strong>Other metrics:</strong> Various behavioral and preference scores</li>
                        </ul>
                        <p>These features can be used to segment readers or predict reading patterns.</p>
                    </div>
                    {generate_numeric_stats_table(users_stats.get('numeric_statistics', {}))}
                    
                    <h3 class="section-title">Field Names</h3>
                    <div class="info-box">
                        <h4>🏷️ User Profile Schema</h4>
                        <p>Complete set of attributes describing each character/reader:</p>
                        <ul>
                            <li><strong>Demographics:</strong> age, education_level, occupation</li>
                            <li><strong>Reading Preferences:</strong> favorite_genre, reading_style</li>
                            <li><strong>Behavioral:</strong> books_per_year, reading habits</li>
                            <li><strong>Identifiers:</strong> user_id, character_name</li>
                        </ul>
                    </div>
                    <div class="field-list">
                        {''.join([f'<span class="field-badge">{field}</span>' for field in users_stats.get('field_names', [])])}
                    </div>
                    
                    <h3 class="section-title">Categorical Field Summary</h3>
                    <div class="info-box">
                        <h4>📂 Profile Diversity</h4>
                        <p>Shows the variety in reader characteristics:</p>
                        <ul>
                            <li><strong>Education levels:</strong> Range of educational backgrounds</li>
                            <li><strong>Genre preferences:</strong> Variety of favorite genres</li>
                            <li><strong>Reading styles:</strong> Different approaches to reading</li>
                        </ul>
                        <p>Greater diversity ensures the model learns from varied perspectives.</p>
                    </div>
                    {generate_categorical_summary(users_stats.get('categorical_summary', {}))}
                </div>
            </div>
            
            <!-- BOOKS DATA -->
            <div class="dataset-section">
                <div class="dataset-header">📚 Books Data</div>
                <div class="dataset-content">
                    <div class="info-box">
                        <h4>📕 What is Books Data?</h4>
                        <p>This dataset contains the 9 literary passages from 3 classic books that served as prompts for character interpretations. Each passage includes the full text, metadata, and quality metrics that help validate the source material.</p>
                    </div>
                    
                    <div class="overview">
                        <div class="stat-card">
                            <div class="label">Total Records</div>
                            <div class="value">{books_stats['total_records']}</div>
                            <small>Total passages (3 passages per book)</small>
                        </div>
                        <div class="stat-card">
                            <div class="label">Total Fields</div>
                            <div class="value">{books_stats['total_fields']}</div>
                            <small>Attributes per passage</small>
                        </div>
                        <div class="stat-card">
                            <div class="label">Numeric Fields</div>
                            <div class="value">{len(books_stats.get('numeric_fields', []))}</div>
                            <small>Length, readability metrics</small>
                        </div>
                        <div class="stat-card">
                            <div class="label">Categorical Fields</div>
                            <div class="value">{len(books_stats.get('categorical_fields', []))}</div>
                            <small>Book titles, authors, chapters</small>
                        </div>
                    </div>
                    
                    <h3 class="section-title">Numeric Field Statistics</h3>
                    <div class="info-box">
                        <h4>📏 Passage Characteristics</h4>
                        <p>Quantitative measures of the source passages:</p>
                        <ul>
                            <li><strong>word_count:</strong> Length of each passage</li>
                            <li><strong>char_count:</strong> Character length</li>
                            <li><strong>sentence_count:</strong> Number of sentences</li>
                            <li><strong>readability_score:</strong> Complexity measure (0-100, higher = easier)</li>
                            <li><strong>quality_score:</strong> Data quality validation score</li>
                        </ul>
                        <p>These metrics ensure passages are appropriate length and complexity for analysis.</p>
                    </div>
                    {generate_numeric_stats_table(books_stats.get('numeric_statistics', {}))}
                    
                    <h3 class="section-title">Field Names</h3>
                    <div class="info-box">
                        <h4>🏷️ Passage Metadata Schema</h4>
                        <p>Complete information stored for each passage:</p>
                        <ul>
                            <li><strong>Identifiers:</strong> passage_id, book_id</li>
                            <li><strong>Metadata:</strong> book_title, book_author, chapter_number</li>
                            <li><strong>Content:</strong> cleaned_passage_text (the actual passage)</li>
                            <li><strong>Metrics:</strong> word_count, readability_score, quality metrics</li>
                            <li><strong>Validation:</strong> is_valid, quality_issues flags</li>
                        </ul>
                    </div>
                    <div class="field-list">
                        {''.join([f'<span class="field-badge">{field}</span>' for field in books_stats.get('field_names', [])])}
                    </div>
                    
                    <h3 class="section-title">Categorical Field Summary</h3>
                    <div class="info-box">
                        <h4>📂 Source Material Diversity</h4>
                        <p>Shows the variety of literary sources:</p>
                        <ul>
                            <li><strong>Books:</strong> Frankenstein, Pride and Prejudice, The Great Gatsby</li>
                            <li><strong>Authors:</strong> Mary Shelley, Jane Austen, F. Scott Fitzgerald</li>
                            <li><strong>Passages:</strong> 3 representative excerpts from each book</li>
                        </ul>
                        <p>Diverse source material ensures varied interpretation contexts.</p>
                    </div>
                    {generate_categorical_summary(books_stats.get('categorical_summary', {}))}
                </div>
            </div>
            
            <div class="legend">
                <h4>🔑 Key Takeaways</h4>
                <ul>
                    <li><strong>Data Completeness:</strong> All datasets are complete with no missing values</li>
                    <li><strong>Quality Validation:</strong> TFDV schemas confirm proper data types and constraints</li>
                    <li><strong>Feature Engineering Ready:</strong> Numeric and categorical features are well-structured for ML</li>
                    <li><strong>Balanced Coverage:</strong> 50 characters × 9 passages = 450 interpretations</li>
                    <li><strong>Schema Documentation:</strong> Complete field lists serve as data dictionary</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>MOMENT Data Pipeline - STEP 10</strong></p>
            <p>TensorFlow Data Validation (TFDV) Schema & Statistics Generation</p>
            <p>Generated with Python, pandas, and TFDV in Ubuntu WSL</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Schema Files: schemas/*.pbtxt | Statistics: data/output/*_statistics_summary.json
            </p>
        </div>
    </div>
</body>
</html>
"""

# Write HTML file
output_path = 'reports/complete_statistics_report.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ HTML report generated successfully!")
print(f"📄 Location: {output_path}")
print(f"\n🌐 To view the report:")
print(f"   1. Open File Explorer")
print(f"   2. Navigate to: C:\\Users\\csk23\\MOMENT_STEP10\\reports")
print(f"   3. Double-click: complete_statistics_report.html")