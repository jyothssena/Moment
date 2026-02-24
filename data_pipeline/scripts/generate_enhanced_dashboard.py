"""
Enhanced Log Dashboard Generator - Task-Separated View
Generates dashboard with logs organized by BOTH task and log level.

Usage:
    python generate_enhanced_dashboard.py --log-file /opt/airflow/logs/team_pipeline_ALL.log
"""

import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json


def parse_log_line(line):
    """Parse log line - handles both Airflow formats."""
    # Pattern 1: With airflow.task
    pattern1 = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - .+ - (\w+) - \[(.+?):(\d+)\] - (.+)'
    match = re.match(pattern1, line)
    
    if match:
        return {
            'timestamp': match.group(1),
            'level': match.group(2),
            'file': match.group(3),
            'line': match.group(4),
            'message': match.group(5).strip()
        }
    
    # Pattern 2: Without airflow.task (simpler format)
    pattern2 = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\w+) - \[(.+?):(\d+)\] - (.+)'
    match = re.match(pattern2, line)
    
    if match:
        return {
            'timestamp': match.group(1),
            'level': match.group(2),
            'file': match.group(3),
            'line': match.group(4),
            'message': match.group(5).strip()
        }
    
    return None


def extract_task_name(message, file):
    """Extract task name from log message or filename."""
    # Check if message has [TaskName] prefix
    task_match = re.match(r'\[([^\]]+)\]', message)
    if task_match:
        return task_match.group(1)
    
    # Look for task indicators in message
    task_keywords = {
        'Data Acquisition': ['acquire', 'acquisition', 'load data', 'download', 'data/raw'],
        'Bias Detection': ['bias', 'demographic', 'gender', 'age'],
        'Preprocessing': ['preprocessing', 'preprocess', 'process_books', 'process_moments', 'data/processed'],
        'Schema Stats': ['schema', 'statistics', 'stats'],
        'Validation': ['validation', 'validate'],
        'Upload to GCS': ['upload', 'uploading', 'gs://', 'bucket', 'blob'],
        'Notification': ['notify', 'notification', 'email', 'pipeline complete']
    }
    
    msg_lower = message.lower()
    for task, keywords in task_keywords.items():
        if any(kw in msg_lower for kw in keywords):
            return task
    
    return 'General'


def categorize_logs_by_task_and_level(log_file_path):
    """Parse logs and categorize by BOTH task and level."""
    logs_by_task = defaultdict(lambda: defaultdict(list))
    logs_by_level = defaultdict(list)
    
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                task = extract_task_name(parsed['message'], parsed['file'])
                parsed['task'] = task
                
                # Store by task and level
                logs_by_task[task][parsed['level']].append(parsed)
                
                # Store by level only
                logs_by_level[parsed['level']].append(parsed)
    
    return logs_by_task, logs_by_level


def generate_enhanced_dashboard(logs_by_task, logs_by_level, output_path):
    """Generate enhanced HTML dashboard with task separation."""
    
    # Calculate totals
    info_total = len(logs_by_level.get('INFO', []))
    warning_total = len(logs_by_level.get('WARNING', []))
    error_total = len(logs_by_level.get('ERROR', [])) + len(logs_by_level.get('CRITICAL', []))
    debug_total = len(logs_by_level.get('DEBUG', []))
    
    # Task statistics
    task_stats = {}
    for task, levels in logs_by_task.items():
        task_stats[task] = {
            'info': len(levels.get('INFO', [])),
            'warning': len(levels.get('WARNING', [])),
            'error': len(levels.get('ERROR', [])) + len(levels.get('CRITICAL', [])),
            'debug': len(levels.get('DEBUG', [])),
            'total': sum(len(logs) for logs in levels.values())
        }
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOMENT Pipeline - Enhanced Logging Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a; color: #e2e8f0; padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px; border-radius: 12px; margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .subtitle {{ opacity: 0.9; font-size: 1.1em; }}
        .timestamp {{ color: #94a3b8; font-size: 0.9em; margin-top: 5px; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }}
        .stat-card {{
            background: #1e293b; padding: 25px; border-radius: 12px;
            border-left: 4px solid; transition: transform 0.2s;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-card.info {{ border-color: #3b82f6; }}
        .stat-card.warning {{ border-color: #f59e0b; }}
        .stat-card.error {{ border-color: #ef4444; }}
        .stat-card.debug {{ border-color: #8b5cf6; }}
        .stat-icon {{ font-size: 2.5em; margin-bottom: 10px; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.7; margin-bottom: 5px; }}
        .stat-value {{ font-size: 2.5em; font-weight: bold; }}
        
        .view-toggle {{
            display: flex; gap: 15px; margin-bottom: 30px;
            padding: 20px; background: #1e293b; border-radius: 12px;
        }}
        .toggle-btn {{
            padding: 12px 24px; border: none; border-radius: 8px;
            cursor: pointer; font-weight: bold; transition: all 0.2s;
            background: #334155; color: white;
        }}
        .toggle-btn.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .filter-buttons {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
        .filter-btn {{
            padding: 10px 20px; border: none; border-radius: 8px;
            cursor: pointer; font-weight: bold; transition: all 0.2s;
        }}
        .filter-btn:hover {{ transform: translateY(-2px); }}
        .filter-btn.active {{ box-shadow: 0 0 0 3px rgba(255,255,255,0.3); }}
        .filter-btn.all {{ background: #475569; color: white; }}
        .filter-btn.info {{ background: #3b82f6; color: white; }}
        .filter-btn.warning {{ background: #f59e0b; color: white; }}
        .filter-btn.error {{ background: #ef4444; color: white; }}
        .filter-btn.debug {{ background: #8b5cf6; color: white; }}
        
        .task-section {{
            background: #1e293b; border-radius: 12px;
            padding: 25px; margin-bottom: 20px;
        }}
        .task-header {{
            display: flex; justify-content: space-between;
            align-items: center; margin-bottom: 20px;
            padding-bottom: 15px; border-bottom: 2px solid #334155;
        }}
        .task-title {{ font-size: 1.5em; font-weight: bold; }}
        .task-counts {{
            display: flex; gap: 15px; font-size: 0.9em;
        }}
        .count-badge {{
            padding: 5px 12px; border-radius: 6px;
            font-weight: bold;
        }}
        .count-badge.info {{ background: #1e40af; }}
        .count-badge.warning {{ background: #b45309; }}
        .count-badge.error {{ background: #b91c1c; }}
        
        .log-section {{ margin-bottom: 20px; }}
        .log-section h2 {{ 
            margin-bottom: 15px; display: flex;
            align-items: center; gap: 10px; font-size: 1.3em;
        }}
        
        .log-entry {{
            background: #0f172a; padding: 12px 15px; border-radius: 8px;
            margin-bottom: 8px; border-left: 3px solid;
            font-family: 'Courier New', monospace; font-size: 0.85em;
            line-height: 1.5;
        }}
        .log-entry.info {{ border-color: #3b82f6; }}
        .log-entry.warning {{ border-color: #f59e0b; background: #1e1507; }}
        .log-entry.error {{ border-color: #ef4444; background: #1e0808; }}
        .log-entry.debug {{ border-color: #8b5cf6; }}
        
        .log-timestamp {{ color: #64748b; margin-right: 10px; }}
        .log-level {{
            padding: 2px 8px; border-radius: 4px;
            font-weight: bold; margin-right: 10px; font-size: 0.85em;
        }}
        .log-level.INFO {{ background: #1e40af; color: #dbeafe; }}
        .log-level.WARNING {{ background: #b45309; color: #fef3c7; }}
        .log-level.ERROR {{ background: #b91c1c; color: #fee2e2; }}
        .log-level.DEBUG {{ background: #6b21a8; color: #f3e8ff; }}
        
        .empty-state {{ text-align: center; padding: 30px; opacity: 0.5; }}
        
        #by-level-view {{ display: none; }}
        #by-task-view {{ display: block; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 MOMENT Pipeline - Enhanced Logging Dashboard</h1>
            <p class="subtitle">Multi-Level Log Analysis & Monitoring</p>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <!-- Overall Statistics -->
        <div class="stats-grid">
            <div class="stat-card info">
                <div class="stat-icon">ℹ️</div>
                <div class="stat-label">INFO Logs</div>
                <div class="stat-value">{info_total}</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-icon">⚠️</div>
                <div class="stat-label">WARNING Logs</div>
                <div class="stat-value">{warning_total}</div>
            </div>
            <div class="stat-card error">
                <div class="stat-icon">❌</div>
                <div class="stat-label">ERROR Logs</div>
                <div class="stat-value">{error_total}</div>
            </div>
            <div class="stat-card debug">
                <div class="stat-icon">🔍</div>
                <div class="stat-label">DEBUG Logs</div>
                <div class="stat-value">{debug_total}</div>
            </div>
        </div>
        
        <!-- View Toggle -->
        <div class="view-toggle">
            <button class="toggle-btn active" onclick="showView('task')">📋 View by Task</button>
            <button class="toggle-btn" onclick="showView('level')">🎨 View by Log Level</button>
        </div>
'''
    
    # ═══════════════════════════════════════════════════════════
    # VIEW 1: BY TASK (Default)
    # ═══════════════════════════════════════════════════════════
    
    html += '<div id="by-task-view">'
    
    # Sort tasks by a logical order
    task_order = ['Data Acquisition', 'Bias Detection', 'Preprocessing', 
                  'Schema Stats', 'Validation', 'Upload to GCS', 'Notification', 'General']
    
    for task_name in task_order:
        if task_name not in logs_by_task:
            continue
        
        task_logs = logs_by_task[task_name]
        stats = task_stats.get(task_name, {})
        
        html += f'''
        <div class="task-section">
            <div class="task-header">
                <div class="task-title">📌 {task_name}</div>
                <div class="task-counts">
                    {f'<span class="count-badge info">ℹ️ {stats["info"]}</span>' if stats.get('info', 0) > 0 else ''}
                    {f'<span class="count-badge warning">⚠️ {stats["warning"]}</span>' if stats.get('warning', 0) > 0 else ''}
                    {f'<span class="count-badge error">❌ {stats["error"]}</span>' if stats.get('error', 0) > 0 else ''}
                </div>
            </div>
        '''
        
        # Show errors first, then warnings, then info
        for level in ['ERROR', 'CRITICAL', 'WARNING', 'INFO', 'DEBUG']:
            level_logs = task_logs.get(level, [])
            if not level_logs:
                continue
            
            # Limit INFO to 50, show all errors/warnings
            if level == 'INFO':
                display_logs = level_logs[:50]
                if len(level_logs) > 50:
                    html += f'<p style="color: #64748b; margin: 10px 0;">Showing first 50 of {len(level_logs)} INFO logs</p>'
            else:
                display_logs = level_logs  # Show ALL errors/warnings
            
            for log in display_logs:
                level_class = level.lower()
                html += f'''
                <div class="log-entry {level_class}">
                    <span class="log-timestamp">{log['timestamp']}</span>
                    <span class="log-level {level}">{level}</span>
                    {log['message']}
                </div>
                '''
        
        html += '</div>'
    
    html += '</div>'
    
    # ═══════════════════════════════════════════════════════════
    # VIEW 2: BY LOG LEVEL
    # ═══════════════════════════════════════════════════════════
    
    html += '<div id="by-level-view">'
    
    # Filter buttons
    html += '''
    <div class="filter-buttons">
        <button class="filter-btn all active" onclick="filterLogs('all')">All Logs</button>
        <button class="filter-btn info" onclick="filterLogs('info')">INFO Only</button>
        <button class="filter-btn warning" onclick="filterLogs('warning')">WARNING Only</button>
        <button class="filter-btn error" onclick="filterLogs('error')">ERROR Only</button>
        <button class="filter-btn debug" onclick="filterLogs('debug')">DEBUG Only</button>
    </div>
    '''
    
    # Log level sections
    sections = [
        ('error', '🔴 Critical & Error Logs', 
         logs_by_level.get('ERROR', []) + logs_by_level.get('CRITICAL', [])),
        ('warning', '⚠️ Warning Logs', logs_by_level.get('WARNING', [])),
        ('info', 'ℹ️ Info Logs', logs_by_level.get('INFO', [])),
        ('debug', '🔍 Debug Logs', logs_by_level.get('DEBUG', []))
    ]
    
    for section_id, title, logs in sections:
        html += f'<div class="log-section" data-level="{section_id}"><h2>{title}</h2>'
        
        if logs:
            # Show all errors/warnings, limit INFO to 100
            display_logs = logs if section_id in ['error', 'warning'] else logs[:100]
            
            if section_id == 'info' and len(logs) > 100:
                html += f'<p style="color: #64748b; margin-bottom: 10px;">Showing first 100 of {len(logs)} INFO logs</p>'
            
            for log in display_logs:
                html += f'''
                <div class="log-entry {section_id}">
                    <span class="log-timestamp">{log['timestamp']}</span>
                    <span class="log-level {log['level']}">{log['level']}</span>
                    <strong>[{log['task']}]</strong> {log['message']}
                </div>
                '''
        else:
            html += '<div class="empty-state">No logs of this type</div>'
        
        html += '</div>'
    
    html += '</div>'
    
    # JavaScript
    html += '''
    <script>
        function showView(view) {
            const taskView = document.getElementById('by-task-view');
            const levelView = document.getElementById('by-level-view');
            const buttons = document.querySelectorAll('.toggle-btn');
            
            if (view === 'task') {
                taskView.style.display = 'block';
                levelView.style.display = 'none';
                buttons[0].classList.add('active');
                buttons[1].classList.remove('active');
            } else {
                taskView.style.display = 'none';
                levelView.style.display = 'block';
                buttons[0].classList.remove('active');
                buttons[1].classList.add('active');
            }
        }
        
        function filterLogs(level) {
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            const sections = document.querySelectorAll('.log-section');
            if (level === 'all') {
                sections.forEach(s => s.style.display = 'block');
            } else {
                sections.forEach(s => {
                    s.style.display = s.dataset.level === level ? 'block' : 'none';
                });
            }
        }
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Enhanced dashboard generated: {output_path}")
    print(f"\n📊 Overall Statistics:")
    print(f"   INFO: {info_total}")
    print(f"   WARNING: {warning_total}")
    print(f"   ERROR: {error_total}")
    print(f"   DEBUG: {debug_total}")
    print(f"\n📋 By Task:")
    for task, stats in sorted(task_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"   {task}: {stats['total']} total (ℹ️{stats['info']} ⚠️{stats['warning']} ❌{stats['error']})")


def main():
    parser = argparse.ArgumentParser(description='Generate enhanced dashboard with task separation')
    parser.add_argument('--log-file', default='/opt/airflow/logs/team_pipeline_ALL.log',
                       help='Path to log file')
    parser.add_argument('--output', default='team_dashboard_enhanced.html',
                       help='Output HTML file path')
    
    args = parser.parse_args()
    
    print(f"📊 Generating enhanced dashboard from: {args.log_file}")
    
    if not Path(args.log_file).exists():
        print(f"❌ ERROR: Log file not found: {args.log_file}")
        return
    
    # Parse and categorize logs
    logs_by_task, logs_by_level = categorize_logs_by_task_and_level(args.log_file)
    
    # Generate dashboard
    generate_enhanced_dashboard(logs_by_task, logs_by_level, args.output)
    
    print(f"\n🌐 Open the dashboard:")
    print(f"   file://{Path(args.output).absolute()}")


if __name__ == "__main__":
    main()