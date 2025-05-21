#!/usr/bin/env python3
"""
Fix formatting issues in Jupyter notebooks.
This script manually fixes specific formatting issues in the model optimization workshop notebooks.
"""

import json
import sys

def fix_notebook(notebook_path):
    """Fix formatting issues in a notebook."""
    print(f"Fixing formatting in {notebook_path}")
    
    # Read the notebook file
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            # Join all source lines to check for issues
            full_source = ''.join(cell['source'])
            
            # Check for import formatting issues
            if 'import' in full_source and len(full_source.split('\n')) <= 3 and len(full_source) > 100:
                print(f"  Fixing import formatting in cell {i}")
                # Split imports onto separate lines
                lines = []
                for line in full_source.split('import'):
                    if not line.strip():
                        continue
                    if line.startswith(' '):
                        lines.append(f"import{line}")
                    else:
                        # This is likely the first part before any import
                        lines.append(line)
                        
                # Clean up and format properly
                formatted_imports = []
                for line in lines:
                    if 'import' in line:
                        parts = line.split()
                        if len(parts) > 1:
                            formatted_imports.append(f"import {parts[0]}\n")
                        else:
                            formatted_imports.append(f"{line}\n")
                    else:
                        formatted_imports.append(f"{line}\n")
                
                # Update the cell source
                cell['source'] = formatted_imports
            
            # Check for try-except formatting issues
            elif 'try:' in full_source and 'except' in full_source and '\ntry:' not in full_source:
                print(f"  Fixing try-except formatting in cell {i}")
                # Format try-except blocks properly
                lines = full_source.split('\n')
                formatted_lines = []
                for line in lines:
                    if 'try:' in line and not line.strip() == 'try:':
                        parts = line.split('try:')
                        formatted_lines.append(f"{parts[0]}\n")
                        formatted_lines.append("try:\n")
                        if len(parts) > 1 and parts[1].strip():
                            formatted_lines.append(f"    {parts[1].strip()}\n")
                    elif 'except' in line and not line.strip().startswith('except'):
                        parts = line.split('except')
                        formatted_lines.append(f"{parts[0]}\n")
                        formatted_lines.append(f"except{parts[1]}\n")
                    else:
                        formatted_lines.append(f"{line}\n")
                
                # Update the cell source
                cell['source'] = formatted_lines
            
            # Check for long line formatting issues
            elif len(full_source) > 500 and len(full_source.split('\n')) < 10:
                print(f"  Fixing long line formatting in cell {i}")
                # Try to break up long lines
                lines = []
                current_line = ""
                indent_level = 0
                
                # Process character by character
                for char in full_source:
                    current_line += char
                    
                    # Track indentation
                    if char == '{' or char == '[' or char == '(':
                        indent_level += 1
                    elif char == '}' or char == ']' or char == ')':
                        indent_level -= 1
                    
                    # Break at appropriate points
                    if char == '\n':
                        lines.append(current_line)
                        current_line = ""
                    elif char == ':' and indent_level == 0:
                        lines.append(current_line + "\n")
                        current_line = ""
                    elif char == ',' and indent_level <= 1 and len(current_line) > 80:
                        lines.append(current_line + "\n")
                        current_line = ""
                
                # Add any remaining content
                if current_line:
                    lines.append(current_line)
                
                # Update the cell source
                cell['source'] = lines
    
    # Write the updated notebook back to file
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Fixed {notebook_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for notebook_path in sys.argv[1:]:
            fix_notebook(notebook_path)
    else:
        # Default notebooks to fix
        notebooks = [
            '02_baseline_evaluation.ipynb',
            '04_pruning.ipynb',
            '05_knowledge_distillation.ipynb'
        ]
        for notebook in notebooks:
            fix_notebook(notebook)
