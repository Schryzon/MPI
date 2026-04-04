import json
import os

def process_notebook(input_file, output_file, is_image=False):
    with open(input_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        # Clear outputs to avoid saving large images and old paths
        if 'outputs' in cell:
            cell['outputs'] = []
            
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell.get('source', []):
                line = line.replace('Windows x64 edition', 'Linux edition')
                new_source.append(line)
            cell['source'] = new_source
            
        elif cell['cell_type'] == 'code':
            new_source = []
            for line in cell.get('source', []):
                # Replace Windows occurrences 
                if is_image:
                    if r'C:\Users\nyoma\scoop\apps\msmpi\10.1.1\mpiexec.exe' in line:
                        line = line.replace(r'"C:\Users\nyoma\scoop\apps\msmpi\10.1.1\mpiexec.exe"', '"mpirun"')
                    line = line.replace('./scripts/parallel_image.exe', './scripts/parallel_image')
                else:
                    if r'C:\Users\nyoma\scoop\apps\msmpi\10.1.1\mpiexec.exe' in line:
                        line = line.replace(r'r"C:\Users\nyoma\scoop\apps\msmpi\10.1.1\mpiexec.exe"', '"mpirun"')
                    line = line.replace('./scripts/sequential_conv.exe', './scripts/sequential_conv')
                    line = line.replace(r'.\scripts\parallel_conv.exe', './scripts/parallel_conv')
                    line = line.replace('shutil.which("mpiexec")', 'shutil.which("mpirun")')
                    
                new_source.append(line)
            cell['source'] = new_source

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

os.chdir(r'c:\Users\nyoma\Downloads\mpyCUDA\CUDA-Speedup-Law')
process_notebook('notebook_windows.ipynb', 'notebook_linux.ipynb', is_image=False)
process_notebook('notebook_image_windows.ipynb', 'notebook_image_linux.ipynb', is_image=True)
