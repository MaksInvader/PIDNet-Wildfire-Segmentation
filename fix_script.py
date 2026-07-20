import json

with open('c:/Users/PC/Documents/Researchfr/PIDNet/pidnet_finetuning_RGCr.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            new_line = line.replace("\\'mode\\'", "'mode'")
            new_line = new_line.replace("\\'exhaustive\\'", "'exhaustive'")
            new_line = new_line.replace("\\'total_tiles\\'", "'total_tiles'")
            
            # cell 21 line 269: unexpected indent
            if 'if (precision[cls] + recall[cls]) > 0 else 0.0' in new_line and line.startswith('                      if'):
                pass
                
            if new_line != line:
                cell['source'][i] = new_line
                
# Let's also fix cell 21 line 268-269 unexpected indent
# Basically, line 268 doesn't have a backslash at the end.
for cell in nb['cells']:
    if cell.get('id') == '67035256': # cell 21
        for i, line in enumerate(cell['source']):
            if 'f1[cls] = (2.0 * precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) \n' == line:
                # Add backslash to the end!
                cell['source'][i] = '            f1[cls] = (2.0 * precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) \\\n'

with open('c:/Users/PC/Documents/Researchfr/PIDNet/pidnet_finetuning_RGCr.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done fixing notebook!')
