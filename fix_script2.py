import json
import ast

with open('c:/Users/PC/Documents/Researchfr/PIDNet/pidnet_finetuning_RGCr.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell 21 line 295 (index 294)
cell_21 = nb['cells'][21]
if cell_21['source'][294] == "            class_name = self.hparams.id2label.get(cls, f'class_{cls}') \n":
    cell_21['source'][294] = "            class_name = self.hparams.id2label.get(cls, f'class_{cls}') \\\n"

with open('c:/Users/PC/Documents/Researchfr/PIDNet/pidnet_finetuning_RGCr.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

# Verify again
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        try:
            ast.parse(source, filename=f'cell_{i}')
        except SyntaxError as e:
            if not source.startswith('%pip'):
                print(f'Syntax error in cell {i}:')
                print(e)
                print(f'Line {e.lineno}: {e.text}')

print('Done fixing!')
