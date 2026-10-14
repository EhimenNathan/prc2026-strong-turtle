"""Verify the source tree before a public GitHub push; never prints secret values."""
import ast, re, sys
from pathlib import Path

root = Path(__file__).resolve().parent
issues = []
for path in root.rglob('*'):
    if not path.is_file() or any(part in ['.git','work','data','submission','.local_checks','__pycache__'] for part in path.relative_to(root).parts):
        continue
    if path.suffix.lower() in ['.npz','.npy','.parquet','.zip','.pkl','.bin','.csv'] and path.parent.name != 'docs':
        issues.append(f'Unexpected data artifact: {path.relative_to(root)}')
    if path.stat().st_size > 5*1024*1024:
        issues.append(f'Large file: {path.relative_to(root)}')
    if path.suffix.lower() not in ['.py','.md','.txt','.json','.sh','.ps1']:
        continue
    text = path.read_text(encoding='utf8')
    if path.suffix == '.py': ast.parse(text)
    # Construct patterns from pieces so the scanner does not match itself.
    personal = 'C:' + r'[/\\]Users[/\\]'
    secret = r'(?i)(?:access' + r'[_-]?key|secret' + r'[_-]?key|password)\s*[:=]\s*[\"\'][A-Za-z0-9+/]{12,}'
    if re.search(personal,text): issues.append(f'Personal path: {path.relative_to(root)}')
    if re.search(secret,text): issues.append(f'Possible embedded credential: {path.relative_to(root)}')
if not (root/'LICENSE').is_file(): issues.append('Missing GPLv3 LICENSE')
if issues:
    print('\n'.join(issues))
    sys.exit(1)
print('Source checks passed: Python syntax, personal paths, obvious embedded credentials and excluded artifacts.')
