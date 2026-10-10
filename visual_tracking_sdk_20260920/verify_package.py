"""Verify the distributed files with standard Python only."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
failed=[]
for name,item in manifest['files'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failed.append(name);continue
    data=path.read_bytes()
    if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:failed.append(name)
print('FAILED: '+', '.join(failed) if failed else 'OK: all manifest files match')
sys.exit(bool(failed))
