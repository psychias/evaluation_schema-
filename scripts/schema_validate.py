"""Schema validation for all paper records."""
import json
from pathlib import Path
from jsonschema import validate, ValidationError

_ROOT = Path(__file__).resolve().parent.parent

schema_path = _ROOT / 'shared_task_submission' / 'schema' / 'eval.schema.json'
if not schema_path.exists():
    schema_path = _ROOT / 'eval.schema.json'

schema = json.loads(schema_path.read_text())
fails = 0
checked = 0

for f in (_ROOT / 'data').rglob('*.json'):
    if '/papers_' not in str(f):
        continue
    try:
        validate(json.loads(f.read_text()), schema)
    except ValidationError as e:
        fails += 1
        print(str(f).split('every_eval_ever_restored/')[-1], '->', e.message[:100])
    checked += 1

print(f'Schema failures: {fails} / {checked} checked')
