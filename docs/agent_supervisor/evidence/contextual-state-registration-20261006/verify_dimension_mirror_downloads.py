"""Compare fresh immutable-revision HF CLI mirror downloads with original states."""
import json
from pathlib import Path

from custody import capture, read, require, write

OUT = Path(__file__).resolve().parent


def main():
    preparation_pin = capture(OUT / 'registration-preparation.json')
    preparation = json.loads(read(preparation_pin))
    plan = json.loads(read(preparation['import_plan_pin']))
    summary_pin = capture(OUT / 'dimension-mirrors/publication-summary.json')
    summary = json.loads(read(summary_pin))
    require(summary['completed'] is True, 'completed mirror publication required')
    outcomes = []
    for record in plan['models']:
        dimension = record['model_metadata']['huggingface_config']['ir_checkpoint']['dimension']
        receipt_pin = capture(OUT / f'dimension-mirrors/{dimension}d-publication-receipt.json')
        receipt = json.loads(read(receipt_pin))
        require(receipt['repository_id'] == f'Publicus/legal-ir-autoencoder-{dimension}d'
                and receipt['files_verified'] is True, 'exact dimension mirror receipt required')
        entries = [row for row in receipt['files'] if row['sha256'] == record['checkpoint_pin']['sha256']]
        require(len(entries) == 1, 'one complete original checkpoint mirror required')
        entry = entries[0]
        downloaded_pin = capture(OUT / 'mirror-download' / str(dimension) / entry['path_in_repo'])
        original_pin = record['checkpoint_pin']
        require(capture(original_pin['path']) == original_pin
                and (downloaded_pin['bytes'], downloaded_pin['sha256'])
                    == (original_pin['bytes'], original_pin['sha256']), 'fresh mirror bytes differ from original')
        outcomes.append({'dimension': dimension, 'repository_id': receipt['repository_id'],
            'revision': receipt['revision'], 'path_in_repo': entry['path_in_repo'],
            'original_checkpoint_pin': original_pin, 'fresh_download_pin': downloaded_pin,
            'publication_receipt_pin': receipt_pin, 'remote_bytes_equal_original': True})
    require(capture(summary_pin['path']) == summary_pin, 'mirror summary changed')
    result = {'schema': 'dimension-mirror-fresh-download-observation/v1', 'completed': True,
        'publication_summary_pin': summary_pin, 'preparation_pin': preparation_pin,
        'observations': outcomes, 'exact_original_state_matches': 2,
        'method': 'Fresh HF CLI downloads pinned to the immutable full revision, then bounded local SHA256 comparison with original retained states.',
        'model_loaded': False, 'training_executed': False, 'inference_executed': False,
        'database_mutated': False, 'new_hub_upload': False, 'runtime_admitted': False, 'proof_authority': False}
    print(json.dumps({'receipt_pin': write(OUT / 'dimension-mirror-fresh-downloads.json', result)}))


if __name__ == '__main__':
    main()
