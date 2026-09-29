"""Choose dataset-specific storage defaults while respecting explicit paths."""

import hashlib
import json
import os
from pathlib import Path
import re


def resolve_dataset_paths(args):
    """Fill unset storage paths; retain legacy defaults for the default Beauty inputs."""
    query_path = Path(args.query_csv).resolve()
    meta_path = Path(args.filtered_meta_jsonl).resolve()
    if (query_path == Path('data/amazon_beauty/query_data1.csv').resolve()
            and meta_path == Path('data/amazon_beauty/meta_Beauty.filtered.jsonl').resolve()):
        defaults = {
            'cache_dir': 'processed/beauty_cache',
            'output_dir': 'processed/beauty_unified_outputs',
            'global_db': 'processed/beauty_global_item_features.db',
            'history_db': 'processed/beauty_user_history.db',
            'collaborative_db_path': 'processed/beauty_collaborative_signal.db',
        }
    else:
        # Both inputs participate: different query splits or metadata snapshots
        # must not accidentally reuse another evaluation's user output files.
        identity = json.dumps([os.path.normcase(str(query_path)), os.path.normcase(str(meta_path))])
        digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]
        label = re.sub(r'[^A-Za-z0-9_-]+', '-', query_path.parent.name).strip('-') or 'dataset'
        root = Path('processed') / f'{label}-{digest}'
        defaults = {
            'cache_dir': str(root / 'cache'),
            'output_dir': str(root / 'outputs'),
            'global_db': str(root / 'global_item_features.db'),
            'history_db': str(root / 'user_history.db'),
            'collaborative_db_path': str(root / 'collaborative_signal.db'),
        }
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    return args
