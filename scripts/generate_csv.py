#!/usr/bin/env python3
"""Generate per-dataset CSV files from fair-SPLIT result JSON files.

Reads all_results_fair.json and all_results_nofair.json, merges the data,
and outputs one CSV per dataset under fair-SPLIT/results/csv/.

CSV structure:
  - Columns (header): model, training_time_s, test_accuracy, majority_baseline,
    plus {feature}_sp, {feature}_di, {feature}_eo for each fairness feature.
  - Rows: cart, split-greedy, split-optimal, resplit, plus calibrated variants
    where available (named "model (calibrated)").
"""

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_feature_names(
    fair_data: Dict[str, Any],
    dataset: str,
) -> List[str]:
    """Return ordered fairness feature names for a dataset.

    Inspects any model that has fairness data for the given dataset.
    """
    for model_name in ['cart', 'split-greedy', 'split-optimal', 'resplit']:
        model_data = fair_data.get(model_name, {})
        ds = model_data.get(dataset)
        if ds and 'fairness' in ds:
            return list(ds['fairness'].keys())
    return []


def _build_columns(feature_names: List[str]) -> List[str]:
    """Build the full header column list."""
    base = ['model', 'training_time_s', 'test_accuracy', 'majority_baseline']
    for feat in feature_names:
        base.extend([f'{feat}_sp', f'{feat}_di', f'{feat}_eo'])
    return base


def _extract_metrics(
    ds_data: Dict[str, Any],
    feature_names: List[str],
    calibrated: bool,
) -> Dict[str, Any]:
    """Extract a flat dict of metrics from a dataset entry.

    Args:
        ds_data: The per-dataset dict from the JSON.
        feature_names: Ordered list of fairness feature names.
        calibrated: If True, use calibrated_* values.

    Returns:
        Flat dict with keys matching CSV columns (minus 'model').
    """
    row: Dict[str, Any] = {
        'training_time_s': ds_data.get('training_time_s', ''),
    }

    if calibrated and 'fairness' in ds_data:
        # Use calibrated_acc as test_accuracy.
        first_feat = feature_names[0] if feature_names else None
        if first_feat and first_feat in ds_data['fairness']:
            row['test_accuracy'] = ds_data['fairness'][first_feat].get(
                'calibrated_acc', ''
            )
        else:
            row['test_accuracy'] = ds_data.get('test_accuracy', '')
    else:
        row['test_accuracy'] = ds_data.get('test_accuracy', '')

    row['majority_baseline'] = ds_data.get('majority_baseline', '')

    fairness = ds_data.get('fairness', {})
    for feat in feature_names:
        feat_data = fairness.get(feat, {})
        if calibrated:
            row[f'{feat}_sp'] = feat_data.get('calibrated_sp_diff', '')
            row[f'{feat}_di'] = feat_data.get('calibrated_di_ratio', '')
            row[f'{feat}_eo'] = feat_data.get('calibrated_eo_diff', '')
        else:
            row[f'{feat}_sp'] = feat_data.get('sp_diff', '')
            row[f'{feat}_di'] = feat_data.get('di_ratio', '')
            row[f'{feat}_eo'] = feat_data.get('eo_diff', '')
    return row


def _has_calibrated_data(
    ds_data: Dict[str, Any],
    feature_names: List[str],
) -> bool:
    """Check whether a dataset entry contains calibrated fairness data."""
    if not feature_names or 'fairness' not in ds_data:
        return False
    first_feat = feature_names[0]
    feat_data = ds_data['fairness'].get(first_feat, {})
    return 'calibrated_sp_diff' in feat_data


def _get_merged_model_data(
    fair_data: Dict[str, Any],
    nofair_data: Dict[str, Any],
    model_name: str,
    dataset: str,
) -> Optional[Dict[str, Any]]:
    """Get dataset data for a model, falling back to nofair if missing in fair."""
    model_fair = fair_data.get(model_name, {})
    if dataset in model_fair:
        return model_fair[dataset]
    # Fallback to nofair.
    model_nofair = nofair_data.get(model_name, {})
    return model_nofair.get(dataset)


def _generate_dataset_csv(
    dataset: str,
    feature_names: List[str],
    model_names: List[str],
    fair_data: Dict[str, Any],
    nofair_data: Dict[str, Any],
    output_dir: str,
) -> None:
    """Generate a single CSV file for one dataset."""
    columns = _build_columns(feature_names)
    output_path = os.path.join(output_dir, f'{dataset}.csv')

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for model_name in model_names:
            ds_data = _get_merged_model_data(
                fair_data, nofair_data, model_name, dataset
            )
            if ds_data is None:
                continue

            # Standard (non-calibrated) row.
            row = _extract_metrics(ds_data, feature_names, calibrated=False)
            row['model'] = model_name
            writer.writerow(row)

            # Calibrated row (only from fair data).
            fair_model = fair_data.get(model_name, {})
            fair_ds = fair_model.get(dataset)
            if fair_ds and _has_calibrated_data(fair_ds, feature_names):
                cal_row = _extract_metrics(fair_ds, feature_names, calibrated=True)
                cal_row['model'] = f'{model_name} (calibrated)'
                writer.writerow(cal_row)

    print(f'  Generated: {output_path}')


def main() -> None:
    """Main entry point: load JSON, generate CSV per dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    output_dir = os.path.join(results_dir, 'csv')

    fair_path = os.path.join(results_dir, 'all_results_fair.json')
    nofair_path = os.path.join(results_dir, 'all_results_nofair.json')

    print(f'Loading {fair_path} ...')
    fair_data = _load_json(fair_path)
    print(f'Loading {nofair_path} ...')
    nofair_data = _load_json(nofair_path)

    os.makedirs(output_dir, exist_ok=True)

    model_names = ['cart', 'split-greedy', 'split-optimal', 'resplit']

    # Collect all unique dataset names from both files.
    all_datasets: List[str] = []
    seen = set()
    for data in (fair_data, nofair_data):
        for model_name in model_names:
            for ds_name in data.get(model_name, {}):
                if ds_name not in seen:
                    all_datasets.append(ds_name)
                    seen.add(ds_name)

    print(f'Found {len(all_datasets)} datasets across all models.')
    print(f'Output directory: {output_dir}\n')

    for dataset in all_datasets:
        feature_names = _get_feature_names(fair_data, dataset)
        if feature_names:
            print(
                f'Dataset: {dataset}  '
                f'(features: {", ".join(feature_names)})'
            )
        else:
            print(f'Dataset: {dataset}  (no fairness features)')
        _generate_dataset_csv(
            dataset=dataset,
            feature_names=feature_names,
            model_names=model_names,
            fair_data=fair_data,
            nofair_data=nofair_data,
            output_dir=output_dir,
        )

    print(f'\nDone. {len(all_datasets)} CSV files written to {output_dir}')


if __name__ == '__main__':
    main()
