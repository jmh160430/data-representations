"""Run only after source inspection, numeric CSV preparation, and selection recording.

selection.json schema: {"selection_complete": true, "datasets": [
 {"slug": "...", "matrix": "data/stage4/....csv", "sha256": "...",
  "title": "...", "uci_id": 0, "url": "...", "source_files": ["..."]}]}
All paths are relative to the repository root. No downloads or selection occur here.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import numpy as np
import mds

QS = ('1', '1.5', '2', '3', '4', 'inf')


def neighbors(d, count=10):
    # Self excluded. Exact ties resolved by original zero-based row index.
    work = d.copy()
    np.fill_diagonal(work, np.inf)
    return np.argsort(work, axis=1, kind='stable')[:, :count]


def main():
    os.chdir(ROOT)
    manifest_path = ROOT / 'stage4/selection.json'
    selection = json.loads(manifest_path.read_text())
    assert selection.get('selection_complete') is True
    datasets = selection['datasets']
    assert len(datasets) == 3 and len({d['uci_id'] for d in datasets}) == 3
    out = ROOT / 'stage4/results'
    out.mkdir(parents=True, exist_ok=True)
    # Record the completed selection and matrix hashes before launching any q run.
    for ds in datasets:
        assert all(ds.get(key) for key in ('title', 'uci_id', 'url', 'source_files', 'slug', 'matrix', 'sha256'))
        assert hashlib.sha256(Path(ds['matrix']).read_bytes()).hexdigest() == ds['sha256']
        x, _ = mds.read_numeric_csv(ds['matrix'])
        assert x.shape[0] > 10 and x.shape[1] >= 3
    (out / 'selection_used.json').write_text(json.dumps(selection, indent=2)+'\n')
    rows, stability, pca_checks = [], {}, []
    for ds in datasets:
        x, features = mds.read_numeric_csv(ds['matrix'])
        baseline = neighbors(mds.minkowski_distances(x, 2))
        overlaps = []
        for q in QS:
            diagnostics = []
            embeddings = []
            for k in (2, 3):
                command = [sys.executable, 'code/mds.py', '--input', ds['matrix'], '--q', q,
                           '--dimensions', str(k), '--output-dir', str(out)]
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
                run = subprocess.run(command, text=True, capture_output=True, env=env)
                (out / f'{ds["slug"]}_q{q}_{k}D.log').write_text(run.stdout + run.stderr)
                if run.returncode:
                    raise RuntimeError(f'{command} exited {run.returncode}: {run.stderr}')
                base = out / f'{Path(ds["matrix"]).stem}_mds_q{q}_{k}D'
                diag = json.loads(Path(str(base)+'_diagnostics.json').read_text())
                assert diag['all_checks_passed']
                diagnostics.append(diag)
                embeddings.append(np.loadtxt(str(base)+'.csv', delimiter=',', skiprows=1))
            e = np.asarray(diagnostics[0]['eigenvalues'])
            tau = diagnostics[0]['gram_tolerance']
            positive = np.maximum(e, 0)
            total = positive.sum()
            d = mds.minkowski_distances(x, float(q))
            nn = neighbors(d)
            overlap = np.array([len(set(a) & set(b))/10 for a,b in zip(nn,baseline)])
            if q != '2':
                overlaps.append(overlap)
            rows.append(dict(dataset=ds['slug'], q=q, negative_count=int((e < -tau).sum()),
                negative_mass=float(-e[e < -tau].sum()), tolerance=tau,
                positive_mass_2d=float(positive[:2].sum()/total) if total else 0.,
                positive_mass_3d=float(positive[:3].sum()/total) if total else 0.,
                normalized_error_2d=diagnostics[0]['relative_distance_rmse'],
                normalized_error_3d=diagnostics[1]['relative_distance_rmse'],
                mean_10nn_overlap=float(overlap.mean())))
            stability.setdefault(ds['slug'], {})[q] = overlap.tolist()
            if q == '2':
                xc = x - x.mean(axis=0)
                _, _, vh = np.linalg.svd(xc, full_matrices=False)
                for k,y,diag in zip((2,3),embeddings,diagnostics):
                    pca = xc @ vh[:k].T
                    err = float(np.max(np.abs(mds.minkowski_distances(y,2)-mds.minkowski_distances(pca,2))))
                    assert err <= diag['distance_tolerance']
                    pca_checks.append(dict(dataset=ds['slug'], dimensions=k,
                        pairwise_distance_max_abs=err, tolerance=diag['distance_tolerance'], passed=True))
        mean = np.mean(overlaps, axis=0)
        stability[ds['slug']]['mean_nonbaseline_overlap'] = mean.tolist()
        stability[ds['slug']]['most_stable_rows_1based'] = (np.flatnonzero(mean == mean.max())+1).tolist()
        stability[ds['slug']]['least_stable_rows_1based'] = (np.flatnonzero(mean == mean.min())+1).tolist()
        delta = np.abs(x[:,None,:] - x[None,:,:])
        # Feature contributions in unscaled original units, upper-triangle pairs only.
        pairs = np.triu_indices(len(x),1)
        contributions = np.mean(delta[pairs]**2, axis=0)
        stability[ds['slug']]['features'] = features
        stability[ds['slug']]['mean_squared_difference_share'] = (contributions/contributions.sum()).tolist()
    assert len(rows)==18 and len(pca_checks)==6
    with (out/'metrics.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'neighborhoods.json').write_text(json.dumps(stability,indent=2)+'\n')
    (out/'pca_validation.json').write_text(json.dumps(pca_checks,indent=2)+'\n')
    print('Completed 36 CLI runs, 18 metric rows, six independent PCA comparisons.')


if __name__ == '__main__':
    main()
