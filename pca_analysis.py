"""Run: python3 pca_analysis.py DATASET [--output-dir DIRECTORY]."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pca-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def run(stem, output_dir):
    path = Path(stem + '.csv')
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    if not header or not rows or any(len(row) != len(header) for row in rows):
        raise ValueError('CSV must have a header and nonempty rectangular observations')
    # Drop only explicitly unnamed export-index columns; never infer an index
    # merely from sequential values in a named numeric feature.
    keep = [i for i, name in enumerate(header)
            if name.strip() and not name.strip().lower().startswith('unnamed:')]
    x = np.array([[float(row[i]) for i in keep] for row in rows], dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 3 or not np.isfinite(x).all():
        raise ValueError('Need at least three observations and three finite numeric features')
    original = x.copy()
    mu = x.mean(axis=0)
    centered = x - mu
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {'dataset': stem, 'input': str(path.resolve()), 'shape': list(x.shape),
              'excluded_index_columns': [header[i] for i in range(len(header)) if i not in keep],
              'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
              'routine': 'numpy.linalg.svd(full_matrices=False)', 'checks': []}
    eps = np.finfo(np.float64).eps
    for variant, matrix in [('upca', x), ('cpca', centered)]:
        # Two separate SVD calls: each basis is fitted to its own input.
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        scale = max(float(np.max(np.abs(matrix))), float(s[0]), np.finfo(float).tiny)
        atol = 100 * eps * max(matrix.shape) * scale
        orth_tol = 100 * eps * max(matrix.shape)
        assert np.all(s >= 0) and np.all(s[:-1] >= s[1:])
        reconstruction_error = float(np.max(np.abs(matrix - (u * s) @ vh)))
        assert reconstruction_error <= atol
        for k in (2, 3):
            v = vh[:k].T
            y = matrix @ v
            discrepancy = float(np.max(np.abs(y - u[:, :k] * s[:k])))
            orth_error = float(np.max(np.abs(v.T @ v - np.eye(k))))
            mean_error = float(np.max(np.abs(y.mean(axis=0))))
            assert discrepancy <= atol
            assert orth_error <= orth_tol
            assert y.shape == (x.shape[0], k) and np.isfinite(y).all()
            if variant == 'cpca':
                # Centering roundoff depends on original data scale, too.
                mean_tol = 100 * eps * max(x.shape) * max(float(np.max(np.abs(x))), scale)
                assert mean_error <= mean_tol
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d' if k == 3 else None)
            ax.scatter(*[y[:, i] for i in range(k)], s=25, alpha=0.75, color='tab:blue')
            ax.set(xlabel='PC1', ylabel='PC2', title=f'{stem}: {"Uncentered" if variant == "upca" else "Centered"} PCA ({k}D)')
            if k == 3:
                ax.set_zlabel('PC3')
            fig.tight_layout()
            filename = output_dir / f'{Path(stem).name}_{variant}_{k}D.png'
            fig.savefig(filename, dpi=160)
            plt.close(fig)
            assert filename.is_file() and filename.stat().st_size > 0
            with Image.open(filename) as im:
                assert im.format == 'PNG'
                im.verify()
            with Image.open(filename) as im:
                im.load()
                image_size = im.size
            report['checks'].append(dict(variant=variant, dimensions=list(y.shape),
                equivalence_max_abs=discrepancy, equivalence_tolerance=atol,
                orthonormality_max_abs=orth_error, orthonormality_tolerance=orth_tol,
                singular_values=s.tolist(), singular_order_passed=True,
                reconstruction_max_abs=reconstruction_error,
                embedding_mean_max_abs=mean_error, centered_mean_tolerance=mean_tol if variant=='cpca' else None,
                finite=True, png=str(filename.resolve()), png_size=list(image_size), png_verified=True))
    assert np.array_equal(x, original)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    report['all_checks_passed'] = True
    (output_dir / f'{Path(stem).name}_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='CSV stem, preserving case')
    parser.add_argument('--output-dir', type=Path, default=Path('pca_outputs'))
    args = parser.parse_args()
    run(args.dataset, args.output_dir)
