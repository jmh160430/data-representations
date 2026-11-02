"""Classical MDS (STRAIN), using unsupervised numeric CSV features."""
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mds-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/mds-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def read_numeric_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    if not header or not rows or any(len(row) != len(header) for row in rows):
        raise ValueError('Expected a nonempty rectangular CSV with feature headers')
    keep = [i for i, name in enumerate(header)
            if name.strip() and not name.strip().lower().startswith('unnamed:')]
    if not keep:
        raise ValueError('No named numeric features')
    x = np.asarray([[float(row[i]) for i in keep] for row in rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Features must be finite numeric values')
    return x, [header[i] for i in keep]


def parse_q(value):
    q = float(value)
    if math.isnan(q) or q < 1:
        raise argparse.ArgumentTypeError('q must be >= 1, including positive infinity')
    return q


def minkowski_distances(x, q):
    if math.isnan(q) or q < 1:
        raise ValueError('q must be >= 1')
    n = len(x)
    d = np.zeros((n, n))
    # Scale each difference vector temporarily to evaluate the same norm
    # without unnecessary overflow from raising large values to powers.
    # This is numerical evaluation of the distance, not feature scaling.
    for i in range(n - 1):
        delta = np.abs(x[i + 1:] - x[i])
        maximum = delta.max(axis=1)
        if math.isinf(q):
            distances = maximum
        else:
            safe = np.where(maximum == 0, 1.0, maximum)
            distances = maximum * np.sum((delta / safe[:, None]) ** q, axis=1) ** (1 / q)
        d[i, i + 1:] = distances
        d[i + 1:, i] = distances
    if not np.isfinite(d).all():
        raise ValueError('Distances overflowed float64; input magnitude is unsupported')
    return d


def classical_mds(d, k):
    n = len(d)
    if not 1 <= k <= n:
        raise ValueError('dimensions must be between 1 and the observation count')
    a = d ** 2
    # Lecture equation (6), equivalent to -0.5 H A H.
    g = -0.5 * (a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean())
    if not np.isfinite(g).all():
        raise ValueError('Squared distances or Gram matrix overflowed float64')
    eigenvalues, vectors = np.linalg.eigh(g)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    retained = np.maximum(eigenvalues[:k], 0)
    y = vectors[:, :k] * np.sqrt(retained)
    return y, g, eigenvalues, retained, vectors[:, :k]


def run(path, q, k, out):
    x, features = read_numeric_csv(path)
    d = minkowski_distances(x, q)
    y, g, eigenvalues, retained, vectors = classical_mds(d, k)
    n = len(x)
    eps = np.finfo(float).eps
    distance_scale = max(float(d.max()), 1.0)
    gram_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    distance_tol = 100 * eps * n * distance_scale
    gram_tol = 100 * eps * n * gram_scale
    coordinate_tol = 100 * eps * n * max(float(np.max(np.abs(y))), 1.0)
    h = np.eye(n) - np.ones((n, n)) / n
    double_center_error = float(np.max(np.abs(g - (-0.5 * h @ (d ** 2) @ h))))
    embedded_d = minkowski_distances(y, 2)
    pairs = np.triu_indices(n, 1)
    rmse = float(np.sqrt(np.mean((embedded_d[pairs] - d[pairs]) ** 2))) if n > 1 else 0.0
    checks = {
        'distance_symmetric': bool(np.allclose(d, d.T, rtol=0, atol=distance_tol)),
        'distance_diagonal_zero': bool(np.all(np.diag(d) == 0)),
        'distance_nonnegative_finite': bool(np.isfinite(d).all() and (d >= 0).all()),
        'gram_symmetric': bool(np.allclose(g, g.T, rtol=0, atol=gram_tol)),
        'gram_double_centering_equivalence': double_center_error <= gram_tol,
        'shape_correct': y.shape == (n, k),
        'finite_coordinates': bool(np.isfinite(y).all()),
        'embedding_centered': bool(np.max(np.abs(y.mean(axis=0))) <= coordinate_tol),
        'eigenvalues_descending': bool(np.all(np.diff(eigenvalues) <= 0)),
        'retained_eigenvalues_descending_nonnegative': bool((retained >= 0).all() and (np.diff(retained) <= 0).all()),
        'negative_eigenvalues_clipped': bool(np.array_equal(retained, np.maximum(eigenvalues[:k], 0))),
        'eigenpairs_valid': bool(np.max(np.abs(g @ vectors - vectors * eigenvalues[:k])) <= gram_tol),
        'orthonormal_vectors': bool(np.max(np.abs(vectors.T @ vectors - np.eye(k))) <= 100 * eps * n),
    }
    report = dict(input=str(Path(path)), observations=n, features=features, numeric_feature_count=x.shape[1],
        q='inf' if math.isinf(q) else q, dimensions=k,
        libraries={'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        checks=checks, distance_tolerance=distance_tol, gram_tolerance=gram_tol,
        coordinate_tolerance=coordinate_tol, double_centering_max_abs=double_center_error,
        embedding_mean_max_abs=float(np.max(np.abs(y.mean(axis=0)))),
        eigenvalues=eigenvalues.tolist(), retained_eigenvalues=retained.tolist(),
        genuine_negative_eigenvalues=int(np.sum(eigenvalues < -gram_tol)),
        roundoff_negative_eigenvalues=int(np.sum((eigenvalues < 0) & (eigenvalues >= -gram_tol))),
        minimum_eigenvalue=float(eigenvalues.min()), distance_rmse=rmse,
        relative_distance_rmse=rmse / float(np.sqrt(np.mean(d[pairs] ** 2))) if n>1 and np.any(d) else 0.0)
    if q == 2 and k <= min(x.shape):
        xc = x - x.mean(axis=0)
        u, s, vh = np.linalg.svd(xc, full_matrices=False)
        pca = xc @ vh[:k].T
        pca_d = minkowski_distances(pca, 2)
        gram_error = float(np.max(np.abs(y @ y.T - pca @ pca.T)))
        distance_error = float(np.max(np.abs(embedded_d - pca_d)))
        # Near coincident points, distance errors can be larger than Gram errors.
        checks['pca_gram_equivalent'] = gram_error <= gram_tol
        checks['pca_distances_equivalent'] = distance_error <= distance_tol
        report['pca_comparison'] = dict(gram_max_abs=gram_error, distance_max_abs=distance_error,
            full_gram_max_abs=float(np.max(np.abs(g - xc @ xc.T))))
    if not all(checks.values()):
        raise AssertionError(json.dumps(report, indent=2))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tag = 'inf' if math.isinf(q) else format(q, 'g')
    base = out / f'{Path(path).stem}_mds_q{tag}_{k}D'
    csv_path = Path(str(base) + '.csv')
    np.savetxt(csv_path, y, delimiter=',', header=','.join(f'MDS{i+1}' for i in range(k)), comments='')
    saved = np.loadtxt(csv_path, delimiter=',', skiprows=1, ndmin=2)
    assert saved.shape == y.shape and np.array_equal(saved, y)
    if k in (2, 3):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d' if k == 3 else None)
        ax.scatter(*[y[:, i] for i in range(k)], s=25, alpha=0.75, color='tab:blue')
        ax.set(xlabel='MDS1', ylabel='MDS2', title=f'{Path(path).stem}: Classical MDS, q={tag}, {k}D')
        if k == 3:
            ax.set_zlabel('MDS3')
        fig.tight_layout()
        png = base.with_suffix('.png') if '.' not in base.name else Path(str(base) + '.png')
        fig.savefig(png, dpi=160)
        plt.close(fig)
        with Image.open(png) as im:
            assert im.format == 'PNG'
            im.verify()
        with Image.open(png) as im:
            im.load()
        checks['png_verified'] = png.stat().st_size > 0
    report['checks'] = {name: bool(value) for name, value in checks.items()}
    report['all_checks_passed'] = all(checks.values())
    Path(str(base) + '_diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(eigenvalues.tolist()))
    print(f'q={tag}, k={k}, shape={y.shape}, RMSE={rmse:.9g}, genuine negatives={report["genuine_negative_eigenvalues"]}, checks=PASS')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--q', required=True, type=parse_q)
    parser.add_argument('--dimensions', required=True, type=int)
    parser.add_argument('--output-dir', type=Path, default=Path('mds_outputs'))
    args = parser.parse_args()
    try:
        run(args.input, args.q, args.dimensions, args.output_dir)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
