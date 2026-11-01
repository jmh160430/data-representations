# Stage 3: Classical MDS results

Input: `Iris.csv`, 150 observations, 4 finite numeric features. No labels, scaling, normalization, or changes to the input were used.

Libraries: NumPy 2.3.5, Matplotlib 3.10.8, Pillow (PNG validation). CSV input uses Python csv.

Reference: supplied MDS.pdf, STRAIN, equation (6) and steps 2–3 (pages 3–4). The row-mean formula implements G = -0.5 H(D²)H; an explicit H multiplication was independently checked. numpy.linalg.eigh provides the symmetric EVD. Eigenpairs are sorted descending; every negative eigenvalue is clipped to zero before square roots. Coordinates are rows of V_k sqrt(Lambda_k), the transpose of the notes’ column-observation convention.

Minkowski distances are implemented directly, mirroring the upper triangle and using max absolute difference for infinity. Finite powers are evaluated with per-pair numerical stabilization; the original features are never rescaled.

Reproduce: `python3 mds.py --input Iris.csv --q 2 --dimensions 2` (also supports `--q inf --dimensions 3`). Output directory defaults to mds_outputs.

## Verification

All 12 runs passed: symmetric nonnegative finite distances, zero diagonal, symmetric Gram matrix, explicit double-centering equivalence, finite n×k coordinates, centered embeddings, descending eigenvalues, negative clipping, eigenpair residuals, orthonormality, CSV round trips, and PNG decoding. Maximum embedding absolute column mean: 5.68434e-16. Float64 tolerances are 100*eps*n*scale (per-run values in JSON).

Independent implementation checks compared all six distance variants with numpy.linalg.norm and recovered the lecture’s 3–4–5 triangle distances. CLI rejected q=0, q=nan, k=0 and k=151.

| q | k | Pairwise distance RMSE | Genuine negative eigenvalues |
|---|---|---:|---:|
| 1 | 2 | 0.292869806 | 90 |
| 1 | 3 | 0.30204256 | 90 |
| 1.5 | 2 | 0.158140551 | 94 |
| 1.5 | 3 | 0.0647200359 | 94 |
| 2 | 2 | 0.126442278 | 0 |
| 2 | 3 | 0.0369441926 | 0 |
| 3 | 2 | 0.115472954 | 55 |
| 3 | 3 | 0.0588617533 | 55 |
| 4 | 2 | 0.128479064 | 62 |
| 4 | 3 | 0.084186951 | 62 |
| inf | 2 | 0.193006846 | 72 |
| inf | 3 | 0.110743597 | 72 |

RMSE uses unique unordered observation pairs. These nonzero errors are genuine truncation/non-Euclidean approximation errors, not numerical rounding. STRAIN optimizes Gram approximation, so distance RMSE need not decrease when adding a dimension (as q=1 demonstrates).

## Euclidean MDS versus centered PCA

| k | Maximum Gram difference | Maximum pairwise-distance difference |
|---|---:|---:|
| 2 | 3.90799e-14 | 7.99361e-15 |
| 3 | 3.90799e-14 | 2.34812e-14 |

Equivalence confirmed using a separate numpy.linalg.svd on mean-subtracted original features, without scaling. Differences are floating-point roundoff. The q=2 minimum eigenvalue is -1.76034e-13, consistent with roundoff; other q values have genuine negative eigenvalues, documented per run and clipped as required.

## Generated files

Each run saves a PNG, embedding CSV, and diagnostic JSON. Iris_summary.json combines all diagnostics.

- `Iris_mds_q1.5_2D.png`
- `Iris_mds_q1.5_2D_diagnostics.json`
- `Iris_mds_q1.5_2D_embedding.csv`
- `Iris_mds_q1.5_3D.png`
- `Iris_mds_q1.5_3D_diagnostics.json`
- `Iris_mds_q1.5_3D_embedding.csv`
- `Iris_mds_q1_2D.png`
- `Iris_mds_q1_2D_diagnostics.json`
- `Iris_mds_q1_2D_embedding.csv`
- `Iris_mds_q1_3D.png`
- `Iris_mds_q1_3D_diagnostics.json`
- `Iris_mds_q1_3D_embedding.csv`
- `Iris_mds_q2_2D.png`
- `Iris_mds_q2_2D_diagnostics.json`
- `Iris_mds_q2_2D_embedding.csv`
- `Iris_mds_q2_3D.png`
- `Iris_mds_q2_3D_diagnostics.json`
- `Iris_mds_q2_3D_embedding.csv`
- `Iris_mds_q3_2D.png`
- `Iris_mds_q3_2D_diagnostics.json`
- `Iris_mds_q3_2D_embedding.csv`
- `Iris_mds_q3_3D.png`
- `Iris_mds_q3_3D_diagnostics.json`
- `Iris_mds_q3_3D_embedding.csv`
- `Iris_mds_q4_2D.png`
- `Iris_mds_q4_2D_diagnostics.json`
- `Iris_mds_q4_2D_embedding.csv`
- `Iris_mds_q4_3D.png`
- `Iris_mds_q4_3D_diagnostics.json`
- `Iris_mds_q4_3D_embedding.csv`
- `Iris_mds_qinf_2D.png`
- `Iris_mds_qinf_2D_diagnostics.json`
- `Iris_mds_qinf_2D_embedding.csv`
- `Iris_mds_qinf_3D.png`
- `Iris_mds_qinf_3D_diagnostics.json`
- `Iris_mds_qinf_3D_embedding.csv`
- `Iris_summary.json`

## Limitations

Dense distance/Gram storage is O(n²); eigendecomposition is O(n³). Numeric feature selection cannot infer supervised meaning from an arbitrary numeric column: supply feature-only CSVs. Only unnamed export-index columns are excluded. Dimensions 1 through n are supported; plots are produced only for 2D/3D. Extreme finite inputs that overflow float64 are rejected. Tiny positive eigenvalues are retained as specified; negative values are clipped, with genuine and roundoff negatives counted separately.

Input and existing PCA files were SHA-256 checked unchanged.
