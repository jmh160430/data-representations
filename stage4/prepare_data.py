"""Prepare feature-only matrices from preserved official UCI archives; no q calculations."""
from pathlib import Path
import hashlib,json,zipfile
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 datasets=[]
 specs=[('wine',109,'Wine','wine.data',slice(1,None),['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280_od315','proline'],178,','),('seeds',236,'Seeds','seeds_dataset.txt',slice(None,7),['area','perimeter','compactness','kernel_length','kernel_width','asymmetry_coefficient','groove_length'],210,None),('glass',42,'Glass Identification','glass.data',slice(1,10),['refractive_index','sodium','magnesium','aluminum','silicon','potassium','calcium','barium','iron'],214,',')]
 for slug,uid,title,source,columns,features,n,delimiter in specs:
  archive=ROOT/f'data/stage4/sources/{slug}.zip'
  with zipfile.ZipFile(archive) as z:
   text=z.read(source).decode()
   for name in z.namelist():
    if name.endswith('.names'):
     (archive.parent/f'{slug}.names').write_bytes(z.read(name))
  rows=[line.split(delimiter) for line in text.splitlines() if line.strip()]
  raw=np.array(rows,dtype=float)
  x=raw[:,columns]
  assert x.shape==(n,len(features)) and np.isfinite(x).all()
  path=ROOT/f'data/stage4/{slug}.csv'
  np.savetxt(path,x,delimiter=',',header=','.join(features),comments='')
  datasets.append(dict(slug=slug,uci_id=uid,title=title,url=f'https://archive.ics.uci.edu/dataset/{uid}/'+('glass+identification' if slug=='glass' else slug),source_files=[source],archive=f'data/stage4/sources/{slug}.zip',matrix=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),observations=n,numeric_features=len(features),feature_names=features,excluded={'wine':'column 1 cultivar class','seeds':'column 8 variety code; official prose lists exactly seven geometric features, despite incomplete API variable metadata','glass':'column 1 ID and column 11 glass class'}[slug],missing='Official documentation: no missing values; no missing marker specified. All rows parse as finite numbers. No imputation or row removal. Zeros retained.'))
 selection=dict(selection_complete=True,recorded_at_utc=datetime.now(timezone.utc).isoformat(),selection_preceded_q_experiments=True,rationale='Chosen before experiments for contrasts in chemical scale imbalance (Wine), correlated geometry (Seeds), and oxide composition plus refractive index (Glass). Exactly three; no replacements. All n<=214, feature counts>=7; dense O(n^2) storage and O(n^3) EVD feasible with one BLAS thread.',datasets=datasets)
 (ROOT/'stage4/selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 print([(d['slug'],d['observations'],d['numeric_features']) for d in datasets])
if __name__=='__main__': main()
