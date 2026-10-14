"""Small, meaningful regression checks for final submission gates."""
import ast, importlib.util, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in ROOT.rglob('*.py'):
    ast.parse(path.read_text(encoding='utf8'))
spec = importlib.util.spec_from_file_location('assembly_checks',ROOT / 'pipeline/assembly_checks.py')
checks = importlib.util.module_from_spec(spec); spec.loader.exec_module(checks)
ref = pd.DataFrame({'apt':['A','A','B']},index=pd.Index([1,2,3],name='mvt_id'))
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    complete = tmp / 'complete.npz'
    np.savez(complete,A_rank_ids=[2,1],A_rank_p=[20.,10.],B_rank_ids=[3],B_rank_p=[30.])
    values = checks.read_parts(complete,['A','B'],ref).reindex(ref.index)
    assert values.tolist() == [10.,20.,30.]
    cases = [dict(A_rank_ids=[1,2],A_rank_p=[10.,20.]),dict(A_rank_ids=[1,1],A_rank_p=[10.,20.],B_rank_ids=[3],B_rank_p=[30.]),dict(A_rank_ids=[1,2],A_rank_p=[10.,np.inf],B_rank_ids=[3],B_rank_p=[30.])]
    for case in cases:
        path = tmp / 'bad.npz'; np.savez(path,**case)
        try: checks.read_parts(path,['A','B'],ref)
        except ValueError: pass
        else: raise AssertionError('Invalid model accepted')
    for ids,c in [([1,1],[1.,2.]),([4],[1.]),([1],[np.nan]),([1.5],[1.]),([float('nan')],[1.])]:
        path = tmp / 'bad.parquet'
        pd.DataFrame({'mvt_id':ids,'c':c}).to_parquet(path,index=False)
        try: checks.read_correction(path,ref)
        except ValueError: pass
        else: raise AssertionError('Invalid correction accepted')
    try: checks.read_correction(tmp / 'missing.parquet',ref)
    except ValueError: pass
    else: raise AssertionError('Missing correction accepted')
    path = tmp / 'integer_float_ids.parquet'
    pd.DataFrame({'mvt_id':[1.0,2.0],'c':[1.,2.]}).to_parquet(path,index=False)
    assert checks.read_correction(path,ref).tolist() == [1.,2.,0.]
print('PASS: Python syntax, portable source paths, ID alignment, missing airports, duplicate IDs, nonfinite predictions and missing corrections.')

# Exercise archive-selection helpers without running their download entry point.
tree = ast.parse((ROOT / 'kaggle/adsb_extract.py').read_text(encoding='utf8'))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ['preferred_urls','other_replica']]
prefix = 'https://github.com/adsblol/globe_history_2026/releases/download/v2026.06.06-planes-readsb-staging-0/v2026.06.06-planes-readsb-staging-0'
namespace = {'_pref': {'2026': [','.join([prefix+'.tar',prefix+'.tar.ab',prefix+'.tar.aa'])]}}
exec(compile(ast.Module(body=functions, type_ignores=[]), 'archive_helpers', 'exec'), namespace)
assert namespace['preferred_urls']('2026-06-06') == [prefix+'.tar.aa',prefix+'.tar.ab']
alternate = namespace['other_replica']([prefix+'.tar'])
assert '-prod-' in alternate[0] and '.tar.tar' not in alternate[0]
print('PASS: alternative archive representations are not concatenated; standalone-tar replica naming is valid.')
