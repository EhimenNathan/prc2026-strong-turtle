"""Merge ground-track archives, preferring the largest copy of each day.

Usage: python merge_adsb_archives.py OUTPUT.zip INPUT.zip [INPUT.zip ...]
This operates on derived daily ground files, not raw globe-history archives.
"""
import argparse, json, zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('inputs',nargs='+',type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Output exists; choose a new output path')
    archives = [zipfile.ZipFile(path) for path in args.inputs]
    preferred = {}
    try:
        for archive in archives:
            for entry in archive.infolist():
                if not entry.filename.startswith('ground_') or not entry.filename.endswith('.parquet'): continue
                old = preferred.get(entry.filename)
                if old is None or entry.file_size > old[1].file_size:
                    preferred[entry.filename] = (archive,entry)
        if not preferred: raise ValueError('No daily ground-track files found')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_STORED) as output:
            for name,(archive,entry) in sorted(preferred.items()):
                output.writestr(name,archive.read(entry))
        print(json.dumps({'days':len(preferred),'output':str(args.output),'bytes':args.output.stat().st_size}))
    finally:
        for archive in archives: archive.close()

if __name__ == '__main__': main()
