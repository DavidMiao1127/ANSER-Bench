"""Package the anonymous source, complete queries, full rule corpora, and samples of large corpora."""
from __future__ import annotations
import argparse
import zipfile
from pathlib import Path


def files(root):
    for name in ('README.md','README_zh.md','LICENSE','requirements.txt','run_benchmark.py','.env.example','.gitignore','.gitattributes'):
        yield root/name
    for name in ('query','src','corpus','assets'):
        for p in sorted((root/name).rglob('*')):
            if p.is_file() and not any(x in p.parts for x in ('__pycache__','.pytest_cache','.git','tests')) and p.name!='.DS_Store' and not p.name.startswith('test_') and p.suffix not in {'.pyc','.pyo'}:
                yield p


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=root/'outputs/ANSER-Bench-source.zip')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    paths=list(files(root))
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            relative=Path('ANSER-Bench')/p.relative_to(root)
            z.write(p,relative.as_posix())
    print(f'Packaged {len(paths)} public files: {args.output.name}')


if __name__=='__main__':main()
