"""Create the challenge's final zip layout without temporary indexes."""
import argparse
import zipfile
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'business_entity_resolution_submission.zip')
    a=ap.parse_args();root=Path(__file__).resolve().parent;a.output.parent.mkdir(parents=True,exist_ok=True)
    required=[root/'output'/'matching_results.tsv',root/'output'/'candidate_pairs.tsv']
    if any(not p.exists() for p in required):raise FileNotFoundError('Run prediction before packaging')
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for p in required:z.write(p,'output/'+p.name)
        for p in [root/'run_pipeline.py',root/'README.md',root/'requirements.txt',root/'package_submission.py']:
            z.write(p,'code/business_entity_resolution/'+p.name)
        for folder in ('src','models','reports'):
            for p in (root/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:
                    z.write(p,'code/business_entity_resolution/'+p.relative_to(root).as_posix())
        z.write(root/'Documentation_template.md','Documentation_template.md')
    print(a.output)
if __name__=='__main__':main()
