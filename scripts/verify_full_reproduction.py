#!/usr/bin/env python3
"""Compare an independent full rerun with the frozen publication evidence.

Usage: python scripts/verify_full_reproduction.py /path/to/fresh-study
First generate the latter with run_study.py --output-root /path/to/fresh-study.
Elapsed time and environment are allowed to differ; reported results are not.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('fresh',type=Path);args=parser.parse_args()
    original=ROOT/'data';fresh=args.fresh
    a=gzip.decompress((original/'results/raw_experiments.csv.gz').read_bytes())
    b=gzip.decompress((fresh/'results/raw_experiments.csv.gz').read_bytes())
    sa=json.loads((original/'results/study.json').read_text());sb=json.loads((fresh/'results/study.json').read_text())
    keys=['study_version','date','quick','workload_metadata','results','exact','sample_sizes']
    fields={k:sa[k]==sb[k] for k in keys}
    traces={p.name:p.read_bytes()==(fresh/'traces'/p.name).read_bytes() for p in (original/'traces').iterdir() if p.is_file()}
    assert not sa['quick'] and not sb['quick']
    assert a==b,'raw realizations changed'
    assert all(fields.values()),fields
    assert all(traces.values()),traces
    report=dict(status='PASS',raw_decompressed_identical=True,raw_sha256=hashlib.sha256(a).hexdigest(),
                raw_realizations=len(a.splitlines())-1,summary_fields_identical=fields,traces_byte_identical=traces,
                rerun_seconds=sb['elapsed_seconds'],rerun_environment=sb['environment'])
    (original/'results/full_reproduction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
