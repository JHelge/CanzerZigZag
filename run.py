#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from zigzag.cli import build_run_parser
from zigzag.pipeline_run import run_pipeline

def main():
    ap = build_run_parser()
    ap.add_argument('--skip_posthoc', action='store_true', help='Skip posthoc stage (can be rerun with posthoc.py).')
    args = ap.parse_args()
    run_pipeline(args)

if __name__ == '__main__':
    main()
