#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from zigzag.pipeline_posthoc import run_posthoc, build_posthoc_parser

def main():
    ap = build_posthoc_parser()
    args = ap.parse_args()
    run_posthoc(args)

if __name__ == '__main__':
    main()
