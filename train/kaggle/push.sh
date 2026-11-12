#!/usr/bin/env bash
# Push the training kernel to Kaggle as YOUR user: fills kernel-metadata.json from the template
# (KAGGLE_USERNAME from `kaggle config view`). The generated file is git-ignored.
set -euo pipefail
cd "$(dirname "$0")"
K="${KAGGLE_BIN:-kaggle}"
U="$("$K" config view 2>/dev/null | sed -n 's/^- username: //p')"
[ -n "$U" ] || { echo "no Kaggle username: run 'kaggle config view' / log in first" >&2; exit 2; }
sed "s/KAGGLE_USERNAME/$U/g" kernel-metadata.template.json > kernel-metadata.json
"$K" kernels push -p .
