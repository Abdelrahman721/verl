#!/usr/bin/env bash
# ./upload_ckpt.sh <dir>   ->  uploads it to s3://$BUCKET/<dir name>/
set -euo pipefail

BUCKET=avey-temp-bucket

aws s3 sync "$1" "s3://$BUCKET/$(basename "${1%/}")/"
