#!/bin/bash
# v2 -- compile the tables. CPU only, a minute or two. submit_v2.sh queues it to run after
# every v2 job has ended (afterany); you can also run it by hand on the login node at any
# time to see partial results:
#   bash cluster/v2/job_compile.sh
#SBATCH --job-name=bark_v2_compile
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.out
set -euo pipefail
HERE="${V2_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=v2_env.sh
source "$HERE/v2_env.sh"

login_venv
trap login_venv_cleanup EXIT
python "$HERE/compile_v2.py" --v2-root "$V2_ROOT" --tree-runs "$TREE_RUNS" \
    --fold-file "$FOLD_FILE" --out-dir "$V2_ROOT/tables"
echo
echo "Pull to your PC:"
echo "  scp -r $USER@rorqual.alliancecan.ca:$V2_ROOT/tables ."
