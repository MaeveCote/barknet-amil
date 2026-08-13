<#
    WHOLE-IMAGE BASELINE (single-instance degenerate case) -- local Windows, 5-fold CV.

    WHAT THIS IS
    The point where MIL collapses to ordinary image classification: each whole trunk photo
    is resized to 224 and classified directly. No patches, no bag, no attention head. It is
    NOT another point on the patch-size curve (a 1-instance "bag" has nothing to aggregate);
    it is the BASELINE the patch-based method is compared against -- H5: "does patch-based
    MIL beat just classifying the whole image?"

    WHY IT USES pretrain_backbone.py
    Stage 1 is exactly a per-image classifier: backbone + classifier head, no attention,
    one label per input. Feeding it whole images instead of patches is the same code path
    with a different input folder. Tree-level splitting still works because tree_id_index=0
    matches the ORIGINAL BarkNet naming (tree_class_circ_device_date_time_crop.jpg) -- the
    tree id is the first token, same as for patches.

    EXPECT OVERFITTING, AND THAT IS THE POINT
    ~18.5k whole images vs ~2.9M patches is ~156x less training data. The model will memorise
    quickly (train acc -> ~100% early). That gap is part of the finding: more instances is
    one reason patch-based MIL can beat whole-image. Report train-vs-val divergence, not just
    final val acc.

    DATA LAYOUT REQUIRED
      <ImageRoot>\<species>\<original_filename>.jpg      e.g.  barknet_raw\CHR\41_CHR_83_...jpg
    (the raw BarkNet download already looks like this.)

    RESUMABLE: a fold whose best_backbone.pth exists is skipped.
#>
param(
    [string] $ImageRoot = "G:\data\barknet_raw",
    [string] $RepoRoot  = "G:\Source\BarkNet_ML\BarkNet_ML",
    [string] $OutRoot   = "G:\Source\BarkNet_ML\runs_wholeimage",
    [int[]]  $Folds     = @(0, 1, 2, 3, 4),
    [int]    $Epochs    = 40,
    [int]    $BatchSize = 64,        # whole images at 224; 64 fits 24 GB easily
    [string] $Device    = "cuda:0",
    [int]    $NumWorkers = 5
)

$ErrorActionPreference = "Continue"
$ConfigFile = Join-Path $RepoRoot "configs\config.yaml"
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

# ---- pre-flight -------------------------------------------------------------
Write-Host "=== PRE-FLIGHT ===" -ForegroundColor Cyan
if (-not (Test-Path $ImageRoot)) {
    Write-Host "  MISSING image root: $ImageRoot" -ForegroundColor Red; exit 1
}
$speciesDirs = Get-ChildItem $ImageRoot -Directory
Write-Host "  image root OK      : $ImageRoot ($($speciesDirs.Count) species dirs)"
$sample = Get-ChildItem (Join-Path $speciesDirs[0].FullName "*") -File | Select-Object -First 1
if ($sample) {
    Write-Host "  sample file        : $($sample.Name)"
    $tok = $sample.BaseName.Split("_")
    Write-Host "  -> tree id (token 0) = '$($tok[0])'  (must be the numeric tree id)"
    if ($tok[0] -notmatch '^\d+$') {
        Write-Host "  ! token 0 is not numeric -- tree_id_index may be wrong for these files" -ForegroundColor Yellow
    }
}
if (-not (Test-Path $ConfigFile)) { Write-Host "  MISSING config: $ConfigFile" -ForegroundColor Red; exit 1 }

# total image count (for the overfitting-context note)
$nImg = ($speciesDirs | ForEach-Object { (Get-ChildItem $_.FullName -File -Recurse).Count } | Measure-Object -Sum).Sum
Write-Host "  total images       : $nImg  (vs ~2.9M patches in Stage 1 -- expect fast overfit)"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogFile = Join-Path $OutRoot ("wholeimage_log_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "`n$($Folds.Count) fold(s), $Epochs epochs each. Log: $LogFile`n"

# ---- main loop --------------------------------------------------------------
foreach ($fold in $Folds) {
    $tag    = "wholeimg_f$fold"
    $runDir = Join-Path $OutRoot $tag
    $bb     = Join-Path $runDir "pretrain\best_backbone.pth"
    if (Test-Path $bb) {
        Write-Host "[$tag] already done, skipping" -ForegroundColor DarkGray; continue
    }
    Write-Host "[$tag] training whole-image classifier (fold $fold)" -ForegroundColor Green
    $sw = [Diagnostics.Stopwatch]::StartNew()

    # pretrain_backbone.py = per-image classifier (no attention, no bag).
    # --patch-root points at the WHOLE-IMAGE folder; the loader treats each image as one
    # labelled instance. Tree-level split via tree_id_index=0 (original BarkNet naming).
    # data.split.n_folds=5 is mandatory (config ships n_folds=null -> would be holdout).
    # NOTE: the training call is NOT piped. tqdm redraws its bar with a carriage return,
    # which only works writing straight to the console. Piping it (2>&1 | Tee-Object) breaks
    # the in-place redraw and stacks a new line per update. So we let it print directly and
    # rely on pretrain_summary.json for the recorded numbers. A per-fold header is written to
    # the log so you still have a record of what ran.
    "[$([DateTime]::Now.ToString('HH:mm:ss'))] $tag  fold=$fold epochs=$Epochs" |
            Out-File -FilePath $LogFile -Append
    & $Py "$RepoRoot\src\pretrain_backbone.py" `
        -c $ConfigFile `
        --patch-root $ImageRoot `
        --output-dir "$runDir\pretrain" `
        --model-size "nano" `
        --input-size 224 `
        --num-workers $NumWorkers `
        --device $Device `
        --fold $fold `
        --epochs $Epochs `
        --set "data.split.n_folds=5" `
        --set "pretrain.batch_size=$BatchSize" `
        --set "pretrain.checkpoint_metric=val_acc" `
        --set "pretrain.early_stopping_patience=999" `
        --set "pretrain.min_epochs=999" `
        --set "data.file.image_id_index=5"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        continue
    }
    $sw.Stop()
    Write-Host ("  done in {0:hh\:mm\:ss}" -f $sw.Elapsed) -ForegroundColor Green
}

# ---- collect val acc per fold ----------------------------------------------
Write-Host "`nCollecting results -> wholeimage_results.csv" -ForegroundColor Cyan
$rows = @()
foreach ($fold in $Folds) {
    $sumP = Join-Path $OutRoot "wholeimg_f$fold\pretrain\pretrain_summary.json"
    if (-not (Test-Path $sumP)) { continue }
    try { $j = Get-Content $sumP -Raw | ConvertFrom-Json } catch { continue }
    $rows += [pscustomobject]@{
        fold          = $fold
        best_val_acc  = $j.best_val_acc
        best_epoch    = $j.best_epoch
    }
}
$csv = Join-Path $OutRoot "wholeimage_results.csv"
$rows | Sort-Object fold | Export-Csv -Path $csv -NoTypeInformation
Write-Host "  wrote $csv ($($rows.Count) folds)"

if ($rows.Count -gt 1) {
    $vals = $rows.best_val_acc
    $m  = ($vals | Measure-Object -Average).Average
    $sd = [math]::Sqrt((($vals | ForEach-Object { ($_-$m)*($_-$m) } | Measure-Object -Sum).Sum)/($vals.Count-1))
    Write-Host ("`nWhole-image baseline val acc: {0:N2} +/- {1:N2}  (over {2} folds)" -f $m,$sd,$rows.Count) -ForegroundColor Cyan
    Write-Host "Compare against the IMAGE-level AMIL accuracy at 224 (patch-based MIL)." -ForegroundColor Cyan
    Write-Host "If patch-MIL > whole-image, that justifies the whole patching approach (H5)."
}
