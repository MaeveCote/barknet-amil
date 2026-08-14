<#
    WHOLE-IMAGE BASELINE -- TEST + COMPILE. Local Windows, 5-fold.

    Runs test-time inference for the whole-image classifier trained by run_wholeimage.ps1,
    then averages across folds.

    HOW IT WORKS
    The whole-image model has NO attention head and NO bags -- it is a plain per-image
    classifier (the Stage-1 backbone + classifier). test_model.py's --mode vote path scores
    exactly that: it loads the backbone, and for each "bag" takes a majority vote over the
    bag's patches. For whole images every bag is a single image (n_patches = 1), so the vote
    is just that image's prediction -- i.e. the whole-image classification result. No AMIL
    checkpoint is needed or used.

    So there is only ONE predictor here (vote = the whole-image classifier). There is no
    amil / amil_vote, because there is no bag to aggregate. That is the point of the
    baseline.

    INPUT   runs_wholeimage\wholeimg_fX\pretrain\best_backbone.pth   (from run_wholeimage.ps1)
    OUTPUT  wholeimage_test_results.csv (per fold) + printed mean +/- std

    tqdm note: the test call is NOT piped, so its progress bar renders in place instead of
    stacking. Results come from the written test_summary.json.

    RESUMABLE: a fold whose test_summary.json exists is skipped.
#>
param(
    [string] $WholeRoot = "G:\Source\BarkNet_ML\runs_wholeimage",
    [string] $ImageRoot = "G:\data\barknet_raw",
    [string] $RepoRoot  = "G:\Source\BarkNet_ML\BarkNet_ML",
    [int[]]  $Folds     = @(0, 1, 2, 3, 4),
    [string] $Device    = "cuda:0",
    [int]    $NumWorkers = 5
)

$ErrorActionPreference = "Continue"
$ConfigFile = Join-Path $RepoRoot "configs\config.yaml"
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

# ---- pre-flight -------------------------------------------------------------
Write-Host "=== PRE-FLIGHT ===" -ForegroundColor Cyan
$fatal = $false
if (-not (Test-Path $ImageRoot)) { Write-Host "  MISSING image root: $ImageRoot" -ForegroundColor Red; $fatal = $true }
foreach ($f in $Folds) {
    $bb = Join-Path $WholeRoot "wholeimg_f$f\pretrain\best_backbone.pth"
    if (-not (Test-Path $bb)) { Write-Host "  MISSING backbone  : $bb" -ForegroundColor Red; $fatal = $true }
}
if (-not (Test-Path $ConfigFile)) { Write-Host "  MISSING config    : $ConfigFile" -ForegroundColor Red; $fatal = $true }
if ($fatal) { Write-Host "`nFix the above and re-run." -ForegroundColor Red; exit 1 }
Write-Host "  all backbones + image root present"

# ---- test each fold ---------------------------------------------------------
foreach ($fold in $Folds) {
    $tag     = "wholeimg_f$fold"
    $runDir  = Join-Path $WholeRoot $tag
    $backbone = Join-Path $runDir "pretrain\best_backbone.pth"
    $summary  = Join-Path $runDir "test\test_summary.json"
    if (Test-Path $summary) {
        Write-Host "[$tag] test already done, skipping" -ForegroundColor DarkGray
        continue
    }
    Write-Host "[$tag] testing whole-image classifier (fold $fold)" -ForegroundColor Green
    $sw = [Diagnostics.Stopwatch]::StartNew()

    # --mode vote: backbone-only, one prediction per image. --bags full (each image is its
    # own single-item bag, so capping is irrelevant). data.split.n_folds=5 is mandatory:
    # config ships n_folds=null, and the test split must match the fold the model trained on.
    # NOT piped -> tqdm bar renders in place.
    & $Py "$RepoRoot\src\test_model.py" `
        -c $ConfigFile `
        --patch-root $ImageRoot `
        --output-dir "$runDir\test" `
        --backbone-checkpoint $backbone `
        --model-size "nano" `
        --input-size 224 `
        --num-workers $NumWorkers `
        --device $Device `
        --fold $fold `
        --mode vote --bags full `
        --set "data.split.n_folds=5" `
        --set "data.file.image_id_index=6"    # crop number: UNIQUE per image within a tree,
    # so each image is its own single-item bag
    # (n_patches=1). Index 5 (time) was the bug:
    # crops of one tree share a timestamp and
    # collapsed into one tree-level bag.
    $code = $LASTEXITCODE

    if ($code -ne 0) {
        Write-Host "  FAILED (exit $code)" -ForegroundColor Red
        continue
    }
    $sw.Stop()
    Write-Host ("  done in {0:mm\:ss}" -f $sw.Elapsed) -ForegroundColor Green
}

# ---- collect + average ------------------------------------------------------
Write-Host "`nCollecting -> wholeimage_test_results.csv" -ForegroundColor Cyan
$rows = @()
foreach ($fold in $Folds) {
    $summary = Join-Path $WholeRoot "wholeimg_f$fold\test\test_summary.json"
    if (-not (Test-Path $summary)) { continue }
    try { $j = Get-Content $summary -Raw | ConvertFrom-Json } catch { continue }
    $e = $j.evaluations.full
    if ($null -eq $e) { $e = $j.evaluations.PSObject.Properties.Value | Where-Object { $_ } | Select-Object -First 1 }
    if ($null -eq $e) { continue }

    # test_model.py writes accuracy into the summary but NOT macro-F1, so compute F1 from the
    # per-image predictions CSV (test_predictions_full.csv: true_class, vote_pred) via a tiny
    # inline Python call -- reusing sklearn rather than reimplementing F1 in PowerShell.
    $predCsv = Join-Path $WholeRoot "wholeimg_f$fold	est	est_predictions_full.csv"
    $macroF1 = $null
    if (Test-Path $predCsv) {
        $f1out = & $Py -c "import pandas as pd,sys; from sklearn.metrics import f1_score; d=pd.read_csv(sys.argv[1]); print(f1_score(d['true_class'],d['vote_pred'],labels=sorted(d['true_class'].unique()),average='macro',zero_division=0))" $predCsv 2>$null
        if ($LASTEXITCODE -eq 0 -and $f1out) { $macroF1 = [math]::Round([double]$f1out, 6) }
    }
    # SANITY: a true whole-image baseline has one image per bag. mean_bag_size must be ~1.
    # If it is larger, the grouping is still wrong (images collapsed into tree/time bags) and
    # the number is NOT the whole-image baseline -- flag it rather than silently reporting.
    $meanBag = if ($e.mean_bag_size) { [double]$e.mean_bag_size } else { $null }
    if ($meanBag -ne $null -and $meanBag -gt 1.05) {
        Write-Host ("  ! WARNING fold {0}: mean_bag_size={1:N1} (expected ~1). Images were grouped into bags -- this is NOT the whole-image baseline. Check image_id_index." -f $fold, $meanBag) -ForegroundColor Red
    }
    $rows += [pscustomobject]@{
        fold           = $fold
        n_images       = $e.n_images
        mean_bag_size  = if ($meanBag -ne $null) { [math]::Round($meanBag, 3) } else { $null }
        vote_accuracy  = if ($e.vote_accuracy) { [math]::Round([double]$e.vote_accuracy, 6) } else { $null }
        vote_macro_f1  = $macroF1
    }
}
$csv = Join-Path $WholeRoot "wholeimage_test_results.csv"
$rows | Sort-Object fold | Export-Csv -Path $csv -NoTypeInformation
Write-Host "  wrote $csv ($($rows.Count) folds)"

if ($rows.Count -ge 1) {
    function MeanStd($vals) {
        $vals = @($vals | Where-Object { $_ -ne $null })
        if ($vals.Count -eq 0) { return @($null, $null) }
        $m = ($vals | Measure-Object -Average).Average
        $sd = if ($vals.Count -gt 1) {
            [math]::Sqrt((($vals | ForEach-Object { ($_-$m)*($_-$m) } | Measure-Object -Sum).Sum)/($vals.Count-1))
        } else { 0 }
        @($m, $sd)
    }
    $a = MeanStd ($rows.vote_accuracy)
    $f = MeanStd ($rows.vote_macro_f1)

    $agg = [pscustomobject]@{
        n_folds            = $rows.Count
        accuracy_mean      = if ($a[0] -ne $null) { [math]::Round($a[0],6) } else { $null }
        accuracy_std       = if ($a[1] -ne $null) { [math]::Round($a[1],6) } else { $null }
        macro_f1_mean      = if ($f[0] -ne $null) { [math]::Round($f[0],6) } else { $null }
        macro_f1_std       = if ($f[1] -ne $null) { [math]::Round($f[1],6) } else { $null }
    }
    $aggCsv = Join-Path $WholeRoot "wholeimage_test_summary.csv"
    $agg | Export-Csv -Path $aggCsv -NoTypeInformation
    Write-Host "  wrote $aggCsv"

    Write-Host "`n=== WHOLE-IMAGE BASELINE (single-instance) ===" -ForegroundColor Cyan
    if ($a[0] -ne $null) { Write-Host ("  accuracy : {0:P2} +/- {1:P2}  ({2} folds)" -f $a[0],$a[1],$rows.Count) }
    if ($f[0] -ne $null) { Write-Host ("  macro-F1 : {0:N4} +/- {1:N4}" -f $f[0],$f[1]) }
    Write-Host "`n  Compare against IMAGE-level AMIL at 224 (patch-based MIL, ~91.9%)." -ForegroundColor Cyan
    Write-Host "  Patch-MIL > whole-image justifies the patching approach (H5)."
}
