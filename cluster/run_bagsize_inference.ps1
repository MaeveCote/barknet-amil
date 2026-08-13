<#
    BAG-SIZE ABLATION -- INFERENCE ONLY. No training.

    For each fold's already-trained AMIL model (abl_p224_nano_fX/train/best_model.pth) and
    each bag size N, subsample every test bag to N random patches and run inference. Records
    amil / amil_vote / vote accuracy per (N, fold), then a separate collector aggregates to
    a paper-ready CSV.

    WHY THIS NEEDS NO NEW CODE
    test_model.py already does exactly this: with --bags capped and
    data.max_patches_per_bag=N it subsamples each test bag to N RANDOM patches
    (data_loader._select_patches -> rng.sample, seeded for reproducibility) and scores all
    three predictors in one forward pass. So this driver just calls the existing, tested
    inference path once per (N, fold). No model-loading or attention code is reimplemented.

    WHAT VARIES vs WHAT IS FIXED
    Fixed: patch size (224), the trained model (one per fold), instance quality.
    Varies: only N, the number of patches the bag is reduced to at inference time. So any
    trend here is attributable to bag size alone -- the decoupling the patch-size ablation
    could not give you.

    ARGUMENTS (all optional; defaults assume the standard local layout)
      -RunsDir    folder with abl_p224_nano_fX\train\best_model.pth        (the models)
      -PatchRoot  merged 224 dataset: one folder of species subdirs         (the data)
      -RepoRoot   repo root containing src\ and configs\
      -OutRoot    where per-run outputs and the final CSV go
      -BagSizes   the N values to sweep
      -Folds      which folds to run

    RESUMABLE: a (N, fold) whose test_summary.json already exists is skipped.
#>
param(
    [string]  $RunsDir   = "G:\Source\BarkNet_ML\runs",
    [string]  $PatchRoot = "G:\data\patches_224\train",
    [string]  $RepoRoot  = "G:\Source\BarkNet_ML\BarkNet_ML",
    [string]  $OutRoot   = "G:\Source\BarkNet_ML\bagsize_infer",
    [int[]]   $BagSizes  = @(4, 8, 16, 32, 64, 128, 256),
    [int[]]   $Folds     = @(0, 1, 2, 3, 4),
    [string]  $Device    = "cuda:0",
    [int]     $NumWorkers = 5
)

$ErrorActionPreference = "Continue"
$ConfigFile = Join-Path $RepoRoot "configs\config.yaml"
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

# ---- pre-flight -------------------------------------------------------------
Write-Host "=== PRE-FLIGHT ===" -ForegroundColor Cyan
$fatal = $false
if (-not (Test-Path $PatchRoot)) {
    Write-Host "  MISSING patch root : $PatchRoot" -ForegroundColor Red; $fatal = $true
} else {
    $nSp = (Get-ChildItem $PatchRoot -Directory).Count
    Write-Host "  patch root OK      : $PatchRoot ($nSp species dirs)"
    if ($nSp -lt 15) { Write-Host "  ! only $nSp species -- point at the merged folder of species subdirs" -ForegroundColor Yellow }
}
foreach ($f in $Folds) {
    $m = Join-Path $RunsDir "abl_p224_nano_f$f\train\best_model.pth"
    if (-not (Test-Path $m)) { Write-Host "  MISSING model      : $m" -ForegroundColor Red; $fatal = $true }
    $b = Join-Path $RunsDir "abl_p224_nano_f$f\pretrain\best_backbone.pth"
    if (-not (Test-Path $b)) { Write-Host "  MISSING backbone   : $b" -ForegroundColor Red; $fatal = $true }
}
if (-not (Test-Path $ConfigFile)) { Write-Host "  MISSING config     : $ConfigFile" -ForegroundColor Red; $fatal = $true }
if ($fatal) { Write-Host "`nFix the above and re-run." -ForegroundColor Red; exit 1 }
Write-Host "  all models + backbones present"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogFile = Join-Path $OutRoot ("infer_log_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$total = $BagSizes.Count * $Folds.Count
Write-Host "`n$total inference run(s): N in [$($BagSizes -join ', ')] x folds [$($Folds -join ', ')]"
Write-Host "Log: $LogFile`n"

# ---- main loop (largest N first: slowest + most VRAM, so OOM shows early) ----
$idx = 0; $failed = @()
$swAll = [Diagnostics.Stopwatch]::StartNew()
foreach ($N in ($BagSizes | Sort-Object -Descending)) {
    foreach ($fold in $Folds) {
        $idx++
        $tag     = "bag{0}_f{1}" -f $N, $fold
        $runDir  = Join-Path $OutRoot $tag
        $summary = Join-Path $runDir "test\test_summary.json"
        $model    = Join-Path $RunsDir "abl_p224_nano_f$fold\train\best_model.pth"
        $backbone = Join-Path $RunsDir "abl_p224_nano_f$fold\pretrain\best_backbone.pth"

        if (Test-Path $summary) {
            Write-Host "[$idx/$total] $tag -- done, skipping" -ForegroundColor DarkGray; continue
        }
        Write-Host "[$idx/$total] $tag  (N=$N fold=$fold)" -ForegroundColor Green
        $sw = [Diagnostics.Stopwatch]::StartNew()

        # --bags capped -> the capped_<N> pass = subsample each bag to N random patches.
        # data.split.n_folds=5 is MANDATORY: config.yaml ships n_folds=null, and without it
        # the test split falls back to holdout and would NOT match the fold the model was
        # trained on -- silently scoring on the wrong (leaked) trees.
        & $Py "$RepoRoot\src\test_model.py" `
            -c $ConfigFile `
            --patch-root $PatchRoot `
            --output-dir "$runDir\test" `
            --checkpoint $model `
            --backbone-checkpoint $backbone `
            --model-size "nano" `
            --input-size 224 `
            --num-workers $NumWorkers `
            --device $Device `
            --fold $fold `
            --bags capped `
            --set "data.split.n_folds=5" `
            --set "data.max_patches_per_bag=$N" 2>&1 | Tee-Object -FilePath $LogFile -Append

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED (exit $LASTEXITCODE)" -ForegroundColor Red; $failed += $tag; continue
        }
        $sw.Stop()
        Write-Host ("  done in {0:mm\:ss}" -f $sw.Elapsed) -ForegroundColor Green
    }
}
$swAll.Stop()
Write-Host "`n=== FINISHED in $($swAll.Elapsed.ToString('hh\:mm\:ss')) ===" -ForegroundColor Cyan
if ($failed.Count) {
    Write-Host "Failed: $($failed -join ', ') -- re-run to retry (done runs skipped)" -ForegroundColor Red
}

# ---- collect into one CSV ---------------------------------------------------
Write-Host "`nCollecting results -> bagsize_results.csv" -ForegroundColor Cyan
$rows = @()
foreach ($N in $BagSizes) {
    foreach ($fold in $Folds) {
        $summary = Join-Path $OutRoot ("bag{0}_f{1}\test\test_summary.json" -f $N, $fold)
        if (-not (Test-Path $summary)) { continue }
        try { $j = Get-Content $summary -Raw | ConvertFrom-Json } catch { continue }
        # test_summary.json keys the capped pass as "capped_<N>"
        $capKey = "capped_$N"
        $e = $j.evaluations.$capKey
        if ($null -eq $e) { $e = $j.evaluations.PSObject.Properties.Value | Where-Object { $_ } | Select-Object -First 1 }
        if ($null -eq $e) { continue }
        $mc = $e.amil_vs_amil_vote.mcnemar
        $rows += [pscustomobject]@{
            bag_size            = $N
            fold                = $fold
            n_images            = $e.n_images
            mean_bag_size       = [math]::Round([double]$e.mean_bag_size, 3)
            amil_accuracy       = [math]::Round([double]$e.amil_accuracy, 6)
            amil_vote_accuracy  = [math]::Round([double]$e.amil_vote_accuracy, 6)
            vote_accuracy       = [math]::Round([double]$e.vote_accuracy, 6)
            amil_minus_vote_pp  = [math]::Round(100.0 * ($e.amil_accuracy - $e.amil_vote_accuracy), 4)
            mcnemar_p           = if ($mc) { [math]::Round([double]$mc.p_value, 5) } else { $null }
            discordant          = if ($mc) { $mc.discordant } else { $null }
            mean_attn_entropy   = if ($e.mean_attn_entropy) { [math]::Round([double]$e.mean_attn_entropy, 4) } else { $null }
        }
    }
}
$perFold = Join-Path $OutRoot "bagsize_results.csv"
$rows | Sort-Object bag_size, fold | Export-Csv -Path $perFold -NoTypeInformation
Write-Host "  wrote $perFold ($($rows.Count) rows)"

# ---- aggregate mean +/- std across folds ------------------------------------
$agg = $rows | Group-Object bag_size | ForEach-Object {
    $g = $_.Group
    function MStd($vals) {
        $m = ($vals | Measure-Object -Average).Average
        $sd = if ($vals.Count -gt 1) {
            [math]::Sqrt((($vals | ForEach-Object { ($_ - $m) * ($_ - $m) } | Measure-Object -Sum).Sum) / ($vals.Count - 1))
        } else { 0 }
        @($m, $sd)
    }
    $a  = MStd ($g.amil_accuracy);      $av = MStd ($g.amil_vote_accuracy)
    $v  = MStd ($g.vote_accuracy);      $d  = MStd ($g.amil_minus_vote_pp)
    [pscustomobject]@{
        bag_size               = [int]$_.Name
        n_folds                = $g.Count
        amil_mean              = [math]::Round($a[0], 6);  amil_std           = [math]::Round($a[1], 6)
        amil_vote_mean         = [math]::Round($av[0], 6); amil_vote_std      = [math]::Round($av[1], 6)
        vote_mean              = [math]::Round($v[0], 6);  vote_std           = [math]::Round($v[1], 6)
        amil_minus_vote_pp_mean= [math]::Round($d[0], 4);  amil_minus_vote_pp_std = [math]::Round($d[1], 4)
    }
} | Sort-Object bag_size
$aggPath = Join-Path $OutRoot "bagsize_summary.csv"
$agg | Export-Csv -Path $aggPath -NoTypeInformation
Write-Host "  wrote $aggPath ($($agg.Count) arms)"

Write-Host "`n=== SUMMARY (mean across folds) ===" -ForegroundColor Cyan
$agg | Format-Table bag_size, n_folds,
    @{L="AMIL";     E={ "{0:P2}" -f $_.amil_mean }},
    @{L="amil_vote";E={ "{0:P2}" -f $_.amil_vote_mean }},
    @{L="vote";     E={ "{0:P2}" -f $_.vote_mean }},
    @{L="AMIL-vote pp"; E={ "{0:N3} +/- {1:N3}" -f $_.amil_minus_vote_pp_mean, $_.amil_minus_vote_pp_std }} -AutoSize