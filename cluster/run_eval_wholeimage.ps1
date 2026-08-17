<#
    Whole-image evaluation, all 5 folds + average. Calls eval_wholeimage.py (which bypasses
    the bag loader: one resized image -> one prediction through best_backbone.pth).
#>
param(
    [string] $ImageRoot = "G:\Source\BarkNet_ML\BarkNet_ML\data\barknet\dataset",
    [string] $WholeRoot = "G:\Source\BarkNet_ML\runs_wholeimage",
    [string] $RepoRoot  = "G:\Source\BarkNet_ML\BarkNet_ML",
    [int[]]  $Folds     = @(0,1,2,3,4),
    [string] $ModelSize = "nano",
    [string] $Device    = "cuda:0"
)
$ErrorActionPreference = "Continue"
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$Script = Join-Path $PSScriptRoot "eval_wholeimage.py"

$summaries = @()
foreach ($fold in $Folds) {
    $bb = Join-Path $WholeRoot "wholeimg_f$fold\pretrain\best_backbone.pth"
    if (-not (Test-Path $bb)) { Write-Host "MISSING backbone: $bb" -ForegroundColor Red; continue }
    $out = Join-Path $WholeRoot "wholeimg_f$fold\wholeimage_preds.csv"
    Write-Host "[fold $fold] evaluating whole images" -ForegroundColor Green
    # NOT piped -> progress prints cleanly
    & $Py $Script --image-root $ImageRoot --backbone $bb --repo-root $RepoRoot `
        --fold $fold --n-folds 5 --model-size $ModelSize --device $Device --out $out
    $sc = Join-Path $WholeRoot "wholeimg_f$fold\wholeimage_preds.summary.csv"
    if (Test-Path $sc) { $summaries += (Import-Csv $sc) }
}

if ($summaries.Count -ge 1) {
    $accs = $summaries | ForEach-Object { [double]$_.accuracy }
    $f1s  = $summaries | ForEach-Object { [double]$_.macro_f1 }
    function MStd($v){ $m=($v|Measure-Object -Average).Average; $sd=if($v.Count-gt1){[math]::Sqrt((($v|%{($_-$m)*($_-$m)}|Measure-Object -Sum).Sum)/($v.Count-1))}else{0}; @($m,$sd) }
    $a=MStd $accs; $f=MStd $f1s
    $summaries | Export-Csv (Join-Path $WholeRoot "wholeimage_allfolds.csv") -NoTypeInformation
    Write-Host "`n=== WHOLE-IMAGE BASELINE (single instance, one prediction per image) ===" -ForegroundColor Cyan
    Write-Host ("  accuracy : {0:P2} +/- {1:P2}  ({2} folds)" -f $a[0],$a[1],$summaries.Count)
    Write-Host ("  macro-F1 : {0:N4} +/- {1:N4}" -f $f[0],$f[1])
    Write-Host "  Compare against 224 image-level AMIL (~91.9%) for H5."
}