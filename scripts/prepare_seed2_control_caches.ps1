$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$CacheBase = Join-Path $Repo "save\fedavg\dict_users_cifar10_lt0.05\cnn4conv_num10_C1.0_r100_le5_rquery0.050\dir_balance_0.1\seed2\reset_random\qmode_global\fairfal"
$Source = Join-Path $CacheBase "c10_lambda1_seed2"

$Required = @(
    "dict_users_train_test_total.pkl",
    "dict_users_train_label_0.050.pkl"
)

foreach ($name in $Required) {
    $p = Join-Path $Source $name
    if (-not (Test-Path $p)) {
        throw "Missing source cache file: $p"
    }
}

$DestNames = @(
    "c10_ctrl_fedral_seed2",
    "c10_ctrl_oracle_seed2",
    "c10_ctrl_shuffle_seed2"
)

foreach ($destName in $DestNames) {
    $Dest = Join-Path $CacheBase $destName
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null

    foreach ($name in $Required) {
        Copy-Item -Force (Join-Path $Source $name) (Join-Path $Dest $name)
    }

    $forbidden = Get-ChildItem -Path $Dest -Filter "dict_users_train_label_*.pkl" |
        Where-Object { $_.Name -ne "dict_users_train_label_0.050.pkl" }
    if ($forbidden) {
        throw "Forbidden later-stage label cache found in $Dest : $($forbidden.Name -join ', ')"
    }

    Write-Host "Prepared $Dest"
    Get-ChildItem $Dest | Select-Object Name, Length
}
