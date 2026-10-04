$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot
$LOGDIR = Join-Path $ROOT "control_logs_seed34"

Set-Location $ROOT
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

function SeedBase($seed) {
    return Join-Path $ROOT `
        "save\fedavg\dict_users_cifar10_lt0.05\cnn4conv_num10_C1.0_r300_le5_rquery0.050\dir_balance_0.1\seed$seed\reset_random\qmode_global\fairfal"
}

function CheckSoftStart($seed) {

    $base = SeedBase $seed
    $soft = Join-Path $base "c10_ctrl_fedral_seed$seed"

    if (!(Test-Path $soft)) {
        Write-Host "Seed $seed FedRBA directory does not exist yet. This is OK."
        return
    }

    $pkls = @(Get-ChildItem $soft -File -Filter "*.pkl")
    $names = @($pkls.Name | Sort-Object)

    $expected = @(
        "dict_users_train_label_0.050.pkl",
        "dict_users_train_test_total.pkl"
    )

    if ($pkls.Count -eq 2) {

        $diff = Compare-Object $names $expected

        if (!$diff) {
            Write-Host "Seed $seed has only the clean initial caches. Safe to continue."
            return
        }
    }

    throw "Seed $seed FedRBA directory contains files beyond the clean 5% initialization. Stop to avoid mixing old experiments: $soft"
}

function CheckControlAbsent($seed, $type) {

    $base = SeedBase $seed

    if ($type -eq "oracle") {
        $p = Join-Path $base "c10_ctrl_oracle_seed$seed"
    }
    elseif ($type -eq "shuffle") {
        $p = Join-Path $base "c10_ctrl_shuffle_seed$seed"
    }
    else {
        throw "Unknown control type."
    }

    if (Test-Path $p) {
        throw "Control directory already exists: $p"
    }
}

function RunExp($seed, $customName, $budgetMode, $logName) {

    $args = @(
        "-u", "main.py",
        "--gpu", "0",
        "--seed", "$seed",
        "--custom_name", $customName,
        "--al_method", "fairfal",
        "--noise_mode", "hetero",
        "--model", "cnn4conv",
        "--dataset", "cifar10_lt",
        "--imb_ratio", "0.05",
        "--partition", "dir_balance",
        "--dd_beta", "0.1",
        "--num_users", "10",
        "--frac", "1.0",
        "--num_classes", "10",
        "--rounds", "300",
        "--local_ep", "5",
        "--reset", "random",
        "--query_ratio", "0.05",
        "--end_ratio", "0.45",
        "--budget_mode", $budgetMode,
        "--budget_min_ratio", "0.5",
        "--reliability_ema_beta", "0.5",
        "--reliability_budget_blend", "0.5"
    )

    $log = Join-Path $LOGDIR $logName

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running seed $seed : $customName"
    Write-Host "mode = $budgetMode"
    Write-Host "log  = $log"
    Write-Host "============================================================"

    & python @args 2>&1 | Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $customName"
    }
}

function PrepareCaches($seed) {

    $base = SeedBase $seed

    $soft = Join-Path $base "c10_ctrl_fedral_seed$seed"
    $oracle = Join-Path $base "c10_ctrl_oracle_seed$seed"
    $shuffle = Join-Path $base "c10_ctrl_shuffle_seed$seed"

    $total = Join-Path $soft "dict_users_train_test_total.pkl"
    $init5 = Join-Path $soft "dict_users_train_label_0.050.pkl"

    if (!(Test-Path $total)) {
        throw "Missing cache: $total"
    }

    if (!(Test-Path $init5)) {
        throw "Missing cache: $init5"
    }

    foreach ($dest in @($oracle, $shuffle)) {

        if (Test-Path $dest) {
            throw "Refusing to overwrite existing control directory: $dest"
        }

        New-Item -ItemType Directory -Force -Path $dest | Out-Null

        Copy-Item $total $dest -Force
        Copy-Item $init5 $dest -Force

        $pkls = @(Get-ChildItem $dest -File -Filter "*.pkl")

        if ($pkls.Count -ne 2) {
            throw "Cache sanity failed in $dest"
        }
    }

    Write-Host ""
    Write-Host "Seed $seed cache preparation OK."
}

function GetAccuracies($log) {

    $result = @()

    $matches = Select-String `
        -Path $log `
        -Pattern "Round\s+300,.*Test accuracy:\s*([0-9.]+)"

    foreach ($m in $matches) {
        $result += [double]$m.Matches[0].Groups[1].Value
    }

    return $result
}

function CheckRun($log) {

    $sumMatches = Select-String `
        -Path $log `
        -Pattern "^sum=(\d+)$"

    if ($sumMatches.Count -ne 8) {
        throw "Expected 8 budget allocations in $log, found $($sumMatches.Count)"
    }

    foreach ($m in $sumMatches) {

        $v = [int]$m.Matches[0].Groups[1].Value

        if ($v -ne 850) {
            throw "Budget error in $log : sum=$v"
        }
    }
}

function CheckTrio($seed) {

    $softLog = Join-Path $LOGDIR "seed${seed}_fedrba_soft.log"
    $oracleLog = Join-Path $LOGDIR "seed${seed}_oracle.log"
    $shuffleLog = Join-Path $LOGDIR "seed${seed}_shuffled.log"

    CheckRun $softLog
    CheckRun $oracleLog
    CheckRun $shuffleLog

    $soft = @(GetAccuracies $softLog)
    $oracle = @(GetAccuracies $oracleLog)
    $shuffle = @(GetAccuracies $shuffleLog)

    if (
        $soft.Count -ne 9 -or
        $oracle.Count -ne 9 -or
        $shuffle.Count -ne 9
    ) {
        throw "Seed ${seed}: expected 9 Round-300 checkpoints in each run."
    }

    if (
        $soft[0] -ne $oracle[0] -or
        $soft[0] -ne $shuffle[0]
    ) {
        throw "Seed ${seed}: 5% accuracy mismatch."
    }

    if (
        $soft[1] -ne $oracle[1] -or
        $soft[1] -ne $shuffle[1]
    ) {
        throw "Seed ${seed}: 10% accuracy mismatch."
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "SEED $seed SANITY PASSED"
    Write-Host "============================================================"

    Write-Host "FedRBA   : $($soft -join ', ')"
    Write-Host "Oracle   : $($oracle -join ', ')"
    Write-Host "Shuffled : $($shuffle -join ', ')"
    Write-Host "All budget sums = 850."
    Write-Host "5% and 10% match across all three."
}

# ============================================================
# PRECHECK
# ============================================================

foreach ($seed in @(3,4)) {

    CheckSoftStart $seed
    CheckControlAbsent $seed "oracle"
    CheckControlAbsent $seed "shuffle"
}

Write-Host ""
Write-Host "PRECHECK PASSED."
Write-Host "Starting all six experiments..."

# ============================================================
# RUN SEED 3 AND 4
# ============================================================

foreach ($seed in @(3,4)) {

    Write-Host ""
    Write-Host "############################################################"
    Write-Host "START SEED $seed"
    Write-Host "############################################################"

    RunExp `
        $seed `
        "c10_ctrl_fedral_seed$seed" `
        "estimated_r_soft_blend" `
        "seed${seed}_fedrba_soft.log"

    PrepareCaches $seed

    RunExp `
        $seed `
        "c10_ctrl_oracle_seed$seed" `
        "oracle_r_blend" `
        "seed${seed}_oracle.log"

    RunExp `
        $seed `
        "c10_ctrl_shuffle_seed$seed" `
        "shuffled_r_soft_blend" `
        "seed${seed}_shuffled.log"

    CheckTrio $seed
}

Write-Host ""
Write-Host "============================================================"
Write-Host "ALL SEED3 / SEED4 CONTROLS FINISHED SUCCESSFULLY"
Write-Host "============================================================"
Write-Host "Logs saved in:"
Write-Host $LOGDIR
