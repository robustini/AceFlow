$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repoArchiveUrl = 'https://github.com/robustini/AceFlow/archive/refs/heads/main.zip'
$soundfontUrl = 'https://musical-artifacts.com/artifacts/3677/LiteGM_v1.03.sf2'
$soundfontFileName = 'LiteGM_v1.03.sf2'
$workingRoot = (Get-Location).Path
$aceStepDir = Join-Path $workingRoot 'acestep'

if (-not (Test-Path $aceStepDir -PathType Container)) {
    throw "Current directory does not look like an ACE-Step root. Missing folder: $aceStepDir"
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('aceflow_install_' + [System.Guid]::NewGuid().ToString('N'))
$zipPath = Join-Path $tempRoot 'aceflow_main.zip'
$extractDir = Join-Path $tempRoot 'extract'

function Test-Sf2File {
    param([string]$Path)
    if (-not (Test-Path $Path -PathType Leaf)) {
        return $false
    }
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 12) {
        return $false
    }
    $riff = [System.Text.Encoding]::ASCII.GetString($bytes[0..3])
    $kind = [System.Text.Encoding]::ASCII.GetString($bytes[8..11])
    return $riff -eq 'RIFF' -and $kind -eq 'sfbk'
}

New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

try {
    Write-Host 'Downloading AceFlow repository archive...'
    Invoke-WebRequest -Uri $repoArchiveUrl -OutFile $zipPath

    Write-Host 'Extracting archive...'
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    $repoRoot = Join-Path $extractDir 'AceFlow-main'
    if (-not (Test-Path $repoRoot -PathType Container)) {
        $firstDir = Get-ChildItem -Path $extractDir -Directory | Select-Object -First 1
        if ($null -eq $firstDir) {
            throw 'Unable to locate extracted repository root.'
        }
        $repoRoot = $firstDir.FullName
    }

    $sourceUi = Join-Path $repoRoot 'acestep/ui/aceflow'
    $sourceBat = Join-Path $repoRoot 'start_aceflow_ui.bat'
    $sourceSh = Join-Path $repoRoot 'start_aceflow_ui.sh'

    if (-not (Test-Path $sourceUi -PathType Container)) {
        throw 'Missing source folder in archive: acestep/ui/aceflow'
    }
    if (-not (Test-Path $sourceBat -PathType Leaf)) {
        throw 'Missing source file in archive: start_aceflow_ui.bat'
    }
    if (-not (Test-Path $sourceSh -PathType Leaf)) {
        throw 'Missing source file in archive: start_aceflow_ui.sh'
    }

    $targetUiParent = Join-Path $workingRoot 'acestep/ui'
    $targetUi = Join-Path $targetUiParent 'aceflow'
    $targetBat = Join-Path $workingRoot 'start_aceflow_ui.bat'
    $targetSh = Join-Path $workingRoot 'start_aceflow_ui.sh'
    $targetSoundfontsDir = Join-Path $targetUi 'soundfonts'
    $targetSoundfont = Join-Path $targetSoundfontsDir $soundfontFileName

    New-Item -ItemType Directory -Force -Path $targetUiParent | Out-Null

    if (Test-Path $targetUi) {
        Write-Host 'Removing previous acestep/ui/aceflow...'
        Remove-Item -Recurse -Force $targetUi
    }

    Write-Host 'Installing AceFlow files...'
    Copy-Item -Path $sourceUi -Destination $targetUiParent -Recurse -Force
    Copy-Item -Path $sourceBat -Destination $targetBat -Force
    Copy-Item -Path $sourceSh -Destination $targetSh -Force
    New-Item -ItemType Directory -Force -Path $targetSoundfontsDir | Out-Null

    Write-Host 'Downloading default SoundFont into acestep/ui/aceflow/soundfonts/...'
    try {
        Invoke-WebRequest -Uri $soundfontUrl -OutFile $targetSoundfont
        if (Test-Sf2File -Path $targetSoundfont) {
            Write-Host "Default SoundFont installed: $targetSoundfont"
        } else {
            if (Test-Path $targetSoundfont) {
                Remove-Item -Force $targetSoundfont
            }
            Write-Warning 'Downloaded file does not look like a valid .sf2. AceFlow was installed anyway and can still use the internal chord synth fallback.'
        }
    }
    catch {
        if (Test-Path $targetSoundfont) {
            Remove-Item -Force $targetSoundfont
        }
        Write-Warning 'Unable to download the default SoundFont. AceFlow was installed anyway and can still use the internal chord synth fallback.'
    }

    Write-Host ''
    Write-Host 'AceFlow installation completed.'
    Write-Host 'To run AceFlow, launch start_aceflow_ui.bat or start_aceflow_ui.sh from the ACE-Step root.'
    if (Test-Path $targetSoundfont -PathType Leaf) {
        Write-Host "Installed default SoundFont: $targetSoundfont"
    } else {
        Write-Host "Default SoundFont is not present. AceFlow can still render chord references with the internal synth, or you can place a valid .sf2 manually in $targetSoundfontsDir"
    }
}
finally {
    if (Test-Path $tempRoot) {
        Remove-Item -Recurse -Force $tempRoot
    }
}
