$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$ConfigPath = if ($args.Count -ge 1) { $args[0] } else { Join-Path $ScriptRoot "configs/example.openai.json" }

Push-Location $ScriptRoot
try {
  & $PythonExe -m sampler.cli validate-dataset --config $ConfigPath
} finally {
  Pop-Location
}
