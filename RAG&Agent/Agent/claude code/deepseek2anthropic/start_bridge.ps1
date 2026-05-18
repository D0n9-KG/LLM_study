Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = "C:\Users\jd\.conda\envs\llm\python.exe"

& $python -m uvicorn bridge_server:app --host 0.0.0.0 --port 4000
