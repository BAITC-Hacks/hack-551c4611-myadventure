# Dedicated, local-only runtime for JINALYS. Stop with Ctrl+C.
$ErrorActionPreference = 'Stop'
$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$ollamaPath = if ($ollamaCommand) { $ollamaCommand.Source } else { Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
if (-not (Test-Path -LiteralPath $ollamaPath)) { throw 'Install Ollama from https://ollama.com/download first.' }
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_NO_CLOUD = '1'
& $ollamaPath serve
exit $LASTEXITCODE
