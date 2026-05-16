# Запуск uv из just на Windows (Git Bash под just не видит uv в PATH).
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& uv @Args
exit $LASTEXITCODE
