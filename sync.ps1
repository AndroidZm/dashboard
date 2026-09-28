# 将本地 dashboard 数据同步到本仓库（排除 HTML），然后提交并推送
# 用法：在本仓库根目录的 PowerShell 中执行  .\sync.ps1
#   可选参数：-Source <本地数据目录>；推送到当前所在分支

param(
    [string]$Source = "D:\AppGallery\YMOS\持仓与关注\dashboard"
)

$ErrorActionPreference = "Stop"
$Repo = $PSScriptRoot

if (-not (Test-Path $Source)) { throw "源目录不存在: $Source" }

# /MIR 镜像同步（本地删除的文件也会从仓库删除）
# /XF 排除 HTML 和仓库自身的配置文件，/XD 排除 .git
robocopy $Source $Repo /MIR /XF *.html *.htm .gitignore sync.ps1 README.md /XD .git /NFL /NDL /NP
if ($LASTEXITCODE -ge 8) { throw "robocopy 失败，退出码 $LASTEXITCODE" }

Set-Location $Repo
git config core.quotepath false
git add -A
if (git status --porcelain) {
    git commit -m "sync data $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    git push -u origin HEAD
} else {
    Write-Host "没有变化，无需提交。"
}
