# dashboard

持仓与关注的数据仓库，源目录为本地 `D:\AppGallery\YMOS\持仓与关注\dashboard`。

数据分析用的 HTML 文件只在本地查看，已通过 `.gitignore` 排除，不会同步到仓库。

## 同步

```powershell
git clone https://github.com/AndroidZm/dashboard.git
cd dashboard
.\sync.ps1                         # 从默认源目录同步并推送到当前分支
.\sync.ps1 -Source "E:\其他目录"    # 指定其他源目录
```

如 PowerShell 禁止运行脚本，先执行：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
