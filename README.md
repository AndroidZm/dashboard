# dashboard

YMOS 抄底信号策略的回测数据仓库：信号表、等比复权日线、指数与回测结果。
数据来自本地 `D:\AppGallery\ymos-backtest-data`，字段说明、回测规则和已知问题见 [DATA_README.md](DATA_README.md)。

HTML 分析页面只在本地查看，已通过 `.gitignore` 排除，不会同步到仓库。

## 同步

```powershell
git clone -b claude/eager-curie-9u4wsm https://github.com/AndroidZm/dashboard.git   # 仅首次
cd dashboard
.\sync.ps1                          # 从默认源目录 D:\AppGallery\ymos-backtest-data 同步，并推送到当前分支
.\sync.ps1 -Source "E:\其他目录"    # 指定其他源目录
```

- 镜像复制：源目录里删掉的文件，仓库里也会删掉。
- 源目录里名为 `README.md` 的文件不会被复制（避免覆盖本文件），所以数据说明叫 `DATA_README.md`。
- 本机访问 GitHub 要走代理，首次执行一次：`git config http.proxy http://127.0.0.1:7897`（只对本仓库生效）。
- 如 PowerShell 禁止运行脚本，先执行：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
