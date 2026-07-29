# 任意目录启动 maka CLI

## 问题

maka-agent 是 `"private": true` 的 workspace 包,正常只能在仓库内用
`npm --workspace maka-agent run xxx` 启动。想在其他目录直接敲 `maka`,需做一次全局链接。

## 一次性设置

```bash
cd <repo>/packages/cli
npm link
```

这会在全局 bin 下创建 `maka` / `maka-agent` 软链,指向 `packages/cli/dist/cli.js`。
之后在任意目录执行 `maka` 即可启动 TUI。

## 验证

```bash
cd /tmp && maka --version    # 应输出 0.1.0
```

## 日常开发

`npm link` 指向的是 **dist/cli.js**(编译产物),不是 src。改完源码后需重新构建:

```bash
npm --workspace maka-agent run build
```

开发时可挂一个 watch 自动编译,改完即生效:

```bash
tsc -p packages/cli/tsconfig.json --watch
```

## 卸载

```bash
cd <repo>/packages/cli && npm unlink -g maka-agent
```

## 注意事项

1. **切 Node 版本后需重新 link**
   全局 bin 绑在特定 Node 版本路径下(如 `~/.nvm/versions/node/vX/bin`),
   `nvm use` 换版本后链接会失效,在新版本下重跑一次 `npm link` 即可。

2. **npx 缓存遮蔽**
   如果之前用过 `npx maka`,会在 `~/.npm/_npx/` 下留一份旧的缓存副本,
   可能把 `which maka` 抢走、导致行为异常。清理方法:

   ```bash
   rm -rf ~/.npm/_npx/*/node_modules/.bin/maka
   # 或直接删整个对应的 npx 缓存目录
   ```

   确认当前命中的是正确的全局链接:

   ```bash
   which maka            # 应为 ~/.nvm/.../bin/maka,而非 ~/.npm/_npx/...
   readlink -f $(which maka)   # 应解析到 packages/cli/dist/cli.js
   ```
