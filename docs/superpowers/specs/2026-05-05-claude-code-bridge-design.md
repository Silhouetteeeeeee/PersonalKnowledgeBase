# Claude Code Bridge — 企业微信远程编码桥接

**Problem:** 在家/公司无法直接操作本地开发机，需要通过企业微信远程执行编码任务（代码修改、git 操作、文件编辑等），像在本地终端中使用 Claude Code 一样。

**Goal:** 通过企业微信消息与本地 Claude Code 持久会话交互，实现远程编码控制。

**Non-goal:** 多人协作、权限管理、文件上传下载（已有文件处理功能）、替换现有知识图谱问答。

---

## Architecture

```
企业微信 → bot.py (_on_text)
              │
              ├── "/code ..." 且开关打开 → claude_bridge.handle()
              │                              │
              │                              ├── tmux 会话未启动？ → 创建 claude 会话
              │                              ├── tmux send-keys 发送消息
              │                              ├── 轮询 capture-pane 直到输出稳定
              │                              ├── 清洗 ANSI 控制码
              │                              └── 返回格式化文本 → 回复企业微信
              │
              ├── "/code ..." 且开关关闭 → 回复"功能未启用"
              │
              └── 其他消息 → 现有知识图谱(graph.invoke)
```

### Key decisions

- **tmux 管理会话**: 利用 tmux 的持久化能力，无需自研 PTY 管理。Claude Code 在 tmux 中运行与手动使用体验一致。
- **前缀路由**: `/code` 前缀区分编码命令与知识问答，零额外 LLM 调用开销。
- **技术开关**: `CLAUDE_CODE_BRIDGE_ENABLED` 环境变量控制功能启用/关闭，默认关闭。

---

## Components

### `server/config.py` — 新增配置项

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `CLAUDE_CODE_BRIDGE_ENABLED` | `bool` | `false` | 功能总开关 |

### `server/claude_bridge.py` — 新建，核心桥接模块

| 方法 | 职责 |
|------|------|
| `ClaudeCodeBridge.__init__()` | 初始化配置，设置 tmux session 名称 |
| `ClaudeCodeBridge.handle(message: str) -> str` | 入口：处理单条 `/code` 消息 |
| `ClaudeCodeBridge._ensure_session()` | 检查 tmux 会话，不存在则创建 |
| `ClaudeCodeBridge._send(message: str)` | `tmux send-keys -t {session}` |
| `ClaudeCodeBridge._capture() -> str` | `tmux capture-pane` + 清洗 |
| `ClaudeCodeBridge._wait_for_output() -> str` | 轮询直到输出稳定 |
| `ClaudeCodeBridge._cleanup()` | 退出时销毁 tmux 会话 |
| `_strip_ansi(text: str) -> str` | 模块级：清洗 ANSI 控制码 |
| `_split_long_message(text: str, max_len: int) -> list[str]` | 模块级：长消息分段 |

### `server/bot.py` — 修改，轻量路由

在 `_on_text` 中，知识图谱流程前插入 5-8 行路由逻辑：

```python
# 在 content 判空之后，graph.invoke 之前插入
if content.startswith("/code"):
    if CLAUDE_CODE_BRIDGE_ENABLED:
        response = claude_bridge.handle(content[5:].strip())
    else:
        response = "⚠️ 远程编码功能未启用，请在 .env 中设置 CLAUDE_CODE_BRIDGE_ENABLED=true"
    await self.client.reply(frame, {
        "msgtype": "markdown",
        "markdown": {"content": response},
    })
    return
```

`claude_bridge` 实例在模块级别创建，与 `graph` 同级：

```python
from server.claude_bridge import ClaudeCodeBridge

claude_bridge = ClaudeCodeBridge()
```

---

## Data Flow

### 正常流程

```
1. 用户发: "/code 查看当前 git 状态"
2. bot.py 检测 /code 前缀
3. claude_bridge.handle("查看当前 git 状态")
4. _ensure_session() → tmux 不存在则: tmux new-session -d -s claude-code 'PAGER=cat claude'
5. _send("查看当前 git 状态")
6. _wait_for_output()
   a. 每 0.5s 执行 tmux capture-pane
   b. 与新 capture 结果比较
   c. 连续 2 次无变化 → 认为输出完毕
   d. 超时 120s → 返回已有内容
7. _strip_ansi() 清洗
8. 返回文本 → 企业微信回复
```

### 边界情况

| 情况 | 处理 |
|------|------|
| Claude Code 未安装 | `_ensure_session` 检查 `which claude`，失败则回复错误提示 |
| tmux 不可用 | 同样检查，提示安装 tmux |
| 会话崩溃 | `_ensure_session` 检测到 tmux session 不存在时自动重建 |
| 命令超时 (120s) | 返回已捕获的输出，告知超时 |
| 输出过长 | `_split_long_message` 按 ≤4000 字符分段 |
| `/code exit` | 发送 exit 到 tmux，然后销毁 session |
| 多条消息并发 | 用户单聊，无并发问题 |
| `/code` 后无内容 | 回复简短使用说明 |

---

## Session Lifecycle

```
bot 启动 → 无 tmux 会话（延迟创建）

第一条 /code 消息
  → _ensure_session()
    → tmux new-session -d -s claude-code 'env PAGER=cat claude'
    → 等待 claude 就绪（检测 prompt 出现）

后续 /code 消息
  → 直接复用会话
  → _send → _wait_for_output → 返回

/code exit
  → _send("exit") → tmux session 自动结束
  → _cleanup() → 重置内部状态

bot 关闭
  → 注册 atexit → _cleanup() 销毁 tmux 会话

会话异常退出
  → 下次 _ensure_session 检测 session 不存在 → 自动重建
```

---

## Error Handling

| 错误 | 表现 |
|------|------|
| claude 命令找不到 | `handle()` 返回"Claude Code 未安装，请先安装" |
| tmux 不可用 | 同上，提示安装 tmux |
| 会话创建失败 | 返回"无法创建编码会话，请检查环境" |
| 命令执行超时 | 返回已有输出 + "命令执行超时" |
| 企业微信回复失败 | bot 已有异常处理，不影响 tmux 会话 |

---

## Testing

| 测试 | 内容 |
|------|------|
| `test_strip_ansi()` | 清洗各种 ANSI 控制码 |
| `test_split_long_message()` | 分段逻辑，边界情况 |
| `test_ensure_session_creates()` | mock subprocess，验证 tmux 命令 |
| `test_ensure_session_reuses()` | session 已存在时不再创建 |
| `test_handle_prefix()` | `/code` 前缀正确处理 |
| `test_handle_disabled()` | 开关关闭时返回未启用提示 |
| `test_wait_for_output_timeout()` | 超时场景 |
| `test_capture_parsing()` | capture-pane 输出正确解析 |

---

## Files Changed

| File | Action | Responsibility |
|------|--------|---------------|
| `server/claude_bridge.py` | Create | tmux 会话管理、消息收发、输出清洗 |
| `server/config.py` | Modify | 添加 `CLAUDE_CODE_BRIDGE_ENABLED` |
| `server/bot.py` | Modify | 添加 `/code` 前缀路由 |
| `tests/test_claude_bridge.py` | Create | 单元测试 |
