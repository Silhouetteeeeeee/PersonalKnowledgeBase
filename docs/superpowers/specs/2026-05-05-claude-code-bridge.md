# Claude Code Bridge — 企业微信远程编码桥接

**Problem:** Windows 环境下 tmux 存在兼容性问题（MSYS2 管道不通、中文编码错误），需要通过更可靠的方式实现企业微信远程控制 Claude Code。

**Solution:** 放弃 tmux，改用 `claude --print` 非交互模式。每条 `/code` 消息调用一次 `claude --print "消息"`，后续消息加上 `--continue` 维持对话上下文。Claude Code 自身的会话管理取代 tmux 持久会话。

**Non-goal:** 实时流式输出、多人协作、文件上传下载（已有文件处理）。

---

## Architecture

```
企业微信 → bot.py (_on_text)
              │
              ├── "/code ..." 且开关打开 → claude_bridge.handle()
              │                              │
              │                              ├── 首条消息: claude --print "消息"
              │                              ├── 后续消息: claude --print --continue "消息"
              │                              ├── 捕获 stdout 文本
              │                              └── 长消息分段 → 回复企业微信
              │
              ├── "/code ..." 且开关关闭 → 回复"功能未启用"
              │
              └── 其他消息 → 现有知识图谱(graph.invoke)
```

### Key decisions

- **`claude --print` 替代 tmux**: 无持久进程、无 PTY、无管道兼容性问题。Windows/Unix 均可靠。
- **`--continue` 维持上下文**: Claude Code 自动管理会话文件，无需手动维护状态。
- **前缀路由 + 技术开关**: 保持不变，`/code` 前缀 + `CLAUDE_CODE_BRIDGE_ENABLED`。

---

## Components

### `server/claude_bridge.py` — 重写，精简桥接模块

| 方法 | 职责 |
|------|------|
| `ClaudeCodeBridge.__init__()` | 初始化，注册 atexit |
| `ClaudeCodeBridge.handle(message: str) -> str` | 入口：执行 `claude --print` 并返回结果 |
| `_check_claude() -> bool` | 模块级：检测 claude 是否可用 |
| `_split_long_message(text, max_len) -> list[str]` | 模块级：长消息分段 |

`handle()` 内部逻辑：

```python
def handle(self, message: str) -> str:
    if not message: return 用法说明
    if message == "exit": return 已关闭

    cmd = ["claude", "--print", message]
    if self._has_conversation:
        cmd.append("--continue")

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        self._has_conversation = True
        output = r.stdout.strip()
        return 分段后的输出 or 错误信息
    except subprocess.TimeoutExpired:
        return 超时提示
    except FileNotFoundError:
        return 未安装提示
```

### `server/bot.py` — **无改动**

路由代码和 bridge 实例化已在之前提交中完成，保持不变。

### `server/config.py` — **无改动**

`CLAUDE_CODE_BRIDGE_ENABLED` 标志已在之前提交中完成，保持不变。

---

## Data Flow

### 正常流程

```
1. 用户: "/code 查看当前 git 状态"
2. bot.py 检测 /code 前缀
3. claude_bridge.handle("查看当前 git 状态")
4. subprocess.run(["claude", "--print", "查看当前 git 状态"])    [首次]
   或 subprocess.run(["claude", "--print", "--continue", "消息"]) [后续]
5. claude 在当前项目目录执行命令，输出结果到 stdout
6. stdout → split_long_message → 企业微信回复
7. 进程退出，Claude Code 自动保存会话到 ~/.claude/projects/
```

### 边界情况

| 情况 | 处理 |
|------|------|
| claude 未安装 | `subprocess.run` 抛 `FileNotFoundError` → 回复"请先安装" |
| 命令超时 (120s) | 回复"命令执行超时" |
| 输出过长 | `_split_long_message` 按 ≤4000 字符分段 |
| `/code exit` | 重置 `_has_conversation = False`，下次消息重新开始 |
| 机器人重启 | `_has_conversation = False`，自动开始新会话 |
| 多条消息并发 | 单聊场景无并发问题；如有需要可通过锁排队 |
| `/code` 后无内容 | 回复简短使用说明 |

---

## 从旧方案迁移（tmux → claude -p）

旧方案（tmux）的代码和提交无需删除。`server/claude_bridge.py` 将被整体重写，外部接口不变（`handle(message) -> str`），bot.py 无需任何修改。

### 删除的复杂性

| 旧方案 | 新方案 |
|--------|--------|
| tmux session 管理 (`_ensure_session`, `_cleanup`) | 无 |
| `tmux send-keys` 发送消息 | `subprocess.run` |
| `tmux capture-pane` 轮询 | `capture_output=True` |
| ANSI 控制码清洗 (`_strip_ansi`) | claude --print 输出纯文本 |
| Windows/bash 路由 (`_run_tmux`, `_BASH_PATH`) | 无 |
| 输出稳定性检测 (`_wait_for_output`, `STABLE_THRESHOLD`) | 无 |
| 临时文件捕获 (`_capture_pane` tempfile) | 无 |
| 提示符检测 (`_parse_prompt_from_output`) | 无 |

保留的工具函数：
- `_split_long_message` — 仍然需要（企业微信消息长度限制）

---

## 兼容性

| 平台 | claude --print | tmux 方案 |
|------|---------------|-----------|
| Windows (原生 Python) | ✅ 正常工作 | ❌ 管道/编码问题 |
| Windows (Git Bash) | ✅ | ❌ |
| macOS/Linux | ✅ | ✅ 但不再需要 |
| 核心依赖 | 仅 claude 命令 | tmux + claude |

---

## Testing

| 测试 | 内容 |
|------|------|
| `test_handle_empty()` | 空消息返回用法说明 |
| `test_handle_exit()` | `/code exit` 重置会话状态 |
| `test_first_call_no_continue()` | 首次调用不加 `--continue` |
| `test_subsequent_call_with_continue()` | 后续调用加 `--continue` |
| `test_timeout_returns_error()` | mock timeout，返回超时提示 |
| `test_claude_not_found()` | mock FileNotFoundError，返回安装提示 |
| `test_split_long_message()` | 长消息分段逻辑 |

---

## Files Changed

| File | Action | Responsibility |
|------|--------|---------------|
| `server/claude_bridge.py` | Rewrite | 从 tmux 方案改为 `claude --print` 方案 |
| `tests/test_claude_bridge.py` | Rewrite | 更新测试 mock claude 而非 tmux |
