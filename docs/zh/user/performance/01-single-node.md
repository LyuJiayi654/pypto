# 单机手段

一份目录，不是一张清单。每条都写明适用场景、代价，以及最要紧的那件事——**怎么确认它起作用了**。

> **前置**：[度量闭环](00-methodology.md)。来这一页时，闭环里应当已经有某一步指向了某个地方；没有度量就照目录施加手段，正是 kernel 变慢的由来。

## 怎么读一个条目

下面每种手段都带上一页那四栏。**「怎么确认」不是可选项**——一个你在那些工具里都看不见的改动，是一个你无法辩护的改动，也会是别人第一个删掉的东西。

## 切分与并行

| 手段 | 适用场景 |
| ---- | -------- |
| `pl.at(level=...)` | 决定某个区域到底跑在哪一层硬件上 |
| `pl.spmd(N)` | 同一个 kernel 跑在 N 个互相独立的 block 上 |
| `pl.split(SplitMode)` | 一个作用域里混有 cube 与 vector 工作 |
| `pl.split_aiv(n, mode=)` | `pl.split` 表达不了的形状 —— 逐通道寻址 |
| `pl.cluster()` | 应当在一个 cluster 内协同调度的工作 |

**`pl.spmd(N)`** —— *代价：* 它是一个**断言**，声称这 N 个 block 互相独立。如果它们并不独立，你得到的是竞态而不是诊断。*确认：* 依赖图（`enable_dep_gen`）里应当出现 N 个兄弟节点，而不是一条链。

**`pl.split(SplitMode)`** —— cube/vector 重叠，见 [混合 kernel](../tutorials/03-mixed-kernel.md)。*代价：* 跨核环是按边界上的整 tile 计量的，所以它可能把 kernel 顶出 vector 预算——要与 `pl.cross_core_slot` 配合。*确认：* L2 swimlane 里 cube 与 vector 的 span 应当**重叠**而不是首尾相接。

> **致命陷阱：** `pl.split` 只对 **vector** 子区域对半；cube 子区域保持全尺寸。如果你选 mode 时以为两个单元各拿半个 tile，测出来的结果不会符合那个心智模型。

## 流水与展开

| 手段 | 适用场景 |
| ---- | -------- |
| `pl.pipeline(..., stage=N)` | 迭代之间可以重叠在飞的循环 |
| `pl.unroll` | 逐迭代开销占主导的短循环 |
| `pl.cross_core_slot(slot_num=N)` | 给 cube↔vector 的环定尺寸 |

**`pl.pipeline`** —— *代价：* 缓冲区按阶段数增加。重叠 N 个阶段意味着被搬运的 tile 有 N 份同时存活。*确认：* [memory map](00-methodology.md) 里会看到多出来的那几份；如果分配没变大，说明流水没生效。

**`pl.cross_core_slot(slot_num=N)`** —— *代价与收益是同一件事：* 更深的环让生产者在阻塞前跑得更靠前，代价是 `N × tile` 的 vector 缓冲区。默认是 8 槽，对 `[128, 128]` FP32 的 tile 就是 512 KB，而预算约 184 KB —— 所以在真实形状上这个参数通常是往**小**调而不是往大调。**在装得下的前提下选最大的深度。** *确认：* 首先是能编译通过；然后看 swimlane，判断生产者是否仍在卡。

## matmul 通路

| 手段 | 适用场景 |
| ---- | -------- |
| `AutoTileMatmulL0` | 自动 —— 装不进 cube L0 缓冲区的 matmul |
| `enable_pypto_l0c_double_buffer` | L0c 还有余量，且 matmul 是累加受限的 |
| `a_trans` / `b_trans` | 某个操作数的排布与 cube 想要的不一致 |
| split-K + 原子加 | K 很大，而 M/N 填不满这些核 |

**`AutoTileMatmulL0`** 无论你要不要都会跑，这一点值得知道，恰恰因为它使得 tensor 级操作数上的 `pl.matmul` **不是一条指令**——它是该 pass 写出来的一个循环嵌套。手工分块 K（见 [分块 matmul](../tutorials/02-matmul.md)）只在那一个轴上覆盖它的选择。

**split-K** —— *代价：* **跨核的累加顺序不固定**，所以重复运行的末位可能不同，而且输出必须先清零。*确认：* 图里应当出现 K 分片作为兄弟节点；同时检查你测试所用的容差是否仍然成立。

## 内存

| 手段 | 适用场景 |
| ---- | -------- |
| `pl.create_tile` 上的 `target_memory=` | 编译器把某个 tile 放到了不合适的地方 |
| `memory_planner=` | 在 PyPTO 的规划器与 PTOAS 的之间做选择 |
| 数据常驻 | 同一批权重跨多次启动使用 |

**`memory_planner=PTOAS`** 把分配交给 PTOAS，跳过 PyPTO 的 `MemoryReuse` 与 `AllocateMemoryAddr`。*代价：* 不同的失效模式和另一组规划器缺陷；语义必需的 alias 两条路都照跑。*确认：* **不要**用 memory map —— PTOAS 下编译器跳过 `AllocateMemoryAddr`，pass dump 里没有已分配的偏移可供该工具绘制。改用端到端对比：benchmark 树，以及在另一个规划器拒绝的形状上它能否编译通过。

**数据常驻** —— `pypto.runtime.DeviceTensor` 让张量跨启动留在设备上，省掉每次调用的一次 H2D 拷贝。*何时：* 权重、KV cache，任何内容比单次启动活得久的东西。*确认：* benchmark 树里的 **host** span 变小；device span 不应移动。

## 调度

这些是 [塑形任务图](../tutorials/04-task-graph.md) 的主题；这里列出是为了给出它们的代价。

| 手段 | 改变正确性 | 代价 |
| ---- | ---------- | ---- |
| `no_dep` / `manual_dep=True` / `manual_scope` | **是** | 一个编译器无法检验的断言 |
| `predicate=` | 否 | 可表达的只有 `tensor[indices] OP int 字面量` |
| `allow_early_resolve=` | 否 | 消费者只有在其*所有*生产者都被标记后才预置 |
| `pl.system.task_dummy` | 否 | 多一个任务，用来收敛扇入 |
| `ring_task_window` / `ring_heap` / `ring_dep_pool` | 否 | 运行时内存 |

*以上全部的确认方式：* `enable_dep_gen` 看图的形状，swimlane 看这个形状有没有变成真正的重叠。环的尺寸是唯一能由 `scope_stats` 直接回答的一项 —— 如果峰值本来就没顶到容量，调大那个环不可能有用。

## 只付一次 setup

worker 的 setup 是按 worker 而非按 program 计的。把若干 program 注册到同一个 worker 上，就能让第一次之后的每次运行都省掉一整次 setup —— 见 `examples/runtime/multi_program_kv_cache.py`，那里一个 prefill 与一个 decode program 共享一份 KV cache 和一个 worker。

*确认：* 同样是 host span。这件事永远不会体现在 device 时间上。

## 进攻顺序

当度量指向设备、而上面这些手段你有好几个可选时：

1. **先修 `perf_hints` 标出来的。** 一个 `PH001` 的 tile 粒度问题会给其余一切封顶。
2. **先修图，再修 kernel** —— 串行的图会把 kernel 省下来的时间浪费掉。
3. **先让两个单元重叠**（`pl.split`），再去微调其中任何一个。
4. **然后**才是流水深度、环深度、分配。

把第 2 步和第 3 步的顺序颠倒是最常见的错误：一个做得再漂亮的混合 kernel，如果图里没有任何东西被允许与它并行，它仍然是自己一个人在跑。

## 参见

- [度量闭环](00-methodology.md) —— 先决定**改什么**。
- [分布式性能](02-distributed.md) —— 多 rank 之后有什么不同。
- [混合 kernel](../tutorials/03-mixed-kernel.md) —— `pl.split` 动手篇。
- [塑形任务图](../tutorials/04-task-graph.md) —— 调度控制项。
