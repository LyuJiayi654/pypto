# 度量闭环

五个问题，按顺序问。每个问题都有对应工具，而每个答案都决定下一个问题值不值得问。

> **前置**：[调度调优](../tutorials/05-scheduling-tuning.md) 讲的是这套流程里调度的那一半；本页是完整的闭环。

## 顺序为什么重要

这些工具个个都乐意给你一个数字。顺序之所以存在，是因为：当第 1 步已经告诉你 kernel 写法有问题时，第 3 步的数字毫无意义；当第 2 步显示时间花在 host 上时，第 5 步的调度优化是白做。

```text
性能不达预期
├─ 1. 编译器是不是已经说了？   → report/perf_hints.log
├─ 2. 是哪一段？               → benchmark span 树（host vs device）
├─ 3. 单个 kernel 内部？       → incore msprof，逐指令
├─ 4. 资源是满了还是浪费了？   → memory map、scope stats
└─ 5. 调度被串行化了？         → 依赖图、L2 swimlane
```

从上往下走，在第一个能解释差距的步骤停下。

## 工具

| 步骤 | 观测对象 | 入口 | 产物 |
| ---- | -------- | ---- | ---- |
| 1 | 代码模式 | `compile()` / `@pl.jit` 期间自动 | `<output_dir>/report/perf_hints.log` |
| 2 | 端到端分段 | `pypto.runtime.benchmark` | `stats.print_mean_tree(...)` |
| 3 | 单 kernel，cycle 级 | Ascend op-simulator（`msprof`） | Insight trace |
| 4 | 片上缓冲区 | `python -m pypto.tools.memory_map DUMP.py` | HTML |
| 4 | 运行时环水位 | `RunConfig(enable_scope_stats=True)` | `dfx_outputs/scope_stats/scope_stats.jsonl` |
| 3 | 逐 pipe 利用率 | `RunConfig(enable_pmu=2)` | `dfx_outputs/pmu.csv` |
| 5 | 任务图、并发 | `RunConfig(enable_dep_gen / enable_l2_swimlane)` | `dfx_outputs/*` |

## 第 1 步：先看编译器已经说过什么

性能提示是编译期发出的 —— 你不需要打开它。只要上下文里有 report instrument（走 `compile()` 与 `@pl.jit` 时永远有），每条提示都会追加进 `perf_hints.log`，stderr 上给一行指向该文件的摘要：

```text
[perf_hint] N hints across M sites; see <path>/perf_hints.log
```

`PYPTO_LOG_LEVEL` 的发布默认值是 `INFO`，所以这行摘要不需要任何配置就会出现在控制台。`PYPTO_LOG_LEVEL=warn` 可以静音它——文件里的明细照旧。

第一个值得记住名字的检查是 **`PH001` `TileInnermostDimGranularity`**：tile 的最内维没有落在硬件搬运的粒度上。这是**代码模式**问题而不是调度问题，所以本列表往下的任何调优都补不回来。完整注册表与 `disabled_diagnostics` 见 [诊断](../../dev/passes/92-diagnostics.md)。

## 第 2 步：优化任何东西之前，先找到那一段

`benchmark` 把编译产物注册一次，然后派发 N 次计时启动，所以数字里不含 setup：

```python
from pypto.runtime import benchmark

stats = benchmark(compiled, args, rounds=100, warmup=3)
stats.print_mean_tree(spread="stdev")
```

`warmup=3` 是有意义的：最初几次启动要付一次性开销，否则它会被平摊进每一轮。

这棵树把 **host** 与 **device** 的 span 分开。这个分野正是本步骤的意义：

- 时间在 device span → 继续第 3 步或第 5 步。
- 时间在 host span → 改 kernel 毫无用处。去看派发开销、参数编排，或者你在按次付、本该只付一次的 setup（见 [单机手段](01-single-node.md)）。

`spread=` 决定每个均值旁边打印什么；`stdev` 是个好默认值，因为离散度大通常意味着你量的不是 kernel。

## 第 3 步：单个 kernel 内部

当某个 device span 占主导、且它就是一个 kernel 时，op-simulator 给出逐指令、逐 pipe 的时序，可用 Insight trace 查看。这是这里唯一能告诉你**哪条 pipe 是上限**、而不只是「这个 kernel 慢」的工具。

`RunConfig(enable_pmu=2)`（`PIPE_UTILIZATION`）是更便宜的近似，写出 `dfx_outputs/pmu.csv` —— 用它来判断值不值得跑一次完整的模拟器。

## 第 4 步：资源是满了，还是浪费了？

两种不同资源，两个工具：

**片上缓冲区** —— `pypto.tools.memory_map` 把分配渲染成 HTML：横轴地址、纵轴生命期，旁边并排 IR。它的输入是 **pass dump** 而不是一次运行，所以先在编译期要一份：

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

prog = kernel.lower(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
```

```bash
DUMP=path/to/output_dir/passes_dump/NN_after_SomePass.py
python -m pypto.tools.memory_map "$DUMP" -o map.html
```

读它是为了找出活得比需要更久的 tile，以及那些决定「更深的流水或更深的跨核环装不装得下」的余量。

**运行时的环** —— `RunConfig(enable_scope_stats=True)` 按作用域记录 `task_window`、`heap`、`tensormap` 三个环的峰值。峰值顶到容量就是天花板；峰值远低于容量则说明环不是你的问题，你不该去调它们的大小。

## 第 5 步：调度是不是上限？

动手部分见 [调度调优](../tutorials/05-scheduling-tuning.md)。简言之：`enable_dep_gen=True` 看运行时建出来的图，`enable_l2_swimlane=True` 看任务到底有没有重叠。

> **模拟器注意事项：** 在 `*sim` 平台上 swimlane 是单趟的，只产出 `l2_swimlane_records.json`。在真机平台上同一个开关会把负载**跑两遍**，因为采集会扰动时序 —— 所以绝不要从开了 swimlane 的真机运行里读挂钟时间。

## 怎么评价一次改动

后面两页里的每种手段都按同样四栏描述，因为一个不说明代价的加速不算结果：

| 栏目 | 回答什么 |
| ---- | -------- |
| **适用场景** | 必须成立的形状或模式 |
| **代价** | 你放弃了什么 —— 确定性、内存、可读性、可移植性 |
| **怎么开** | 确切的开关、关键字或改写 |
| **怎么确认生效** | 上面那些工具中的哪一个能看出来 |

第四栏是最常被跳过的那一栏。一个无法用上述工具确认的改动就是一次猜测，而猜测会累积。

## 参见

- [单机手段](01-single-node.md) —— 单卡的手段目录。
- [分布式性能](02-distributed.md) —— 多 rank 之后有什么不同。
- [实例](03-cases.md) —— 闭环的端到端应用。
- [诊断](../../dev/passes/92-diagnostics.md) —— 编译期提示注册表。
