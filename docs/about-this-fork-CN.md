# 关于此 Fork

本仓库是 [OpenEvolve](https://github.com/codelion/openevolve) 的一个 fork，区别如下：

1. 支持整代码库进化，使 OpenEvolve 成为 [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms) 的一个完整复现：
   1. 现在进化个体是一个 commit，可能修改多个文件；
   2. 使用局部敏感哈希来增量、可靠和快速地计算相似度；
   3. 新的进化个体是父代的下一个 commit，从而 Git 历史为进化过程提供了天然的可解释性；
2. 充分利用 KV-cache，在避免成本暴涨的同时为模型提供更丰富的上下文，从而提高表现；
3. 现在 OpenEvolve 自由调用工具而不是运行固定的工作流；
4. 一些其他小优化和 bugfix。
