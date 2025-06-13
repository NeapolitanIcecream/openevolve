# LLM 模块架构

本文档描述了 `llm` 模块的架构、功能和接口设计。

## 1. 架构

`llm` 模块负责封装与各种大型语言模型（LLM）服务交互的逻辑。它被设计为一个可扩展的系统，允许轻松添加对新模型的支持。

- **`base.py`**: 定义了所有 LLM 客户端必须实现的抽象基类 `LLMInterface`。这个基类规定了所有模型接口的统一契约，主要是 `generate_with_context` 方法，它接收结构化的消息列表进行对话式生成。
- **`openai.py`**: 提供了与 OpenAI API（例如 GPT-3.5, GPT-4）交互的具体实现。`OpenAILLM` 类继承自 `LLMInterface` 基类。
- **`ensemble.py`**: 实现了一个 `LLMEnsemble` 类，它可以管理多个 `OpenAILLM` 客户端实例。它的主要生成方法通过加权随机采样来选择一个模型进行调用，而不是并行调用所有模型。它不直接实现 `LLMInterface`，而是一个高级的管理器。该设计旨在未来支持更多不同类型的LLM客户端。
- **`__init__.py`**: 作为包的入口，将模块内的主要类（如 `OpenAILLM`, `LLMEnsemble`, `LLMInterface`）暴露出来，供其他模块直接导入和使用。

## 2. 功能

- **代码生成**: 模块的主要功能是接收一个结构化的提示（prompt），并调用指定的 LLM 服务来生成代码。
- **模型抽象**: 将不同 LLM 服务的 API 差异抽象掉，为上层模块（如 `controller`）提供一个统一的调用接口 `LLMInterface`。
- **可扩展性**: 用户可以通过继承 `LLMInterface` 基类并实现其方法，来添加对新的 LLM（如 Google Gemini, Anthropic Claude 等）的支持。
- **模型管理**: `LLMEnsemble` 类提供了一种管理和使用多个 LLM 配置的机制。

## 3. 接口设计

- **输入**: `LLMInterface` 的核心方法 `generate_with_context` 接受一个系统消息字符串和代表对话历史的消息列表。
- **输出**: `generate_with_context` 方法返回一个字符串，即模型生成的代码或文本。
- **配置**: 模块通过顶层的 `config.py` 进行配置。`controller` 模块负责根据配置来直接实例化 `LLMEnsemble` 类。这种设计将模型创建的职责交给了使用者。
- **与外部模块的交互**: `controller` 是本模块的主要使用者。它直接创建 `LLMEnsemble` 实例，然后调用该实例的 `generate_with_context()` 方法来驱动代码的进化生成。 