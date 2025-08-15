# llm-huggingface

[![PyPI - Version](https://img.shields.io/pypi/v/llm-huggingface-plugin)](https://pypi.org/project/llm-huggingface-plugin/)
[![GitHub Release](https://img.shields.io/github/v/release/SecKatie/llm-huggingface-plugin)](https://github.com/SecKatie/llm-huggingface-plugin/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/SecKatie/llm-huggingface-plugin/blob/main/LICENSE)

Access Hugging Face models via the Inference API

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-huggingface-plugin
```

## Configuration

Configure the plugin by setting your [Hugging Face API token](https://huggingface.co/settings/tokens):
```bash
llm keys set huggingface
```
```
<paste key here>
```
You can also set the API key by assigning it to the environment variable `HUGGINGFACE_TOKEN`.

## Usage

The plugin automatically discovers and registers all available text-generation and image-text-to-text (multimodal) models from Hugging Face. Run a model using the `hf/` prefix:

```bash
llm -m hf/meta-llama/Llama-3.2-3B-Instruct "Write a haiku about coding"
```

You can list all available Hugging Face models:
```bash
llm models -q hf/
```

Set a default model to avoid the `-m` option:
```bash
llm models default hf/mistralai/Mistral-7B-Instruct-v0.3
llm "Explain quantum computing in simple terms"
```

## Features

### Streaming Responses

The plugin supports streaming for real-time output:
```bash
llm -m hf/meta-llama/Llama-3.2-3B-Instruct "Tell me a story" --stream
```

### JSON Schema Output

Force structured JSON output using schemas:
```bash
llm -m hf/meta-llama/Llama-3.2-3B-Instruct \
  --schema '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}' \
  "Generate a person's profile"
```

Or use the simpler DSL format:
```bash
llm -m hf/meta-llama/Llama-3.2-3B-Instruct \
  --schema 'name: str, age: int, hobbies: list[str]' \
  "Generate a person's profile with hobbies"
```

**Note:** JSON schema support varies by model. Some models may not fully support structured output or may produce inconsistent results. Models specifically fine-tuned for instruction following and structured output generation tend to perform better with schemas.

### Function Calling / Tools

The plugin supports function calling for models that have this capability:
```bash
llm -m hf/meta-llama/Llama-3.2-3B-Instruct \
  --tool calculate 'def calculate(expression: str) -> float: """Evaluate a mathematical expression"""' \
  "What is 15% of 240?"
```

### Generation Parameters

Control generation with various parameters:

- **Temperature** (0.0-2.0): Controls randomness
  ```bash
  llm -m hf/meta-llama/Llama-3.2-3B-Instruct -o temperature 0.7 "Write creatively"
  ```

- **Top-p** (0.0-1.0): Nucleus sampling threshold  
  ```bash
  llm -m hf/meta-llama/Llama-3.2-3B-Instruct -o top_p 0.9 "Generate text"
  ```

- **Max tokens**: Limit response length
  ```bash
  llm -m hf/meta-llama/Llama-3.2-3B-Instruct -o max_tokens 100 "Explain AI"
  ```

- **Stop sequences**: Halt generation at specific strings
  ```bash
  llm -m hf/meta-llama/Llama-3.2-3B-Instruct -o stop '[".", "!"]' "Generate until punctuation"
  ```

### Multimodal Support

The plugin supports image attachments for vision models that support the `image-text-to-text` task:

```bash
# Analyze an image
llm -m hf/meta-llama/Llama-3.2-11B-Vision-Instruct \
  -a image.jpg "Describe this image in detail"

# Multiple images
llm -m hf/meta-llama/Llama-3.2-11B-Vision-Instruct \
  -a photo1.png -a photo2.jpg "Compare these two images"
```

Supported image formats:
- PNG (image/png)
- JPEG (image/jpeg)
- WebP (image/webp)
- GIF (image/gif)

### Interactive Chat

Start an interactive chat session:
```bash
llm chat -m hf/meta-llama/Llama-3.2-3B-Instruct
```

### Conversation History

Continue previous conversations:
```bash
llm -m hf/meta-llama/Llama-3.2-3B-Instruct "What is Python?" -c
llm -c "What are its main uses?"
```

## Advanced Options

### Token Limits

Control the maximum number of tokens in the response using the `max_tokens` parameter:

```bash
llm -m hf/meta-llama/Llama-3.2-3B-Instruct -o max_tokens 500 "Tell a story"
```

### Response Metadata

Access detailed response metadata using the Python API:
```python
import llm

model = llm.get_model("hf/meta-llama/Llama-3.2-3B-Instruct")
response = model.prompt("Hello")

# Access metadata
print(response.response_json)
# {'usage': {...}, 'model': '...', 'finish_reason': 'stop', ...}
```

## Python API

Use the plugin programmatically with both sync and async support:

### Synchronous API

```python
import llm

# Get a model
model = llm.get_model("hf/meta-llama/Llama-3.2-3B-Instruct")

# Simple prompt
response = model.prompt("Explain machine learning")
print(response.text())

# With options
response = model.prompt(
    "Write a poem",
    temperature=0.9,
    max_tokens=200
)

# Streaming
for chunk in model.prompt("Tell me a story", stream=True):
    print(chunk, end="", flush=True)

# With system prompt
response = model.prompt(
    "Translate to French: Hello",
    system="You are a helpful translation assistant."
)
```

### Asynchronous API

```python
import asyncio
import llm

async def main():
    # Get an async model
    model = llm.get_async_model("hf/meta-llama/Llama-3.2-3B-Instruct")
    
    # Async prompt
    response = await model.prompt("Explain quantum computing")
    print(response.text())
    
    # Async streaming
    async for chunk in model.prompt("Write a story", stream=True):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Multimodal API

```python
import llm

# Get a vision model
model = llm.get_model("hf/meta-llama/Llama-3.2-11B-Vision-Instruct")

# Analyze an image
response = model.prompt(
    "What's in this image?",
    attachments=[llm.Attachment(path="photo.jpg")]
)
print(response.text())
```

## Model Discovery

The plugin automatically discovers models from Hugging Face that support:
- `text-generation` task - for standard text models
- `image-text-to-text` task - for multimodal vision models

Models are registered with the `hf/` prefix to distinguish them from other LLM providers. The discovery process is cached to improve performance.

## Performance & Caching

The plugin implements intelligent caching to avoid repeated API calls to Hugging Face:

### Cache Configuration

- **Default cache duration**: 7 days
- **Cache location**: `~/.llm/llm-huggingface/`
- **Cached data**: Lists of available models by task type

### Environment Variables

- `LLM_HUGGINGFACE_CACHE_EXPIRY`: Set cache expiry in seconds (default: 604800 = 7 days)
  ```bash
  export LLM_HUGGINGFACE_CACHE_EXPIRY=86400  # 1 day
  ```

- `LLM_HUGGINGFACE_FORCE_REFRESH`: Force refresh of cached model lists
  ```bash
  export LLM_HUGGINGFACE_FORCE_REFRESH=true
  llm models -q hf/  # Will fetch fresh model list from Hugging Face
  ```

The caching system provides approximately 2000x speedup for model discovery after the initial fetch.

## Limitations

- **Attachments**: Only image attachments are supported for multimodal models. Audio and other file types are not currently supported
- **Model availability**: Not all Hugging Face models may be accessible via the Inference API
- **Rate limits**: Subject to Hugging Face API rate limits based on your account type
- **Multimodal models**: Only models that support the `image-text-to-text` task can process image attachments

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-huggingface
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```

To run the tests:
```bash
pytest
```

## License

Apache 2.0
