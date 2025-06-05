#!/bin/zsh

curl -X POST "https://your-app--ollama.modal.run/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3:instruct", "messages": [{"role": "user", "content": "Hello!"}]}' 