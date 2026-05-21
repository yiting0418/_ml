#!/usr/bin/env python3
# agent0.py - AI Agent with sandbox and manual authorization
# Run: python agent0.py

import subprocess
import os
import re
from mlx_lm import load, generate

# ─── Configuration ───

WORKSPACE = os.path.expanduser("~/.agent0")
MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MAX_TURNS = 5

# ─── Sandbox & Authorization ───

def resolve_path(path):
    """Resolve a path to absolute, handling ~ and relative paths."""
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(WORKSPACE, path)
    return os.path.abspath(path)

def is_path_safe(path):
    """Check if resolved path is within WORKSPACE using commonpath."""
    abs_path = resolve_path(path)
    ws = os.path.normpath(WORKSPACE)
    return os.path.commonpath([abs_path, ws]) == ws

AUTO_AUTHORIZE = False

def authorize(message):
    if AUTO_AUTHORIZE:
        return True
    print(f"\n⚠️  {message}")
    answer = input("是否核可？(y/N): ").strip().lower()
    return answer == "y"

def handle_read_file(path):
    safe = is_path_safe(path)
    if not safe and not authorize(f"LLM 請求讀取外部檔案：{path}"):
        return "Error: Permission denied by user."
    target = resolve_path(path)
    if not safe:
        print(f"  外部路徑：{target}")
    try:
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error: {e}"

def handle_write_file(path, content):
    safe = is_path_safe(path)
    if not safe and not authorize(f"LLM 請求寫入外部檔案：{path}"):
        return "Error: Permission denied by user."
    target = resolve_path(path)
    if not safe:
        print(f"  外部路徑：{target}")
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {target}"
    except Exception as e:
        return f"Error: {e}"

def extract_tools(text):
    """Extract all tool calls from LLM response, preserving order."""
    text = re.sub(r'```\w*\n?', '', text)
    text = text.replace('```', '')
    tools = []
    for m in re.finditer(r'<shell>(.+?)</shell>', text, re.DOTALL):
        tools.append((m.start(), 'shell', m.group(1).strip(), None))
    for m in re.finditer(r'<read_file\s+path="([^"]*)"\s*/?>', text):
        tools.append((m.start(), 'read_file', m.group(1).strip(), None))
    for m in re.finditer(r'<write_file\s+path="([^"]*)">(.*?)</write_file>', text, re.DOTALL):
        tools.append((m.start(), 'write_file', m.group(1).strip(), m.group(2)))
    tools.sort(key=lambda x: x[0])
    return tools

# ─── Memory ───

conversation_history = []
key_info = []

# ─── MLX Model ───

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None:
        print(f"正在載入模型 {MODEL}...")
        _model, _tokenizer = load(MODEL)
        print("模型載入完成！")
    return _model, _tokenizer

def call_ollama(prompt: str, system: str = "") -> str:
    """Call MLX model for generation"""
    model, tokenizer = load_model()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=formatted, max_tokens=3072, verbose=False)
    return response.strip()

# ─── Memory Management ───

def build_context():
    context_parts = []
    if key_info:
        items_xml = "\n".join(f"  <item>{k}</item>" for k in key_info)
        context_parts.append(f"<memory>\n{items_xml}\n</memory>")
    if conversation_history:
        context_parts.append("<history>\n" + "\n".join(conversation_history[-MAX_TURNS*2:]) + "\n</history>")
    return "\n\n".join(context_parts)

def update_memory(user_input, assistant_response, tool_result=None):
    conversation_history.append(f"  <user>{user_input}</user>")
    conversation_history.append(f"  <assistant>{assistant_response}</assistant>")
    if tool_result:
        conversation_history.append(f"  <tool>{tool_result[:500]}</tool>")
    
    while len(conversation_history) > MAX_TURNS * 4:
        conversation_history.pop(0)

def extract_key_info(user_input, assistant_response):
    extract_prompt = f"""根據這段對話，有沒有需要長期記憶的關鍵資訊？
如果有，用以下格式輸出（最多 2 項）。如果沒有，輸出 <memory></memory>。

<memory>
  <item>要記憶的資訊 1</item>
  <item>要記憶的資訊 2</item>
</memory>

對話：
<user>{user_input}</user>
<assistant>{assistant_response}</assistant>"""
    
    try:
        result = call_ollama(extract_prompt, "")
        matches = re.findall(r'<item>(.*?)</item>', result, re.DOTALL)
        for item in matches:
            item = item.strip()
            if item and item not in key_info:
                key_info.append(item)
    except:
        pass

# ─── Agent ───

SYSTEM_PROMPT = """你是一個只能輸出 XML 工具標籤的 AI 助手。

可用工具：
1. <shell>命令</shell> — shell 命令（用於 mkdir、npm install 等）
2. <write_file path="路徑">內容</write_file> — 寫入完整的檔案內容
3. <end/> — 結束

重要規則：
- 檔案內容（程式碼）請用 <write_file>，不要用 touch
- 建立目錄用 <shell>mkdir -p 目錄</shell>
- 多個步驟可以分多次輸出工具標籤
- 完成後必須輸出 <end/>

流程範例：
<shell>mkdir -p blog2 && cd blog2 && npm init -y</shell>
<write_file path="blog2/app.js">const express = require('express');...</write_file>
<end/>

禁止輸出工具標籤以外的任何文字。"""

def main():
    import sys
    global AUTO_AUTHORIZE, WORKSPACE
    if "--auto" in sys.argv:
        AUTO_AUTHORIZE = True
    for i, arg in enumerate(sys.argv):
        if arg == "--workspace" and i + 1 < len(sys.argv):
            WORKSPACE = os.path.abspath(sys.argv[i + 1])
    
    os.makedirs(WORKSPACE, exist_ok=True)
    
    print(f"Agent0 - Qwen2.5-1.5B（MLX 本機模型，含沙盒與授權機制）")
    print(f"工作區：{WORKSPACE}")
    print("指令：/quit、/memory（顯示關鍵資訊）\n")
    
    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ["/quit", "/exit", "/q"]:
            print("再見！")
            break
        if user_input.lower() == "/memory":
            print(f"關鍵資訊：{key_info}")
            continue
        
        context = build_context()
        if context:
            full_prompt = f"對話記錄：\n{context}\n\n使用者現在要求：{user_input}"
        else:
            full_prompt = f"使用者要求：{user_input}"
        
        response = call_ollama(full_prompt, SYSTEM_PROMPT)
        
        tool_result = None
        current_response = response
        max_tool_loops = 5
        tool_loop_count = 0
        
        while True:
            tool_loop_count += 1
            if tool_loop_count > max_tool_loops:
                break
            
            has_end = "<end/>" in current_response
            if has_end:
                text_before_end = current_response.split("<end/>")[0].strip()
            else:
                text_before_end = current_response
            
            tools = extract_tools(text_before_end)
            if not tools:
                response = text_before_end
                break
            
            all_outputs = []
            for _, tool_type, arg1, arg2 in tools:
                result = None
                
                if tool_type == 'shell':
                    cmd = arg1
                    print(f"\n=== 執行命令 ===\n{cmd}")
                    if not authorize("是否允許執行此 shell 命令？"):
                        result = "Error: Permission denied by user."
                    else:
                        try:
                            r = subprocess.run(
                                cmd, shell=True, capture_output=True, text=True,
                                timeout=30, cwd=WORKSPACE
                            )
                            result = r.stdout + r.stderr
                        except Exception as e:
                            result = f"Error: {e}"
                    output = result if result else "（無輸出）"
                    print(f"結果：{output}\n")
                    all_outputs.append(f"$ {cmd}\n{output}")
                
                elif tool_type == 'read_file':
                    path = arg1
                    result = handle_read_file(path)
                    output = result if result else "（空檔案）"
                    print(f"\n=== 讀取檔案 ===\n{path}\n\n結果：{output[:300]}{'…' if len(output) > 300 else ''}\n")
                    all_outputs.append(f"<read_file path=\"{path}\">\n{output}")
                
                elif tool_type == 'write_file':
                    path = arg1
                    content = arg2
                    result = handle_write_file(path, content)
                    print(f"\n=== 寫入檔案 ===\n{path}\n\n結果：{result}\n")
                    all_outputs.append(f"<write_file path=\"{path}\">\n{result}")
            
            tool_result = (tool_result or "") + "\n" + "\n".join(all_outputs)
            
            if has_end:
                response = text_before_end
                break
            
            follow_up_prompt = f"""使用者要求：{user_input}

你剛剛做了：
{"\n".join(all_outputs)}

還需要完成未做完的步驟。接下來要做什麼？只輸出工具標籤，完成後輸出 <end/>。"""
            current_response = call_ollama(follow_up_prompt, SYSTEM_PROMPT)
        
        print(f"\n🤖 {response}\n")
        
        update_memory(user_input, response, tool_result)
        if tool_result:
            extract_key_info(user_input, response)

if __name__ == "__main__":
    main()
