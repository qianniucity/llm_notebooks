import os
import json
import requests

BASE_URL = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

# 使用提供的模型为给定的提示生成响应。这是一个流式端点，因此会产生一系列响应。
# 最终的响应对象将包括来自请求的统计数据和附加数据。使用回调函数覆盖
# 默认处理函数。
def generate(model_name, prompt, system=None, template=None, context=None, options=None, callback=None):
    try:
        url = f"{BASE_URL}/api/generate"
        payload = {
            "model": model_name, 
            "prompt": prompt, 
            "system": system, 
            "template": template, 
            "context": context, 
            "options": options
        }
        
        # 删除无效值
        payload = {k: v for k, v in payload.items() if v is not None}
        
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            
            # 创建一个变量来保存最终块的上下文历史记录
            final_context = None
            
            # 如无回调，则用于保存串联响应字符串的变量
            full_response = ""

            # 逐行遍历回复并显示详情
            for line in response.iter_lines():
                if line:
                    # 解析每一行（JSON 块）并提取详细信息
                    chunk = json.loads(line)
                    
                    # 如果提供了回调函数，则使用块作为参数调用回调函数
                    if callback:
                        callback(chunk)
                    else:
                        # 如果这不是最后一个块，将“response”字段值添加到 full_response 并打印它
                        if not chunk.get("done"):
                            response_piece = chunk.get("response", "")
                            full_response += response_piece
                            print(response_piece, end="", flush=True)
                    
                    # 检查是否为最后一块（done 为 true）
                    if chunk.get("done"):
                        final_context = chunk.get("context")
            
            # 返回完整的响应和最终上下文
            return full_response, final_context
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None

# # 从 Modelfile 创建模型。使用回调函数覆盖默认处理程序。
def create(model_name, model_path, callback=None):
    try:
        url = f"{BASE_URL}/api/create"
        payload = {"name": model_name, "path": model_path}
        
        # 发出 POST 请求，并将流参数设置为 True，以处理流式响应
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            # 逐行遍历回复并显示状态
            for line in response.iter_lines():
                if line:
                    # 解析每一行（JSON 块）并提取状态
                    chunk = json.loads(line)

                    if callback:
                        callback(chunk)
                    else:
                        print(f"Status: {chunk.get('status')}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# 从模型注册表中提取模型。取消的拉取将从原处恢复，并且多个
# 调用将共享相同的下载进度。使用回调函数覆盖默认处理程序。
def pull(model_name, insecure=False, callback=None):
    try:
        url = f"{BASE_URL}/api/pull"
        payload = {
            "name": model_name,
            "insecure": insecure
        }

        # 发出 POST 请求，并将流参数设置为 True，以处理流式响应
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            # 逐行遍历回复并显示详情
            for line in response.iter_lines():
                if line:
                    # 解析每一行（JSON 块）并提取详细信息
                    chunk = json.loads(line)

                    # 如果提供了回调函数，则使用块作为参数调用回调函数
                    if callback:
                        callback(chunk)
                    else:
                        # 将状态信息直接打印到控制台
                        print(chunk.get('status', ''), end='', flush=True)
                    
                    # 如果有图层数据，可能还需要打印（根据需要进行调整）
                    if 'digest' in chunk:
                        print(f" - Digest: {chunk['digest']}", end='', flush=True)
                        print(f" - Total: {chunk['total']}", end='', flush=True)
                        print(f" - Completed: {chunk['completed']}", end='\n', flush=True)
                    else:
                        print()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# 将模型推送到模型注册表。使用回调函数覆盖默认处理程序。
def push(model_name, insecure=False, callback=None):
    try:
        url = f"{BASE_URL}/api/push"
        payload = {
            "name": model_name,
            "insecure": insecure
        }

        # 发出 POST 请求，并将流参数设置为 True，以处理流式响应
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            # 逐行遍历回复并显示详情
            for line in response.iter_lines():
                if line:
                    # 解析每一行（JSON 块）并提取详细信息
                    chunk = json.loads(line)

                    # 如果提供了回调函数，则使用块作为参数调用回调函数
                    if callback:
                        callback(chunk)
                    else:
                        # 将状态信息直接打印到控制台
                        print(chunk.get('status', ''), end='', flush=True)
                    
                    # 如果有图层数据，可能还需要打印（根据需要进行调整）
                    if 'digest' in chunk:
                        print(f" - Digest: {chunk['digest']}", end='', flush=True)
                        print(f" - Total: {chunk['total']}", end='', flush=True)
                        print(f" - Completed: {chunk['completed']}", end='\n', flush=True)
                    else:
                        print()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# 列出本地可提供的型号。
def list():
    try:
        response = requests.get(f"{BASE_URL}/api/tags")
        response.raise_for_status()
        data = response.json()
        models = data.get('models', [])
        return models

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# 复制模型 从现有模型创建另一个名称的模型。
def copy(source, destination):
    try:
        # 创建 JSON 有效负载
        payload = {
            "source": source,
            "destination": destination
        }
        
        response = requests.post(f"{BASE_URL}/api/copy", json=payload)
        response.raise_for_status()
        
        # 如果请求成功，则返回一条信息，说明复制成功
        return "Copy successful"

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# 删除模型及其数据。
def delete(model_name):
    try:
        url = f"{BASE_URL}/api/delete"
        payload = {"name": model_name}
        response = requests.delete(url, json=payload)
        response.raise_for_status()
        return "Delete successful"
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# 显示有关模型的信息。
def show(model_name):
    try:
        url = f"{BASE_URL}/api/show"
        payload = {"name": model_name}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # 解析 JSON 响应并返回
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def heartbeat():
    try:
        url = f"{BASE_URL}/"
        response = requests.head(url)
        response.raise_for_status()
        return "Ollama is running"
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return "Ollama is not running"

