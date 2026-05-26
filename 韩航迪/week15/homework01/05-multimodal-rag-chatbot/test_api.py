"""
多模态RAG聊天机器人API测试脚本
包含所有接口的测试逻辑
"""
import requests
import json
import os
import time

# API基础URL
BASE_URL = "http://localhost:8000"


def test_root():
    """测试根路径"""
    print("\n=== 测试根路径 ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    assert response.status_code == 200
    print("✓ 根路径测试通过")


def test_upload_document(file_path):
    """测试文档上传接口"""
    print(f"\n=== 测试文档上传: {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"✗ 文件不存在: {file_path}")
        return None
    
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
        response = requests.post(f"{BASE_URL}/upload/document", files=files)
    
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    if response.status_code == 200:
        print("✓ 文档上传测试通过")
        return response.json().get('file_id')
    else:
        print("✗ 文档上传测试失败")
        return None


def test_list_documents():
    """测试获取文档列表"""
    print("\n=== 测试获取文档列表 ===")
    response = requests.get(f"{BASE_URL}/documents")
    
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    documents = response.json().get('documents', [])
    print(f"文档数量: {len(documents)}")
    print("✓ 文档列表测试通过")
    return documents


def test_delete_document(file_id):
    """测试删除文档"""
    print(f"\n=== 测试删除文档 ID: {file_id} ===")
    
    if file_id is None:
        print("✗ 文件ID为空，跳过删除测试")
        return False
    
    response = requests.delete(f"{BASE_URL}/documents/{file_id}")
    
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    if response.status_code == 200:
        print("✓ 文档删除测试通过")
        return True
    else:
        print("✗ 文档删除测试失败")
        return False


def test_retrieve(query="什么是深度学习？"):
    """测试检索接口"""
    print(f"\n=== 测试检索接口: '{query}' ===")
    
    payload = {
        "query": query,
        "top_k": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/retrieve",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"检索结果数量: {result.get('total', 0)}")
    
    if result.get('results'):
        print("\n前3个检索结果:")
        for i, item in enumerate(result['results'][:3], 1):
            print(f"\n结果 {i}:")
            print(f"  文本: {item['text'][:100]}...")
            print(f"  文件名: {item['file_name']}")
            print(f"  相似度: {item['score']:.4f}")
    
    assert response.status_code == 200
    print("\n✓ 检索接口测试通过")
    return result


def test_chat(question="请介绍一下这个项目的主要内容"):
    """测试问答接口"""
    print(f"\n=== 测试问答接口: '{question}' ===")
    
    payload = {
        "question": question,
        "top_k": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    
    if response.status_code == 200:
        print(f"\n答案:\n{result.get('answer', '')}\n")
        
        sources = result.get('sources', [])
        if sources:
            print(f"\n来源 ({len(sources)}个):")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source['file_name']}")
        
        retrieval_results = result.get('retrieval_results', [])
        if retrieval_results:
            print(f"\n检索结果 ({len(retrieval_results)}个):")
            for i, item in enumerate(retrieval_results[:3], 1):
                print(f"  {i}. 相似度: {item['score']:.4f}")
                print(f"     文本: {item['text'][:100]}...")
        
        print("\n✓ 问答接口测试通过")
        return result
    else:
        print(f"✗ 问答接口测试失败: {result}")
        return None


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("多模态RAG聊天机器人API测试套件")
    print("=" * 60)
    
    # 测试1: 根路径
    try:
        test_root()
    except Exception as e:
        print(f"✗ 根路径测试异常: {e}")
    
    # 测试2: 上传文档（需要提供测试文件路径）
    test_file_path = "doc1/demo.png"  # 修改为实际的PDF文件路径
    uploaded_file_id = None
    
    if os.path.exists(test_file_path):
        try:
            uploaded_file_id = test_upload_document(test_file_path)
            # 等待一段时间让后台处理完成
            print("\n等待文档解析...")
            time.sleep(5)
        except Exception as e:
            print(f"✗ 文档上传测试异常: {e}")
    else:
        print(f"\n⚠ 测试文件不存在: {test_file_path}")
        print("跳过上传测试，请提供有效的PDF文件路径")
    
    # 测试3: 获取文档列表
    try:
        documents = test_list_documents()
    except Exception as e:
        print(f"✗ 文档列表测试异常: {e}")
        documents = []
    
    # 测试4: 检索接口
    try:
        retrieve_result = test_retrieve("人工智能")
    except Exception as e:
        print(f"✗ 检索接口测试异常: {e}")
    
    # 测试5: 问答接口
    try:
        chat_result = test_chat("请总结一下文档内容")
    except Exception as e:
        print(f"✗ 问答接口测试异常: {e}")
    
    # 测试6: 删除文档（如果上传成功）
    if uploaded_file_id:
        try:
            test_delete_document(uploaded_file_id)
        except Exception as e:
            print(f"✗ 文档删除测试异常: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
