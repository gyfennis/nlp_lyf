"""
多模态RAG API 使用示例
展示如何使用Python调用API接口
"""
import requests
import json

# API基础URL
BASE_URL = "http://localhost:8000"


def example_upload_document():
    """示例1: 上传文档"""
    print("=" * 60)
    print("示例1: 上传文档")
    print("=" * 60)
    
    # 准备要上传的文件
    file_path = "test.pdf"  # 替换为你的PDF文件路径
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'application/pdf')}
            response = requests.post(
                f"{BASE_URL}/upload/document",
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 上传成功!")
            print(f"  文件ID: {result['file_id']}")
            print(f"  文件名: {result['filename']}")
            print(f"  保存路径: {result['filepath']}")
            return result['file_id']
        else:
            print(f"✗ 上传失败: {response.text}")
            return None
    except FileNotFoundError:
        print(f"✗ 文件不存在: {file_path}")
        return None
    except Exception as e:
        print(f"✗ 错误: {e}")
        return None


def example_list_documents():
    """示例2: 获取文档列表"""
    print("\n" + "=" * 60)
    print("示例2: 获取文档列表")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/documents")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 共找到 {result['total']} 个文档\n")
        
        for doc in result['documents']:
            print(f"  ID: {doc['id']}")
            print(f"  文件名: {doc['filename']}")
            print(f"  状态: {doc['filestate']}")
            print(f"  路径: {doc['filepath']}")
            print()
    else:
        print(f"✗ 获取失败: {response.text}")


def example_delete_document(file_id):
    """示例3: 删除文档"""
    print("=" * 60)
    print("示例3: 删除文档")
    print("=" * 60)
    
    response = requests.delete(f"{BASE_URL}/documents/{file_id}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result['message']}")
    else:
        print(f"✗ 删除失败: {response.text}")


def example_retrieve():
    """示例4: 多模态检索"""
    print("\n" + "=" * 60)
    print("示例4: 多模态检索")
    print("=" * 60)
    
    query = "什么是深度学习？"
    print(f"查询: {query}\n")
    
    payload = {
        "query": query,
        "top_k": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/retrieve",
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 找到 {result['total']} 个相关结果\n")
        
        for i, item in enumerate(result['results'], 1):
            print(f"结果 {i}:")
            print(f"  文本: {item['text'][:200]}...")
            print(f"  来源: {item['file_name']}")
            print(f"  相似度: {item['score']:.4f}")
            print()
    else:
        print(f"✗ 检索失败: {response.text}")


def example_chat():
    """示例5: 多模态问答"""
    print("=" * 60)
    print("示例5: 多模态问答")
    print("=" * 60)
    
    question = "请介绍一下深度学习的主要应用"
    print(f"问题: {question}\n")
    
    payload = {
        "question": question,
        "top_k": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print("答案:")
        print("-" * 60)
        print(result['answer'])
        print("-" * 60)
        
        if result['sources']:
            print(f"\n参考来源 ({len(result['sources'])}个):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['file_name']}")
        
        if result['retrieval_results']:
            print(f"\n检索到的相关内容 ({len(result['retrieval_results'])}条):")
            for i, item in enumerate(result['retrieval_results'][:3], 1):
                print(f"  {i}. [相似度: {item['score']:.4f}] {item['text'][:100]}...")
    else:
        print(f"✗ 问答失败: {response.text}")


def example_complete_workflow():
    """示例6: 完整工作流程"""
    print("\n" + "=" * 60)
    print("示例6: 完整工作流程演示")
    print("=" * 60)
    
    # 步骤1: 查看当前文档列表
    print("\n[步骤1] 查看当前文档列表")
    example_list_documents()
    
    # 步骤2: 上传新文档（如果有测试文件）
    print("\n[步骤2] 上传新文档")
    file_id = example_upload_document()
    
    if file_id:
        # 等待一段时间让后台处理
        print("\n等待文档解析...")
        import time
        time.sleep(5)
        
        # 步骤3: 检索相关内容
        print("\n[步骤3] 检索相关内容")
        example_retrieve()
        
        # 步骤4: 进行问答
        print("\n[步骤4] 进行智能问答")
        example_chat()
        
        # 步骤5: 清理（可选）
        print("\n[步骤5] 清理测试数据")
        confirm = input("是否删除刚才上传的文档？(y/n): ")
        if confirm.lower() == 'y':
            example_delete_document(file_id)


def main():
    """主函数 - 运行所有示例"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "多模态RAG API 使用示例" + " " * 24 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    print("请选择要运行的示例:")
    print("1. 上传文档")
    print("2. 获取文档列表")
    print("3. 删除文档")
    print("4. 多模态检索")
    print("5. 多模态问答")
    print("6. 完整工作流程")
    print("0. 退出")
    print()
    
    while True:
        choice = input("请输入选项 (0-6): ").strip()
        
        if choice == '0':
            print("再见！")
            break
        elif choice == '1':
            example_upload_document()
        elif choice == '2':
            example_list_documents()
        elif choice == '3':
            file_id = input("请输入要删除的文件ID: ").strip()
            if file_id.isdigit():
                example_delete_document(int(file_id))
            else:
                print("无效的ID")
        elif choice == '4':
            example_retrieve()
        elif choice == '5':
            example_chat()
        elif choice == '6':
            example_complete_workflow()
        else:
            print("无效的选项，请重新选择")
        
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    # 检查服务是否运行
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        if response.status_code != 200:
            print("⚠ 警告: 服务可能未正常运行")
            print("请先启动服务: python main.py")
            print()
    except requests.exceptions.ConnectionError:
        print("✗ 错误: 无法连接到服务")
        print("请先启动服务: python main.py")
        exit(1)
    
    # 运行交互式示例
    main()
