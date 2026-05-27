"""
快速测试脚本 - 验证API基本功能
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def quick_test():
    """快速测试所有接口"""
    print("=" * 60)
    print("多模态RAG API 快速测试")
    print("=" * 60)
    
    # 测试1: 健康检查
    print("\n[1/4] 测试健康检查...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✓ 服务正常运行")
            print(f"  API版本: {response.json().get('version', 'unknown')}")
        else:
            print(f"✗ 服务异常: {response.status_code}")
            return
    except Exception as e:
        print(f"✗ 无法连接到服务: {e}")
        print("  请确保服务已启动: python main.py")
        return
    
    # 测试2: 获取文档列表
    print("\n[2/4] 测试获取文档列表...")
    try:
        response = requests.get(f"{BASE_URL}/documents")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 获取成功，共 {data.get('total', 0)} 个文档")
        else:
            print(f"✗ 获取失败: {response.status_code}")
    except Exception as e:
        print(f"✗ 请求失败: {e}")
    
    # 测试3: 测试检索接口
    print("\n[3/4] 测试检索接口...")
    try:
        response = requests.post(
            f"{BASE_URL}/retrieve",
            json={"query": "测试", "top_k": 3}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 检索成功，返回 {data.get('total', 0)} 个结果")
        else:
            print(f"✗ 检索失败: {response.status_code}")
            print(f"  错误信息: {response.text}")
    except Exception as e:
        print(f"✗ 请求失败: {e}")
    
    # 测试4: 测试问答接口
    print("\n[4/4] 测试问答接口...")
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "你好", "top_k": 3}
        )
        if response.status_code == 200:
            data = response.json()
            print("✓ 问答成功")
            print(f"  答案长度: {len(data.get('answer', ''))} 字符")
            print(f"  来源数量: {len(data.get('sources', []))}")
        else:
            print(f"✗ 问答失败: {response.status_code}")
            print(f"  错误信息: {response.text}")
    except Exception as e:
        print(f"✗ 请求失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n提示:")
    print("- 访问 http://localhost:8000/docs 查看完整API文档")
    print("- 运行 python test_api.py 进行详细测试")


if __name__ == "__main__":
    quick_test()
