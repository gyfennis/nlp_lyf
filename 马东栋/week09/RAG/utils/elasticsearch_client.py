#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elasticsearch客户端（使用HTTP请求）
"""

import requests
import json


class ElasticsearchClient:
    """Elasticsearch客户端"""
    
    def __init__(self, host='localhost', port=9200):
        """
        初始化Elasticsearch客户端
        
        Args:
            host: Elasticsearch主机地址
            port: Elasticsearch端口
        """
        self.base_url = f"http://{host}:{port}"
    
    def create_index(self, index_name, mappings=None):
        """
        创建索引
        
        Args:
            index_name: 索引名称
            mappings: 索引映射
            
        Returns:
            响应结果
        """
        url = f"{self.base_url}/{index_name}"
        
        # 检查索引是否存在
        response = requests.head(url)
        if response.status_code == 200:
            print(f"索引 {index_name} 已存在，删除后重新创建")
            requests.delete(url)
        
        # 创建索引
        if mappings:
            response = requests.put(url, json=mappings)
        else:
            response = requests.put(url)
        
        return response.json()
    
    def index_document(self, index_name, document, doc_id=None):
        """
        索引文档
        
        Args:
            index_name: 索引名称
            document: 文档内容
            doc_id: 文档ID
            
        Returns:
            响应结果
        """
        if doc_id:
            url = f"{self.base_url}/{index_name}/_doc/{doc_id}"
        else:
            url = f"{self.base_url}/{index_name}/_doc"
        
        response = requests.post(url, json=document)
        return response.json()
    
    def search(self, index_name, query):
        """
        搜索文档
        
        Args:
            index_name: 索引名称
            query: 搜索查询
            
        Returns:
            搜索结果
        """
        url = f"{self.base_url}/{index_name}/_search"
        response = requests.get(url, json=query)
        return response.json()
    
    def delete_index(self, index_name):
        """
        删除索引
        
        Args:
            index_name: 索引名称
            
        Returns:
            响应结果
        """
        url = f"{self.base_url}/{index_name}"
        response = requests.delete(url)
        return response.json()
    
    def get_document(self, index_name, doc_id):
        """
        获取文档
        
        Args:
            index_name: 索引名称
            doc_id: 文档ID
            
        Returns:
            文档内容
        """
        url = f"{self.base_url}/{index_name}/_doc/{doc_id}"
        response = requests.get(url)
        return response.json()
