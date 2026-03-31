#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG工程主入口
"""

import os
import sys
from rag.rag_qa import RAGQA
from retrieval.retriever import Retriever
from utils.pdf_extractor import PDFExtractor
from utils.elasticsearch_client import ElasticsearchClient


def main():
    """主函数"""
    # 初始化PDF提取器
    pdf_extractor = PDFExtractor()
    
    # 初始化Elasticsearch客户端
    es_client = ElasticsearchClient()
    
    # 初始化检索器
    retriever = Retriever(es_client)
    
    # 初始化RAG问答系统
    rag_qa = RAGQA(retriever)
    
    # 构建知识库
    print("开始构建知识库...")
    pdf_dir = "e:\\BaiduNetdiskDownload\\nlp\\week09\\week09\\RAG\\检索文本"
    rag_qa.build_knowledge_base(pdf_dir)
    print("知识库构建完成！")
    
    # 交互式问答
    print("\nRAG问答系统已启动，输入'quit'退出")
    while True:
        query = input("\n请输入问题：")
        if query.lower() == 'quit':
            break
        
        # 意图识别
        intent = rag_qa.recognize_intent(query)
        print(f"意图识别结果：{intent}")
        
        # 生成回答
        answer = rag_qa.generate_answer(query, intent)
        print(f"\n回答：{answer}")


if __name__ == "__main__":
    main()
