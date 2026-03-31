#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG问答系统
"""

import os
import json
import requests
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from utils.pdf_extractor import PDFExtractor
from openai import OpenAI


class RAGQA:
    """RAG问答系统"""
    
    def __init__(self, retriever):
        """
        初始化RAG问答系统
        
        Args:
            retriever: 检索器实例
        """
        self.retriever = retriever
        self.pdf_extractor = PDFExtractor()
        self.document_embeddings = []
        
        # 加载意图识别模型
        self.tokenizer = BertTokenizer.from_pretrained(r'E:\BaiduNetdiskDownload\nlp\models\google-bert\bert-base-chinese')
        self.intent_model = BertForSequenceClassification.from_pretrained('./results/checkpoint-96')
        
        # 标签映射
        self.label_map = {0: '机器学习', 1: 'llm', 2: '其他'}
        
        # 从环境变量获取阿里云API密钥
        self.api_key = os.environ.get('aliyunAPI_KEY', 'sk-4f0981ea000c45faa4274d3f17a8479')
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    def recognize_intent(self, query):
        """
        识别用户意图
        
        Args:
            query: 用户查询
            
        Returns:
            意图标签
        """
        inputs = self.tokenizer(
            query,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=64
        )
        
        with torch.no_grad():
            outputs = self.intent_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_idx = torch.argmax(predictions, dim=-1).item()
        
        return self.label_map.get(predicted_class_idx, '其他')
    
    def build_knowledge_base(self, pdf_dir):
        """
        构建知识库
        
        Args:
            pdf_dir: PDF文件目录
        """
        # 提取PDF文本
        pdf_texts = self.pdf_extractor.extract_text_from_directory(pdf_dir)
        
        # 准备文档
        documents = []
        for filename, text in pdf_texts.items():
            # 简单分块
            chunks = self._split_text(text, chunk_size=500, overlap=50)
            documents.extend(chunks)
        
        # 构建TFIDF索引
        self.retriever.build_tfidf_index(documents)
        
        # 生成BGE嵌入
        self.document_embeddings = []
        for doc in documents:
            embedding = self.retriever.get_bge_embedding(doc)
            self.document_embeddings.append(embedding)
        
        # 构建Elasticsearch索引
        self._build_elasticsearch_index(documents)
    
    def _split_text(self, text, chunk_size=500, overlap=50):
        """
        文本分块
        
        Args:
            text: 文本
            chunk_size: 块大小
            overlap: 重叠大小
            
        Returns:
            分块后的文本列表
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
    
    def _build_elasticsearch_index(self, documents):
        """
        构建Elasticsearch索引
        
        Args:
            documents: 文档列表
        """
        # 构建机器学习知识库
        ml_mappings = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "ik_max_word"
                    }
                }
            }
        }
        
        # 构建LLM知识库
        llm_mappings = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "ik_max_word"
                    }
                }
            }
        }
        
        # 创建索引
        self.retriever.es_client.create_index('ml_knowledge_base', ml_mappings)
        self.retriever.es_client.create_index('llm_knowledge_base', llm_mappings)
        
        # 索引文档
        for i, doc in enumerate(documents):
            # 简单判断文档类型，实际应用中可能需要更复杂的分类
            if any(keyword in doc.lower() for keyword in ['机器学习', '深度学习', '神经网络', '算法']):
                index_name = 'ml_knowledge_base'
            else:
                index_name = 'llm_knowledge_base'
            
            self.retriever.es_client.index_document(index_name, {'text': doc}, str(i))
    
    def generate_answer(self, query, intent):
        """
        生成回答
        
        Args:
            query: 用户查询
            intent: 意图
            
        Returns:
            生成的回答
        """
        # 根据意图选择知识库
        if intent == '机器学习':
            index_name = 'ml_knowledge_base'
        elif intent == 'llm':
            index_name = 'llm_knowledge_base'
        else:
            return "抱歉，我只能回答关于机器学习和LLM的问题。"
        
        # 多路召回
        retrieved_docs = self.retriever.multi_retrieve(
            query, 
            index_name, 
            self.document_embeddings,
            top_k=3
        )
        
        # 构建上下文
        context = "\n".join([doc['text'] for doc in retrieved_docs])
        
        # 调用阿里云百炼大模型
        prompt = f"根据以下上下文回答问题：\n\n上下文：{context}\n\n问题：{query}\n\n回答："
        answer = self._call_aliyun_llm(prompt)
        
        return answer
    
    def _call_aliyun_llm(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model="qwen-flash",  # 模型的代号
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"调用大模型时出错：{e}")
            return "抱歉，生成回答时出错，请稍后再试。"
