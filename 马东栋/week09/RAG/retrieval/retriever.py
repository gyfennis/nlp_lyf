#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索器模块
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch


class Retriever:
    """检索器"""
    
    def __init__(self, es_client):
        """
        初始化检索器
        
        Args:
            es_client: Elasticsearch客户端
        """
        self.es_client = es_client
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        
        # 加载BGE模型
        bge_model_path = r"E:\BaiduNetdiskDownload\nlp\models\BAAI\bge-small-zh-v1.5"
        self.bge_tokenizer = AutoTokenizer.from_pretrained(bge_model_path)
        self.bge_model = AutoModel.from_pretrained(bge_model_path)
        self.bge_model.eval()
    
    def build_tfidf_index(self, documents):
        """
        构建TFIDF索引
        
        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
    
    def retrieve_tfidf(self, query, top_k=5):
        """
        使用TFIDF检索
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            检索结果列表
        """
        if not self.tfidf_vectorizer:
            return []
        
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'score': similarities[idx]
            })
        
        return results
    
    def retrieve_bm25(self, query, index_name, top_k=5):
        """
        使用BM25检索（通过Elasticsearch）
        
        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回前k个结果
            
        Returns:
            检索结果列表
        """
        bm25_query = {
            "size": top_k,
            "query": {
                "match": {
                    "text": query
                }
            }
        }
        
        response = self.es_client.search(index_name, bm25_query)
        results = []
        
        if 'hits' in response and 'hits' in response['hits']:
            for hit in response['hits']['hits']:
                results.append({
                    'text': hit['_source']['text'],
                    'score': hit['_score']
                })
        
        return results
    
    def get_bge_embedding(self, text):
        """
        获取文本的BGE嵌入
        
        Args:
            text: 文本
            
        Returns:
            嵌入向量
        """
        inputs = self.bge_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.bge_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding
    
    def retrieve_bge(self, query, embeddings, top_k=5):
        """
        使用BGE模型检索
        
        Args:
            query: 查询文本
            embeddings: 文档嵌入列表
            top_k: 返回前k个结果
            
        Returns:
            检索结果列表
        """
        query_embedding = self.get_bge_embedding(query)
        similarities = []
        
        for i, doc_embedding in enumerate(embeddings):
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for idx, score in similarities[:top_k]:
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'score': score
                })
        
        return results
    
    def rerank(self, query, candidates, top_k=3):
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回前k个结果
            
        Returns:
            重排序后的结果列表
        """
        # 简单实现：使用BGE嵌入计算相似度
        query_embedding = self.get_bge_embedding(query)
        rerank_results = []
        
        for candidate in candidates:
            doc_embedding = self.get_bge_embedding(candidate['text'])
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            rerank_results.append({
                'text': candidate['text'],
                'score': similarity
            })
        
        rerank_results.sort(key=lambda x: x['score'], reverse=True)
        return rerank_results[:top_k]
    
    def multi_retrieve(self, query, index_name, embeddings, top_k=5):
        """
        多路召回
        
        Args:
            query: 查询文本
            index_name: 索引名称
            embeddings: 文档嵌入列表
            top_k: 返回前k个结果
            
        Returns:
            重排序后的结果列表
        """
        # TFIDF检索
        tfidf_results = self.retrieve_tfidf(query, top_k)
        
        # BM25检索
        bm25_results = self.retrieve_bm25(query, index_name, top_k)
        
        # BGE检索
        bge_results = self.retrieve_bge(query, embeddings, top_k)
        
        # 合并结果
        all_candidates = {}
        for result in tfidf_results + bm25_results + bge_results:
            if result['text'] not in all_candidates:
                all_candidates[result['text']] = result
        
        # 重排序
        candidates_list = list(all_candidates.values())
        reranked_results = self.rerank(query, candidates_list, top_k)
        
        return reranked_results
