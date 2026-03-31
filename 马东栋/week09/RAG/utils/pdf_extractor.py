#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF文本提取器
"""

import os
import fitz


class PDFExtractor:
    """PDF文本提取器"""
    
    def extract_text(self, pdf_path):
        """
        从PDF文件中提取文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本内容
        """
        try:
            doc = fitz.open(pdf_path)
            text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text.append(page_text)
            
            doc.close()
            return "\n".join(text)
        except Exception as e:
            print(f"提取PDF文本时出错：{e}")
            return ""
    
    def extract_text_from_directory(self, directory):
        """
        从目录中的所有PDF文件提取文本
        
        Args:
            directory: 包含PDF文件的目录
            
        Returns:
            字典，键为文件名，值为提取的文本
        """
        pdf_texts = {}
        
        for filename in os.listdir(directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory, filename)
                text = self.extract_text(pdf_path)
                if text:
                    pdf_texts[filename] = text
                    print(f"已提取：{filename}")
        
        return pdf_texts
