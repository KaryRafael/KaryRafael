import sys
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from difflib import SequenceMatcher
import os

# ==================== 性能优化模块 ====================
# 全局向量化器实例，避免重复初始化
_vectorizer = None

def get_vectorizer():
    """
    获取全局TF-IDF向量化器实例（单例模式）
    优化点：避免每次调用时重复初始化向量化器
    """
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer()
    return _vectorizer

# 分词结果缓存（LRU缓存，最大缓存1000个结果）
@lru_cache(maxsize=1000)
def cached_preprocess_text(text):
    """
    带缓存的分词处理函数
    优化点：对相同文本内容直接返回缓存结果，避免重复分词
    """
    if not text:
        return ""
    words = jieba.lcut(text)
    return " ".join(words)

# ==================== 文件处理模块 ====================
def read_file(file_path):
    """
    读取指定路径的文件内容
    :param file_path: 文件的绝对路径
    :return: 文件内容字符串（去除首尾空白），读取失败则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        return content
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return None
    except UnicodeDecodeError:
        print(f"错误：文件 {file_path} 编码不是UTF-8，无法读取")
        return None
    except Exception as e:
        print(f"读取文件 {file_path} 失败：{str(e)}")
        return None

# ==================== 文本预处理模块 ====================
def preprocess_text(text):
    """
    对文本进行分词预处理（调用缓存版本）
    :param text: 原始文本字符串
    :return: 分词后用空格连接的字符串
    """
    return cached_preprocess_text(text)

# ==================== 相似度计算模块 ====================
def calculate_similarity(original_text, copied_text):
    """
    计算两篇文本的余弦相似度（重复率）
    优化点：使用全局向量化器，避免重复初始化
    :param original_text: 原文文本
    :param copied_text: 抄袭版文本
    :return: 相似度值（0-1之间）
    """
    original_processed = preprocess_text(original_text)
    copied_processed = preprocess_text(copied_text)
    
    # 处理空文本情况
    if not original_processed and not copied_processed:
        return 1.0
    if not original_processed or not copied_processed:
        return 0.0
    
    # 处理文本过短的情况（只有一个词或没有有效词汇）
    original_words = original_processed.split()
    copied_words = copied_processed.split()
    
    # 如果分词后没有有效词汇，直接比较原始文本
    if not original_words and not copied_words:
        # 直接比较原始文本是否完全相同
        return 1.0 if original_text == copied_text else 0.0
    elif not original_words or not copied_words:
        return 0.0
    
    # 如果词汇太少，使用字符级比较
    if len(original_words) == 1 and len(copied_words) == 1:
        # 单字或单词比较
        if original_words[0] == copied_words[0]:
            return 1.0
        else:
            # 计算字符相似度
            from difflib import SequenceMatcher
            return SequenceMatcher(None, original_text, copied_text).ratio()
    
    try:
        # 使用全局向量化器实例
        vectorizer = get_vectorizer()
        tfidf_matrix = vectorizer.fit_transform([original_processed, copied_processed])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity_score
    except ValueError as e:
        # 处理向量化器错误（如词汇表为空）
        if "empty vocabulary" in str(e):
            # 使用简单的文本相似度作为备选方案
            from difflib import SequenceMatcher
            return SequenceMatcher(None, original_text, copied_text).ratio()
        else:
            # 其他错误重新抛出
            raise e

def calculate_similarity_batch(text_pairs):
    """
    批量计算相似度（性能优化版本）
    优化点：一次性处理多组文本，减少重复计算
    :param text_pairs: 文本对列表，每个元素为(原文, 抄袭版)
    :return: 相似度列表
    """
    if not text_pairs:
        return []
    
    # 准备所有文本进行批量处理
    all_texts = []
    for original, copied in text_pairs:
        all_texts.extend([preprocess_text(original), preprocess_text(copied)])
    
    # 批量向量化
    vectorizer = get_vectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # 批量计算相似度
    similarities = []
    for i in range(0, len(all_texts), 2):
        if i + 1 < len(all_texts):
            similarity = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0][0]
            similarities.append(similarity)
    
    return similarities

# ==================== 结果输出模块 ====================
def write_result(output_path, similarity):
    """
    将相似度结果写入指定输出文件
    :param output_path: 输出文件的绝对路径
    :param similarity: 计算得到的相似度值
    """
    try:
        result = round(similarity, 2)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(str(result))
    except Exception as e:
        print(f"写入结果到 {output_path} 失败：{str(e)}")

def write_results_batch(output_dir, results, filenames):
    """
    批量写入结果（性能优化）
    优化点：减少频繁的文件打开关闭操作
    :param output_dir: 输出目录
    :param results: 结果列表
    :param filenames: 对应的输出文件名列表
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, (result, filename) in enumerate(zip(results, filenames)):
        output_path = os.path.join(output_dir, filename)
        write_result(output_path, result)

# ==================== 主控制模块 ====================
def main():
    """主函数：通过命令行参数获取路径并执行查重流程"""
    # 检查命令行参数数量
    if len(sys.argv) != 4:
        print("参数错误！正确用法：")
        print("python main.py [原文文件绝对路径] [抄袭版论文文件绝对路径] [输出答案文件绝对路径]")
        return
    
    # 从命令行参数获取文件路径
    original_path = sys.argv[1]
    copied_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # 读取文件内容
    original_text = read_file(original_path)
    copied_text = read_file(copied_path)
    
    # 检查文件读取是否成功
    if original_text is None or copied_text is None:
        print("文件读取失败！")
        return
    
    # 计算相似度并写入结果
    similarity = calculate_similarity(original_text, copied_text)
    write_result(output_path, similarity)
    print(f"查重完成！相似度: {similarity:.2f}")

def main_batch(input_dir, output_dir):
    """
    批量处理主函数（性能优化版本）
    优化点：支持批量文件处理，提高整体效率
    :param input_dir: 输入目录，包含多个原文和抄袭版文件
    :param output_dir: 输出目录
    """
    # 实现批量文件处理逻辑
    # 这里需要根据实际文件命名规则来实现
    pass

if __name__ == "__main__":
    main()
