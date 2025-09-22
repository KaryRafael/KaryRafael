import sys
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def preprocess_text(text):
    """
    对文本进行分词预处理
    :param text: 原始文本字符串
    :return: 分词后用空格连接的字符串
    """
    if not text:
        return ""
    words = jieba.lcut(text)
    return " ".join(words)

def calculate_similarity(original_text, copied_text):
    """
    计算两篇文本的余弦相似度（重复率）
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
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original_processed, copied_processed])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score

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

def main():
    """主函数：通过命令行参数获取路径并执行查重流程"""
    # 检查命令行参数数量
    if len(sys.argv) != 4:
        print("参数错误！正确用法：")
        print("python main.py [原文文件绝对路径] [抄袭版论文文件绝对路径] [输出答案文件绝对路径]")
    
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
    
    # 计算相似度并写入结果
    similarity = calculate_similarity(original_text, copied_text)
    write_result(output_path, similarity)

if __name__ == "__main__":
    main()
