# 智能语言评估与反馈系统核心代码
import os
import openai
import json
import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 配置OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChineseLanguageAssessment:
    """智能语言评估与反馈系统"""
    
    def __init__(self):
        self.system_prompt = """
        你是一位专业的中文语言评估专家，精通语言评估、错误分析和教学反馈。
        请根据学习者的语言样本，提供客观、全面的评估和有针对性的反馈。
        评估应关注语法准确性、词汇使用、语用得体性和文化理解等多个维度。
        """
        # 加载HSK词汇表和语法点
        self.hsk_vocab = self._load_hsk_vocab()
        self.hsk_grammar = self._load_hsk_grammar()
        
    def _load_hsk_vocab(self):
        """加载HSK词汇库（示例）"""
        # 实际应用中应从文件加载完整HSK词汇表
        hsk_vocab = {
            1: ["我", "你", "好", "是", "不", "的", "在", "有", "这", "那"],
            2: ["和", "中国", "学生", "老师", "朋友", "喜欢", "去", "看", "听", "说"],
            3: ["觉得", "认为", "因为", "所以", "但是", "可以", "应该", "需要", "如果", "已经"],
            4: ["经验", "关系", "情况", "影响", "表示", "参加", "一般", "比较", "提高", "解决"],
            5: ["政府", "经济", "环境", "文化", "政策", "措施", "态度", "观点", "分析", "建议"],
            6: ["综合", "促进", "实施", "投资", "垄断", "概念", "理论", "批评", "思想", "讽刺"]
        }
        # 构建词汇级别映射
        vocab_level = {}
        for level, words in hsk_vocab.items():
            for word in words:
                vocab_level[word] = level
        return vocab_level
    
    def _load_hsk_grammar(self):
        """加载HSK语法点（示例）"""
        # 实际应用中应从文件加载完整HSK语法点列表
        hsk_grammar = {
            1: ["是", "吗", "的", "了", "不", "也"],
            2: ["的", "了", "过", "在", "有点儿", "一点儿"],
            3: ["因为...所以...", "虽然...但是...", "比", "得", "着", "过"],
            4: ["把", "被", "是...的", "再", "又", "就"],
            5: ["不但...而且...", "既...又...", "无论...都...", "只有...才...", "即使...也..."],
            6: ["与其...不如...", "宁可...也...", "不是...而是...", "尽管...还是..."]
        }
        return hsk_grammar
    
    def assess_writing(self, text, target_level=None):
        """评估写作样本"""
        # 分词
        words = list(jieba.cut(text))
        
        # 词汇分析
        vocab_analysis = self._analyze_vocabulary(words)
        
        # 语法分析
        grammar_analysis = self._analyze_grammar(text)
        
        # 内容和结构分析
        structure_analysis = self._analyze_structure(text)
        
        # 利用GPT进行整体评估和反馈
        assessment_prompt = f"""
        请对以下HSK{target_level if target_level else ''}级别的中文写作进行评估和反馈。
        
        写作文本：
        {text}
        
        词汇分析：
        {json.dumps(vocab_analysis, ensure_ascii=False)}
        
        语法分析：
        {json.dumps(grammar_analysis, ensure_ascii=False)}
        
        结构分析：
        {json.dumps(structure_analysis, ensure_ascii=False)}
        
        请提供以下评估和反馈：
        1. 整体评分（满分100分）及各维度评分（词汇、语法、内容、结构、语用）
        2. 优点分析（具体指出2-3个做得好的方面）
        3. 问题分析（具体指出2-3个需要改进的问题，并引用原文中的实例）
        4. 改进建议（针对每个问题提供具体的改进方法和练习建议）
        5. 适合学习者水平的下一步学习目标
        
        请使用鼓励性语言，先肯定优点，再指出问题，保持积极、建设性的反馈风格。
        """
        
        gpt_assessment = self._call_gpt(assessment_prompt)
        
        # 整合结果
        result = {
            "text": text,
            "vocab_analysis": vocab_analysis,
            "grammar_analysis": grammar_analysis,
            "structure_analysis": structure_analysis,
            "gpt_assessment": gpt_assessment
        }
        
        return result
    
    def assess_speaking(self, transcript, audio_features=None, target_level=None):
        """评估口语样本"""
        # 如果有音频特征数据，分析发音、语调和流利度
        pronunciation_analysis = {}
        if audio_features:
            pronunciation_analysis = self._analyze_pronunciation(audio_features)
        
        # 分词并分析词汇
        words = list(jieba.cut(transcript))
        vocab_analysis = self._analyze_vocabulary(words)
        
        # 语法分析
        grammar_analysis = self._analyze_grammar(transcript)
        
        # 交际能力分析
        communication_analysis = self._analyze_communication(transcript)
        
        # 利用GPT进行整体评估和反馈
        assessment_prompt = f"""
        请对以下HSK{target_level if target_level else ''}级别的中文口语进行评估和反馈。
        
        口语文本：
        {transcript}
        
        发音分析：
        {json.dumps(pronunciation_analysis, ensure_ascii=False)}
        
        词汇分析：
        {json.dumps(vocab_analysis, ensure_ascii=False)}
        
        语法分析：
        {json.dumps(grammar_analysis, ensure_ascii=False)}
        
        交际能力分析：
        {json.dumps(communication_analysis, ensure_ascii=False)}
        
        请提供以下评估和反馈：
        1. 整体评分（满分100分）及各维度评分（发音、流利度、词汇、语法、交际能力）
        2. 优点分析（具体指出2-3个做得好的方面）
        3. 问题分析（具体指出2-3个需要改进的问题，并引用原文中的实例）
        4. 改进建议（针对每个问题提供具体的改进方法和练习建议）
        5. 适合学习者水平的下一步学习目标
        
        请使用鼓励性语言，先肯定优点，再指出问题，保持积极、建设性的反馈风格。
        """
        
        gpt_assessment = self._call_gpt(assessment_prompt)
        
        # 整合结果
        result = {
            "transcript": transcript,
            "pronunciation_analysis": pronunciation_analysis,
            "vocab_analysis": vocab_analysis,
            "grammar_analysis": grammar_analysis,
            "communication_analysis": communication_analysis,
            "gpt_assessment": gpt_assessment
        }
        
        return result
    
    def generate_personalized_feedback(self, assessment_result, student_info=None):
        """生成个性化反馈"""
        # 根据学生信息和评估结果生成个性化反馈
        feedback_prompt = f"""
        请根据以下评估结果和学生信息，生成个性化的学习反馈和建议。
        
        学生信息：
        {json.dumps(student_info, ensure_ascii=False) if student_info else "无特定信息"}
        
        评估结果：
        {json.dumps(assessment_result, ensure_ascii=False, indent=2)}
        
        请提供以下内容：
        1. 个性化学习反馈（考虑学生的学习风格、母语背景和学习目标）
        2. 针对性的改进建议（具体、可操作的学习方法和资源推荐）
        3. 为期两周的学习计划建议（包括重点词汇、语法、练习活动等）
        4. 激励性的总结语（强调进步和学习潜力）
        
        请确保反馈友好、鼓励性且实用，避免过于技术性的语言。
        """
        
        personalized_feedback = self._call_gpt(feedback_prompt)
        return personalized_feedback
    
    def generate_practice_exercises(self, assessment_result, quantity=5):
        """基于评估结果生成练习题"""
        # 提取需要加强的词汇和语法点
        weak_areas = self._identify_weak_areas(assessment_result)
        
        # 生成针对性练习
        exercises_prompt = f"""
        请根据以下学习者的薄弱环节，生成针对性的练习题。
        
        薄弱环节：
        {json.dumps(weak_areas, ensure_ascii=False, indent=2)}
        
        请生成以下类型的练习，每种类型{quantity}题：
        1. 词汇练习（针对需要加强的词汇）
        2. 语法练习（针对需要加强的语法点）
        3. 语用练习（针对需要加强的表达能力）
        
        每道题请提供：
        - 题目内容
        - 参考答案
        - 简要解析
        
        练习难度应稍高于学习者当前水平，但不要过难，以确保学习者能够通过努力完成。
        """
        
        practice_exercises = self._call_gpt(exercises_prompt)
        return practice_exercises
    
    def _analyze_vocabulary(self, words):
        """分析词汇水平和使用情况"""
        # 词汇总数
        total_words = len(words)
        
        # 去除重复词后的词汇量
        unique_words = set(words)
        unique_count = len(unique_words)
        
        # 词汇多样性指数（Type-Token Ratio）
        ttr = unique_count / total_words if total_words > 0 else 0
        
        # 词汇HSK级别分布
        level_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, "超出HSK": 0}
        for word in unique_words:
            if word in self.hsk_vocab:
                level = self.hsk_vocab[word]
                level_distribution[level] += 1
            else:
               level_distribution["超出HSK"] += 1
       
       # 估计整体词汇水平
       weighted_sum = 0
       for level, count in level_distribution.items():
           if level != "超出HSK":
               weighted_sum += level * count
       
       avg_level = weighted_sum / unique_count if unique_count > 0 else 0
       estimated_level = round(avg_level)
       
       return {
           "total_words": total_words,
           "unique_words": unique_count,
           "type_token_ratio": ttr,
           "level_distribution": level_distribution,
           "estimated_vocabulary_level": estimated_level
       }
   
   def _analyze_grammar(self, text):
       """分析语法使用情况"""
       # 基本语法错误检测（简化版）
       basic_errors = self._detect_basic_grammar_errors(text)
       
       # 语法结构复杂度分析
       sentence_count = len(re.split(r'[。！？\.!?]', text))
       avg_sentence_length = len(text) / sentence_count if sentence_count > 0 else 0
       
       # 检测语法点使用情况
       grammar_usage = {}
       for level, patterns in self.hsk_grammar.items():
           for pattern in patterns:
               if pattern in text:
                   if level not in grammar_usage:
                       grammar_usage[level] = []
                   grammar_usage[level].append(pattern)
       
       # 估算语法水平
       highest_level_used = max(grammar_usage.keys()) if grammar_usage else 1
       
       return {
           "basic_errors": basic_errors,
           "sentence_count": sentence_count,
           "avg_sentence_length": avg_sentence_length,
           "grammar_usage": grammar_usage,
           "estimated_grammar_level": highest_level_used
       }
   
   def _detect_basic_grammar_errors(self, text):
       """检测基本语法错误（简化版）"""
       # 此处应实现更复杂的语法错误检测
       # 当前仅做简单示例
       errors = []
       
       # 检测"的地得"使用错误（简化版）
       de_errors = re.findall(r'[形容词|动词]\s*的\s*[动词]', text)
       if de_errors:
           errors.append({"type": "的地得用法错误", "examples": de_errors})
       
       # 检测量词使用错误（简化版）
       measure_errors = re.findall(r'[一二三四五六七八九十]\s*[名词]', text)
       if measure_errors:
           errors.append({"type": "量词缺失", "examples": measure_errors})
       
       return errors
   
   def _analyze_structure(self, text):
       """分析文本结构"""
       # 段落分析
       paragraphs = text.split('\n\n')
       paragraph_count = len(paragraphs)
       
       # 连接词使用
       connectives = ["因为", "所以", "但是", "而且", "如果", "虽然", "不过", "然后", "首先", "其次", "最后", "总之"]
       connective_usage = {}
       for conn in connectives:
           count = text.count(conn)
           if count > 0:
               connective_usage[conn] = count
       
       # 主题一致性（简化版）
       # 理想情况下应使用更复杂的主题建模或语义分析
       paragraphs_tfidf = TfidfVectorizer().fit_transform([p for p in paragraphs if p.strip()])
       coherence_scores = []
       
       if len(paragraphs) > 1 and paragraphs_tfidf.shape[0] > 1:
           similarities = cosine_similarity(paragraphs_tfidf)
           # 计算每个段落与其他段落的平均相似度
           for i in range(similarities.shape[0]):
               other_indices = [j for j in range(similarities.shape[0]) if j != i]
               if other_indices:
                   avg_sim = sum(similarities[i, j] for j in other_indices) / len(other_indices)
                   coherence_scores.append(avg_sim)
       
       avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
       
       return {
           "paragraph_count": paragraph_count,
           "connective_usage": connective_usage,
           "coherence_score": avg_coherence
       }
   
   def _analyze_pronunciation(self, audio_features):
       """分析发音特征（示例）"""
       # 实际应用中需要接入语音识别和分析API
       # 这里仅作为示例
       return {
           "tone_accuracy": audio_features.get("tone_accuracy", 0.8),
           "pronunciation_accuracy": audio_features.get("pronunciation_accuracy", 0.75),
           "fluency": audio_features.get("fluency", 0.7),
           "rhythm": audio_features.get("rhythm", 0.65),
           "problematic_sounds": audio_features.get("problematic_sounds", ["zh", "ch", "sh", "r"])
       }
   
   def _analyze_communication(self, transcript):
       """分析交际能力（示例）"""
       # 语用功能分析
       functions = {
           "greeting": ["你好", "早上好", "晚上好", "嗨"],
           "farewell": ["再见", "拜拜", "回头见"],
           "thanks": ["谢谢", "感谢", "多谢"],
           "apology": ["对不起", "抱歉", "不好意思"],
           "request": ["可以", "能", "请", "麻烦"],
           "opinion": ["我认为", "我觉得", "我想", "我相信"]
       }
       
       function_usage = {}
       for func, markers in functions.items():
           for marker in markers:
               if marker in transcript:
                   if func not in function_usage:
                       function_usage[func] = 0
                   function_usage[func] += transcript.count(marker)
       
       # 话轮转换词分析
       turn_taking = ["那么", "所以", "接下来", "另外", "还有", "第一", "第二"]
       turn_taking_count = sum(transcript.count(word) for word in turn_taking)
       
       return {
           "function_usage": function_usage,
           "turn_taking_count": turn_taking_count,
           "estimated_communication_level": self._estimate_communication_level(function_usage)
       }
   
   def _estimate_communication_level(self, function_usage):
       """估计交际能力水平（简化版）"""
       # 基于语用功能多样性估计交际能力
       function_diversity = len(function_usage)
       function_total = sum(function_usage.values())
       
       if function_diversity <= 1:
           return 1
       elif function_diversity <= 2:
           return 2
       elif function_diversity <= 3:
           return 3
       elif function_diversity <= 4:
           return 4
       elif function_diversity <= 5:
           return 5
       else:
           return 6
   
   def _identify_weak_areas(self, assessment_result):
       """识别需要加强的薄弱环节"""
       weak_areas = {
           "vocabulary": [],
           "grammar": [],
           "expression": []
       }
       
       # 提取GPT评估中的问题分析部分
       gpt_assessment = assessment_result.get("gpt_assessment", "")
       
       # 简单使用正则表达式提取问题部分（实际应用中可能需要更复杂的文本分析）
       problem_section = re.search(r"问题分析[：:](.*?)改进建议", gpt_assessment, re.DOTALL)
       
       if problem_section:
           problem_text = problem_section.group(1).strip()
           
           # 根据问题文本识别薄弱环节
           if "词汇" in problem_text or "单词" in problem_text:
               # 提取需要加强的词汇
               vocab_matches = re.findall(r"「([^」]+)」|"([^"]+)"", problem_text)
               for match in vocab_matches:
                   for group in match:
                       if group and len(group) <= 4:  # 假设词汇长度不超过4个字符
                           weak_areas["vocabulary"].append(group)
           
           if "语法" in problem_text or "句子" in problem_text:
               # 提取需要加强的语法点
               for level, patterns in self.hsk_grammar.items():
                   for pattern in patterns:
                       if pattern in problem_text:
                           weak_areas["grammar"].append(pattern)
           
           if "表达" in problem_text or "交际" in problem_text:
               # 提取需要加强的表达能力
               expression_types = ["请求", "建议", "比较", "因果", "条件", "假设"]
               for expr_type in expression_types:
                   if expr_type in problem_text:
                       weak_areas["expression"].append(expr_type)
       
       return weak_areas
   
   def _call_gpt(self, prompt):
       """调用GPT模型生成评估和反馈"""
       try:
           response = openai.ChatCompletion.create(
               model="gpt-4",  # 或者使用其他适合的模型
               messages=[
                   {"role": "system", "content": self.system_prompt},
                   {"role": "user", "content": prompt}
               ],
               temperature=0.3,  # 低温度以确保评估的一致性
               max_tokens=2000
           )
           return response.choices[0].message.content
       except Exception as e:
           return f"评估生成出错: {str(e)}"
   
   def visualize_assessment(self, assessment_result, output_file=None):
       """可视化评估结果"""
       # 创建评估结果的可视化报告
       plt.figure(figsize=(15, 12))
       
       # 1. 词汇分析可视化
       plt.subplot(2, 2, 1)
       vocab_analysis = assessment_result.get("vocab_analysis", {})
       level_dist = vocab_analysis.get("level_distribution", {})
       
       levels = []
       counts = []
       for level, count in level_dist.items():
           if level != "超出HSK":
               levels.append(f"HSK{level}")
               counts.append(count)
       
       if "超出HSK" in level_dist:
           levels.append("超出HSK")
           counts.append(level_dist["超出HSK"])
       
       plt.bar(levels, counts, color='skyblue')
       plt.title('词汇HSK级别分布')
       plt.xlabel('HSK级别')
       plt.ylabel('词汇数量')
       
       # 2. 语法分析可视化
       plt.subplot(2, 2, 2)
       grammar_analysis = assessment_result.get("grammar_analysis", {})
       grammar_usage = grammar_analysis.get("grammar_usage", {})
       
       g_levels = []
       g_counts = []
       for level, patterns in grammar_usage.items():
           g_levels.append(f"HSK{level}")
           g_counts.append(len(patterns))
       
       plt.bar(g_levels, g_counts, color='lightgreen')
       plt.title('语法点使用分布')
       plt.xlabel('HSK级别')
       plt.ylabel('语法点数量')
       
       # 3. 整体评分雷达图
       plt.subplot(2, 2, 3)
       
       # 从GPT评估中提取各维度评分（实际应用中需要更精确的提取方法）
       gpt_assessment = assessment_result.get("gpt_assessment", "")
       
       # 简单使用正则表达式提取评分（实际应用中可能需要更复杂的文本分析）
       scores = {
           "词汇": 0,
           "语法": 0,
           "内容": 0,
           "结构": 0,
           "语用": 0
       }
       
       for dimension in scores.keys():
           score_match = re.search(f"{dimension}[：:]\s*(\d+)", gpt_assessment)
           if score_match:
               scores[dimension] = int(score_match.group(1))
       
       # 创建雷达图
       categories = list(scores.keys())
       values = list(scores.values())
       
       # 闭合多边形
       categories.append(categories[0])
       values.append(values[0])
       
       # 计算角度
       angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
       angles += angles[:1]  # 闭合
       
       # 绘制雷达图
       ax = plt.subplot(2, 2, 3, polar=True)
       ax.plot(angles, values, linewidth=2, linestyle='solid')
       ax.fill(angles, values, alpha=0.25)
       ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
       ax.set_ylim(0, 100)
       plt.title('能力维度评分')
       
       # 4. 文本分析摘要
       plt.subplot(2, 2, 4)
       plt.axis('off')
       
       text_summary = f"""
       评估摘要:
       
       文本长度: {len(assessment_result.get('text', ''))} 字符
       词汇总数: {vocab_analysis.get('total_words', 0)} 词
       词汇多样性: {vocab_analysis.get('type_token_ratio', 0):.2f}
       估计词汇水平: HSK{vocab_analysis.get('estimated_vocabulary_level', 0)}
       
       句子数量: {grammar_analysis.get('sentence_count', 0)}
       平均句长: {grammar_analysis.get('avg_sentence_length', 0):.2f} 字符
       估计语法水平: HSK{grammar_analysis.get('estimated_grammar_level', 0)}
       
       优点摘要:
       {self._extract_section(gpt_assessment, "优点分析", "问题分析")}
       
       问题摘要:
       {self._extract_section(gpt_assessment, "问题分析", "改进建议")}
       """
       
       plt.text(0, 0.5, text_summary, fontsize=10, verticalalignment='center')
       
       plt.tight_layout()
       
       # 保存或显示
       if output_file:
           plt.savefig(output_file)
           return output_file
       else:
           plt.show()
           return None
   
   def _extract_section(self, text, start_marker, end_marker):
       """从文本中提取特定部分"""
       pattern = f"{start_marker}[：:](.*?){end_marker}"
       match = re.search(pattern, text, re.DOTALL)
       if match:
           return match.group(1).strip()
       return "无内容"


def main():
   """主函数示例"""
   # 初始化评估系统
   assessor = ChineseLanguageAssessment()
   
   # 示例写作文本
   sample_writing = """
   我叫李明，今年二十岁。我是中国留学生，现在在美国读大学。我的专业是计算机科学。
   我很喜欢我的专业，因为计算机科学很有意思，而且将来工作机会多。
   我的大学在加利福尼亚，那里的天气很好，不太冷也不太热。
   我的爱好是打篮球和看电影。每个周末，我和朋友一起打篮球，然后一起吃饭。
   我学英语已经十年了，但是说英语的时候还有点儿紧张。我希望明年英语说得更好。
   """
   
   # 评估写作样本
   writing_assessment = assessor.assess_writing(sample_writing, target_level=4)
   
   # 生成个性化反馈
   student_info = {
       "mother_tongue": "英语",
       "learning_time": "2年",
       "learning_goal": "能够在中国工作",
       "learning_style": "视觉学习者"
   }
   feedback = assessor.generate_personalized_feedback(writing_assessment, student_info)
   
   # 生成练习题
   exercises = assessor.generate_practice_exercises(writing_assessment)
   
   # 可视化评估结果
   visualization_file = "assessment_visualization.png"
   assessor.visualize_assessment(writing_assessment, visualization_file)
   
   # 输出结果
   print("=== 评估结果 ===")
   print(writing_assessment["gpt_assessment"])
   print("\n=== 个性化反馈 ===")
   print(feedback)
   print("\n=== 练习题 ===")
   print(exercises)
   print(f"\n评估可视化已保存至: {visualization_file}")


if __name__ == "__main__":
   main()