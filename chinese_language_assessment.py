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
        self.system_prompt = (
            "你是一位专业的中文语言评估专家，精通语言评估、错误分析和教学反馈。"
            "请根据学习者的语言样本，提供客观、全面的评估和有针对性的反馈。"
        )
        self.hsk_vocab = self._load_hsk_vocab()
        self.hsk_grammar = self._load_hsk_grammar()

    def _load_hsk_vocab(self):
        """加载HSK词汇库（示例）"""
        hsk_vocab = {
            1: ["我", "你", "好", "是", "不", "的", "在", "有", "这", "那"],
            2: ["和", "中国", "学生", "老师", "朋友", "喜欢", "去", "看", "听", "说"],
            3: ["觉得", "认为", "因为", "所以", "但是", "可以", "应该", "需要", "如果", "已经"],
            4: ["经验", "关系", "情况", "影响", "表示", "参加", "一般", "比较", "提高", "解决"],
            5: ["政府", "经济", "环境", "文化", "政策", "措施", "态度", "观点", "分析", "建议"],
            6: ["综合", "促进", "实施", "投资", "垄断", "概念", "理论", "批评", "思想", "讽刺"],
        }
        vocab_level = {}
        for level, words in hsk_vocab.items():
            for word in words:
                vocab_level[word] = level
        return vocab_level

    def _load_hsk_grammar(self):
        """加载HSK语法点（示例）"""
        return {
            1: ["是", "吗", "的", "了", "不", "也"],
            2: ["的", "了", "过", "在", "有点儿", "一点儿"],
            3: ["因为...所以...", "虽然...但是...", "比", "得", "着", "过"],
            4: ["把", "被", "是...的", "再", "又", "就"],
            5: ["不但...而且...", "既...又...", "无论...都...", "只有...才...", "即使...也..."],
            6: ["与其...不如...", "宁可...也...", "不是...而是...", "尽管...还是..."],
        }

    def assess_writing(self, text, target_level=None):
        """评估写作样本"""
        words = list(jieba.cut(text))
        vocab_analysis = self._analyze_vocabulary(words)
        grammar_analysis = self._analyze_grammar(text)
        structure_analysis = self._analyze_structure(text)
        level = target_level if target_level else ""
        assessment_prompt = (
            f"请对以下HSK{level}级别的中文写作进行评估和反馈。\n"
            f"写作文本：\n{text}\n"
            f"词汇分析：\n{json.dumps(vocab_analysis, ensure_ascii=False)}\n"
            f"语法分析：\n{json.dumps(grammar_analysis, ensure_ascii=False)}\n"
            f"结构分析：\n{json.dumps(structure_analysis, ensure_ascii=False)}\n"
            "请提供整体评分、优点、问题、改进建议与下一步学习目标。"
        )
        gpt_assessment = self._call_gpt(assessment_prompt)
        return {
            "text": text,
            "vocab_analysis": vocab_analysis,
            "grammar_analysis": grammar_analysis,
            "structure_analysis": structure_analysis,
            "gpt_assessment": gpt_assessment,
        }

    def assess_speaking(self, transcript, audio_features=None, target_level=None):
        """评估口语样本"""
        pronunciation_analysis = {}
        if audio_features:
            pronunciation_analysis = self._analyze_pronunciation(audio_features)
        words = list(jieba.cut(transcript))
        vocab_analysis = self._analyze_vocabulary(words)
        grammar_analysis = self._analyze_grammar(transcript)
        communication_analysis = self._analyze_communication(transcript)
        level = target_level if target_level else ""
        assessment_prompt = (
            f"请对以下HSK{level}级别的中文口语进行评估和反馈。\n"
            f"口语文本：\n{transcript}\n"
            f"发音分析：\n{json.dumps(pronunciation_analysis, ensure_ascii=False)}\n"
            f"词汇分析：\n{json.dumps(vocab_analysis, ensure_ascii=False)}\n"
            f"语法分析：\n{json.dumps(grammar_analysis, ensure_ascii=False)}\n"
            f"交际能力分析：\n{json.dumps(communication_analysis, ensure_ascii=False)}\n"
            "请提供整体评分、优点、问题、改进建议与下一步学习目标。"
        )
        gpt_assessment = self._call_gpt(assessment_prompt)
        return {
            "transcript": transcript,
            "pronunciation_analysis": pronunciation_analysis,
            "vocab_analysis": vocab_analysis,
            "grammar_analysis": grammar_analysis,
            "communication_analysis": communication_analysis,
            "gpt_assessment": gpt_assessment,
        }

    def generate_personalized_feedback(self, assessment_result, student_info=None):
        """生成个性化反馈"""
        info = json.dumps(student_info, ensure_ascii=False) if student_info else "无特定信息"
        feedback_prompt = (
            "请根据以下评估结果和学生信息，生成个性化的学习反馈和建议。\n"
            f"学生信息：\n{info}\n"
            f"评估结果：\n{json.dumps(assessment_result, ensure_ascii=False, indent=2)}\n"
            "请提供个性化反馈、改进建议、两周学习计划与激励性总结。"
        )
        return self._call_gpt(feedback_prompt)

    def generate_practice_exercises(self, assessment_result, quantity=5):
        """基于评估结果生成练习题"""
        weak_areas = self._identify_weak_areas(assessment_result)
        exercises_prompt = (
            "请根据以下学习者的薄弱环节，生成针对性的练习题。\n"
            f"薄弱环节：\n{json.dumps(weak_areas, ensure_ascii=False, indent=2)}\n"
            f"请生成词汇、语法、语用练习各{quantity}题，并提供答案与解析。"
        )
        return self._call_gpt(exercises_prompt)

    def _analyze_vocabulary(self, words):
        """分析词汇水平和使用情况"""
        total_words = len(words)
        unique_words = set(words)
        unique_count = len(unique_words)
        ttr = unique_count / total_words if total_words > 0 else 0
        level_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, "超出HSK": 0}
        for word in unique_words:
            if word in self.hsk_vocab:
                level_distribution[self.hsk_vocab[word]] += 1
            else:
                level_distribution["超出HSK"] += 1
        weighted_sum = 0
        for level, count in level_distribution.items():
            if level != "超出HSK":
                weighted_sum += level * count
        avg_level = weighted_sum / unique_count if unique_count > 0 else 0
        return {
            "total_words": total_words,
            "unique_words": unique_count,
            "type_token_ratio": ttr,
            "level_distribution": level_distribution,
            "estimated_vocabulary_level": round(avg_level),
        }

    def _analyze_grammar(self, text):
        """分析语法使用情况"""
        basic_errors = self._detect_basic_grammar_errors(text)
        sentence_count = len(re.split(r"[。！？\\.!?]", text))
        avg_sentence_length = len(text) / sentence_count if sentence_count > 0 else 0
        grammar_usage = {}
        for level, patterns in self.hsk_grammar.items():
            for pattern in patterns:
                if pattern in text:
                    grammar_usage.setdefault(level, []).append(pattern)
        highest_level_used = max(grammar_usage.keys()) if grammar_usage else 1
        return {
            "basic_errors": basic_errors,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "grammar_usage": grammar_usage,
            "estimated_grammar_level": highest_level_used,
        }

    def _detect_basic_grammar_errors(self, text):
        """检测基本语法错误（简化版）"""
        errors = []
        de_errors = re.findall(r"[形容词|动词]\\s*的\\s*[动词]", text)
        if de_errors:
            errors.append({"type": "的地得用法错误", "examples": de_errors})
        measure_errors = re.findall(r"[一二三四五六七八九十]\\s*[名词]", text)
        if measure_errors:
            errors.append({"type": "量词缺失", "examples": measure_errors})
        return errors

    def _analyze_structure(self, text):
        """分析文本结构"""
        paragraphs = text.split("\n\n")
        paragraph_count = len(paragraphs)
        connectives = ["因为", "所以", "但是", "而且", "如果", "虽然", "不过", "然后", "首先", "其次", "最后", "总之"]
        connective_usage = {c: text.count(c) for c in connectives if text.count(c) > 0}
        coherence_scores = []
        nonempty = [p for p in paragraphs if p.strip()]
        if len(nonempty) > 1:
            paragraphs_tfidf = TfidfVectorizer().fit_transform(nonempty)
            if paragraphs_tfidf.shape[0] > 1:
                similarities = cosine_similarity(paragraphs_tfidf)
                for i in range(similarities.shape[0]):
                    other = [j for j in range(similarities.shape[0]) if j != i]
                    if other:
                        coherence_scores.append(sum(similarities[i, j] for j in other) / len(other))
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
        return {
            "paragraph_count": paragraph_count,
            "connective_usage": connective_usage,
            "coherence_score": avg_coherence,
        }

    def _analyze_pronunciation(self, audio_features):
        """分析发音特征（示例）"""
        return {
            "tone_accuracy": audio_features.get("tone_accuracy", 0.8),
            "pronunciation_accuracy": audio_features.get("pronunciation_accuracy", 0.75),
            "fluency": audio_features.get("fluency", 0.7),
            "rhythm": audio_features.get("rhythm", 0.65),
            "problematic_sounds": audio_features.get("problematic_sounds", ["zh", "ch", "sh", "r"]),
        }

    def _analyze_communication(self, transcript):
        """分析交际能力（示例）"""
        functions = {
            "greeting": ["你好", "早上好", "晚上好", "嗨"],
            "farewell": ["再见", "拜拜", "回头见"],
            "thanks": ["谢谢", "感谢", "多谢"],
            "apology": ["对不起", "抱歉", "不好意思"],
            "request": ["可以", "能", "请", "麻烦"],
            "opinion": ["我认为", "我觉得", "我想", "我相信"],
        }
        function_usage = {}
        for func, markers in functions.items():
            for marker in markers:
                if marker in transcript:
                    function_usage[func] = function_usage.get(func, 0) + transcript.count(marker)
        turn_taking = ["那么", "所以", "接下来", "另外", "还有", "第一", "第二"]
        return {
            "function_usage": function_usage,
            "turn_taking_count": sum(transcript.count(w) for w in turn_taking),
            "estimated_communication_level": self._estimate_communication_level(function_usage),
        }

    def _estimate_communication_level(self, function_usage):
        """估计交际能力水平（简化版）"""
        d = len(function_usage)
        return 1 if d <= 1 else 2 if d <= 2 else 3 if d <= 3 else 4 if d <= 4 else 5 if d <= 5 else 6

    def _identify_weak_areas(self, assessment_result):
        """识别需要加强的薄弱环节"""
        weak_areas = {"vocabulary": [], "grammar": [], "expression": []}
        gpt_assessment = assessment_result.get("gpt_assessment", "")
        problem_section = re.search(r"问题分析[：:](.*?)改进建议", gpt_assessment, re.DOTALL)
        if problem_section:
            problem_text = problem_section.group(1).strip()
            if "词汇" in problem_text or "单词" in problem_text:
                vocab_matches = re.findall(r'「([^」]+)」|"([^"]+)"', problem_text)
                for match in vocab_matches:
                    for group in match:
                        if group and len(group) <= 4:
                            weak_areas["vocabulary"].append(group)
            if "语法" in problem_text or "句子" in problem_text:
                for level, patterns in self.hsk_grammar.items():
                    for pattern in patterns:
                        if pattern in problem_text:
                            weak_areas["grammar"].append(pattern)
            if "表达" in problem_text or "交际" in problem_text:
                for expr_type in ["请求", "建议", "比较", "因果", "条件", "假设"]:
                    if expr_type in problem_text:
                        weak_areas["expression"].append(expr_type)
        return weak_areas

    def _call_gpt(self, prompt):
        """调用GPT模型生成评估和反馈"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"评估生成出错: {str(e)}"

    def visualize_assessment(self, assessment_result, output_file=None):
        """可视化评估结果"""
        plt.figure(figsize=(15, 12))
        plt.subplot(2, 2, 1)
        vocab_analysis = assessment_result.get("vocab_analysis", {})
        level_dist = vocab_analysis.get("level_distribution", {})
        levels, counts = [], []
        for level, count in level_dist.items():
            if level != "超出HSK":
                levels.append(f"HSK{level}")
                counts.append(count)
        if "超出HSK" in level_dist:
            levels.append("超出HSK")
            counts.append(level_dist["超出HSK"])
        plt.bar(levels, counts, color="skyblue")
        plt.title("词汇HSK级别分布")
        plt.xlabel("HSK级别")
        plt.ylabel("词汇数量")

        plt.subplot(2, 2, 2)
        grammar_analysis = assessment_result.get("grammar_analysis", {})
        grammar_usage = grammar_analysis.get("grammar_usage", {})
        g_levels = [f"HSK{level}" for level in grammar_usage]
        g_counts = [len(patterns) for patterns in grammar_usage.values()]
        plt.bar(g_levels, g_counts, color="lightgreen")
        plt.title("语法点使用分布")
        plt.xlabel("HSK级别")
        plt.ylabel("语法点数量")

        plt.subplot(2, 2, 3)
        gpt_assessment = assessment_result.get("gpt_assessment", "")
        scores = {"词汇": 0, "语法": 0, "内容": 0, "结构": 0, "语用": 0}
        for dimension in scores:
            score_match = re.search(rf"{dimension}[：:]\\s*(\\d+)", gpt_assessment)
            if score_match:
                scores[dimension] = int(score_match.group(1))
        categories = list(scores.keys())
        values = list(scores.values())
        # 先按原始类别数计算角度，再闭合多边形（与 compare_assessments 一致）
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        categories.append(categories[0])
        values.append(values[0])
        angles.append(angles[0])
        ax = plt.subplot(2, 2, 3, polar=True)
        ax.plot(angles, values, linewidth=2, linestyle="solid")
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
        ax.set_ylim(0, 100)
        plt.title("能力维度评分")

        plt.subplot(2, 2, 4)
        plt.axis("off")
        text_summary = (
            f"评估摘要:\n文本长度: {len(assessment_result.get('text', ''))} 字符\n"
            f"词汇总数: {vocab_analysis.get('total_words', 0)} 词\n"
            f"估计词汇水平: HSK{vocab_analysis.get('estimated_vocabulary_level', 0)}\n"
            f"估计语法水平: HSK{grammar_analysis.get('estimated_grammar_level', 0)}\n"
            f"优点: {self._extract_section(gpt_assessment, '优点分析', '问题分析')}\n"
            f"问题: {self._extract_section(gpt_assessment, '问题分析', '改进建议')}"
        )
        plt.text(0, 0.5, text_summary, fontsize=10, verticalalignment="center")
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
            return output_file
        plt.show()
        return None

    def _extract_section(self, text, start_marker, end_marker):
        """从文本中提取特定部分"""
        pattern = f"{start_marker}[：:](.*?){end_marker}"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else "无内容"


def main():
    """主函数示例"""
    assessor = ChineseLanguageAssessment()
    sample_writing = "我叫李明，今年二十岁。我是中国留学生，现在在美国读大学。"
    writing_assessment = assessor.assess_writing(sample_writing, target_level=4)
    student_info = {
        "mother_tongue": "英语",
        "learning_time": "2年",
        "learning_goal": "能够在中国工作",
        "learning_style": "视觉学习者",
    }
    feedback = assessor.generate_personalized_feedback(writing_assessment, student_info)
    exercises = assessor.generate_practice_exercises(writing_assessment)
    visualization_file = "assessment_visualization.png"
    assessor.visualize_assessment(writing_assessment, visualization_file)
    print("=== 评估结果 ===")
    print(writing_assessment["gpt_assessment"])
    print("\n=== 个性化反馈 ===")
    print(feedback)
    print("\n=== 练习题 ===")
    print(exercises)
    print(f"\n评估可视化已保存至: {visualization_file}")


if __name__ == "__main__":
    main()
