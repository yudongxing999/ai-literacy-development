# 国际中文教学资源智能生成系统核心代码
import os
import openai
import json
import pandas as pd
import argparse
from datetime import datetime

# 配置OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChineseTeachingAssistant:
    """国际中文教学资源智能生成系统"""
    
    def __init__(self):
        self.system_prompt = """
        你是一位经验丰富的国际中文教师，精通中文教学和教学资源设计。
        你将帮助教师生成高质量的中文教学资源，包括分级阅读材料、情境对话、语法练习、
        文化主题内容和评估测试。请确保内容符合目标学习者的水平，文化敏感且有教育价值。
        """
        self.hsk_levels = {
            1: "包含150个常用词汇，能理解和使用简单的词语和句子，满足具体的交际需求。",
            2: "包含300个常用词汇，能进行简单的日常交流。",
            3: "包含600个常用词汇，能满足日常交流需求。",
            4: "包含1200个常用词汇，能讨论广泛话题，表达清晰的观点。",
            5: "包含2500个常用词汇，能流利交流，阅读报纸杂志，观看中文影视节目。",
            6: "包含5000个常用词汇，接近母语水平，能自如地进行交际，阅读理解难度较大的文章。"
        }
    
    def generate_reading_material(self, level, topic, length=300, cultural_elements=True):
        """生成分级阅读材料"""
        level_description = self.hsk_levels.get(level, "中级水平")
        prompt = f"""
        请为HSK{level}级别的学生创建一篇中文阅读材料。
        
        要求：
        1. 主题：{topic}
        2. 长度约{length}字
        3. 难度：{level_description}
        4. 只使用HSK{level}级别及以下的词汇和语法
        5. 句式简单清晰，段落结构合理
        6. {"加入适当的中国文化元素" if cultural_elements else "不需要特别强调文化元素"}
        7. 在文章后添加5-8个生词解释（中英双语）
        8. 在文章后添加3-5个理解性问题
        
        请按以下格式输出：
        ---
        标题：
        
        正文：
        （阅读材料内容）
        
        生词：
        1. 词语（拼音）：中文解释（英文翻译）
        2. ...
        
        问题：
        1. ...
        2. ...
        ---
        """
        return self._call_gpt(prompt)
    
    def generate_dialogue(self, level, scenario, grammar_points=None, roles=2):
        """生成情境对话脚本"""
        level_description = self.hsk_levels.get(level, "中级水平")
        grammar_points_str = f"包含以下语法点：{', '.join(grammar_points)}" if grammar_points else "使用适合该级别的自然语法"
        
        prompt = f"""
        请创建一段HSK{level}级别的中文情境对话。
        
        要求：
        1. 情境：{scenario}
        2. 角色数量：{roles}人
        3. 难度：{level_description}
        4. {grammar_points_str}
        5. 对话应自然、实用，反映真实交际场景
        6. 长度适中（10-15个对话轮次）
        7. 在对话后提供以下内容：
           - 重点词汇和表达（中英双语）
           - 语法点解释
           - 文化注释（如有必要）
           - 2-3个基于对话的练习题
        
        请按以下格式输出：
        ---
        情境介绍：
        （简要描述情境和角色）
        
        对话：
        A：...
        B：...
        
        重点词汇和表达：
        1. 表达（拼音）：解释（英文翻译）
        2. ...
        
        语法点：
        1. ...解释和例句...
        2. ...
        
        练习：
        1. ...
        2. ...
        ---
        """
        return self._call_gpt(prompt)
    
    def generate_grammar_exercises(self, level, grammar_point, quantity=10, exercise_types=None):
        """生成语法练习题库"""
        if exercise_types is None:
            exercise_types = ["选择题", "填空题", "改错题"]
        
        level_description = self.hsk_levels.get(level, "中级水平")
        exercise_types_str = "、".join(exercise_types)
        
        prompt = f"""
        请为HSK{level}级别的学生创建关于"{grammar_point}"语法点的练习题。
        
        要求：
        1. 难度：{level_description}
        2. 题目类型：{exercise_types_str}
        3. 题目数量：共{quantity}题，各类型题目数量均衡
        4. 所有题目应符合实际语言使用场景
        5. 提供每题的参考答案和解析
        6. 先提供该语法点的简要解释和用法说明
        
        请按以下格式输出：
        ---
        语法点：{grammar_point}
        
        解释：
        （语法点的解释、用法和例句）
        
        练习题：
        
        一、选择题
        1. ...
        A. ...
        B. ...
        C. ...
        D. ...
        
        二、填空题
        ...
        
        三、改错题
        ...
        
        参考答案：
        1. ... （解析：...）
        2. ...
        ---
        """
        return self._call_gpt(prompt)
    
    def generate_cultural_content(self, level, cultural_topic, format_type="介绍文章"):
        """生成文化主题内容"""
        level_description = self.hsk_levels.get(level, "中级水平")
        
        prompt = f"""
        请为HSK{level}级别的国际中文学习者创建关于"{cultural_topic}"的文化内容。
        
        要求：
        1. 内容类型：{format_type}
        2. 语言难度：{level_description}
        3. 内容应准确、客观地介绍中国文化
        4. 考虑跨文化视角，适当进行中外文化比较
        5. 内容既有趣味性又有教育价值
        6. 提供关键词汇的解释（中英双语）
        7. 添加2-3个与主题相关的讨论问题
        8. 如有必要，包含文化禁忌提示或注意事项
        
        请按以下格式输出：
        ---
        主题：{cultural_topic}
        
        内容：
        （文化内容主体）
        
        关键词汇：
        1. 词语（拼音）：解释（英文翻译）
        2. ...
        
        讨论问题：
        1. ...
        2. ...
        
        文化提示：
        （如有相关禁忌或注意事项）
        ---
        """
        return self._call_gpt(prompt)
    
    def generate_assessment(self, level, assessment_type, content_focus, quantity=10):
        """生成评估测试工具"""
        level_description = self.hsk_levels.get(level, "中级水平")
        
        prompt = f"""
        请为HSK{level}级别的学生创建一份"{assessment_type}"类型的评估测试。
        
        要求：
        1. 测试重点：{content_focus}
        2. 难度：{level_description}
        3. 题目数量：{quantity}道题
        4. 测试应全面评估学生在该领域的能力
        5. 提供明确的评分标准和参考答案
        6. 测试指导说明简洁清晰
        
        请按以下格式输出：
        ---
        评估名称：HSK{level}级{content_focus}{assessment_type}
        
        测试说明：
        （时间限制、答题方式等）
        
        评分标准：
        （详细说明如何评分）
        
        测试内容：
        
        一、...
        1. ...
        2. ...
        
        二、...
        ...
        
        参考答案：
        1. ...
        2. ...
        ---
        """
        return self._call_gpt(prompt)
    
    def _call_gpt(self, prompt):
        """调用GPT模型生成内容"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # 或者使用其他适合的模型
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成内容时出错: {str(e)}"
    
    def save_to_file(self, content, filename=None, folder="generated_content"):
        """将生成的内容保存到文件"""
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"content_{timestamp}.txt"
        
        filepath = os.path.join(folder, filename)
        
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)
        
        return filepath


def main():
    parser = argparse.ArgumentParser(description='国际中文教学资源智能生成系统')
    parser.add_argument('--type', type=str, required=True, 
                        choices=['reading', 'dialogue', 'grammar', 'culture', 'assessment'],
                        help='要生成的资源类型')
    parser.add_argument('--level', type=int, required=True, choices=range(1, 7),
                        help='HSK级别 (1-6)')
    parser.add_argument('--topic', type=str, required=True,
                        help='主题或内容重点')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件名')
    
    args = parser.parse_args()
    
    assistant = ChineseTeachingAssistant()
    
    if args.type == 'reading':
        content = assistant.generate_reading_material(args.level, args.topic)
    elif args.type == 'dialogue':
        content = assistant.generate_dialogue(args.level, args.topic)
    elif args.type == 'grammar':
        content = assistant.generate_grammar_exercises(args.level, args.topic)
    elif args.type == 'culture':
        content = assistant.generate_cultural_content(args.level, args.topic)
    elif args.type == 'assessment':
        content = assistant.generate_assessment(args.level, "综合测试", args.topic)
    
    if args.output:
        filepath = assistant.save_to_file(content, args.output)
        print(f"内容已保存到 {filepath}")
    else:
        print(content)


if __name__ == "__main__":
    main()