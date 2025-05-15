# 布鲁姆认知分类法与AI教学设计整合模型
# 将布鲁姆六个认知层次与AI教学应用相结合

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
import random

class BloomAITeachingDesign:
    """布鲁姆认知分类法与AI教学设计整合模型"""
    
    def __init__(self):
        """初始化模型"""
        # 布鲁姆认知层次
        self.cognitive_levels = ["记忆", "理解", "应用", "分析", "评价", "创造"]
        
        # 每个认知层次的关键动词
        self.key_verbs = {
            "记忆": ["识别", "回忆", "列举", "描述", "匹配", "命名", "选择", "背诵"],
            "理解": ["解释", "概括", "举例", "推断", "比较", "对比", "分类", "总结"],
            "应用": ["执行", "实施", "使用", "操作", "演示", "计算", "解决", "验证"],
            "分析": ["区分", "组织", "归因", "比较", "对照", "检验", "质疑", "测试"],
            "评价": ["检查", "评论", "批判", "判断", "评估", "验证", "支持", "反驳"],
            "创造": ["设计", "建构", "计划", "产生", "制作", "创作", "开发", "撰写"]
        }
        
        # 每个认知层次的AI教学应用
        self.ai_applications = {
            "记忆": [
                "AI词汇学习卡片",
                "智能记忆复习系统",
                "个性化记忆曲线优化",
                "多模态记忆辅助工具",
                "汉字书写辅助系统",
                "情境化词汇记忆游戏"
            ],
            "理解": [
                "多模态概念解释生成",
                "智能概念图谱构建",
                "个性化解释生成器",
                "语法规则可视化工具",
                "文化背景知识拓展",
                "交互式理解检查系统"
            ],
            "应用": [
                "情境对话模拟系统",
                "任务型学习场景构建",
                "虚拟角色扮演环境",
                "语言应用能力评估",
                "跨文化交际模拟",
                "实时语言应用反馈"
            ],
            "分析": [
                "语言结构分析工具",
                "文本比较与对比系统",
                "语法模式识别助手",
                "篇章结构可视化",
                "语言现象规律发现",
                "修辞手法分析工具"
            ],
            "评价": [
                "智能同伴评价系统",
                "语言表达质量评估",
                "标准化评价辅助工具",
                "自我评估反馈系统",
                "多维度语言能力评估",
                "评价证据收集助手"
            ],
            "创造": [
                "AI辅助创意写作",
                "跨文化创意项目设计",
                "语言创新实验平台",
                "协作创作支持系统",
                "多模态语言作品创作",
                "原创语言资源开发"
            ]
        }
        
        # 每个认知层次的教学目标模板
        self.objective_templates = {
            "记忆": [
                "学生能够识别和回忆{content}",
                "学生能够列举{content}",
                "学生能够描述{content}的基本特征",
                "学生能够匹配{content}与其对应的{attribute}"
            ],
            "理解": [
                "学生能够解释{content}的含义",
                "学生能够用自己的话概括{content}",
                "学生能够举例说明{content}",
                "学生能够比较和对比{content}与{related_content}"
            ],
            "应用": [
                "学生能够在新情境中使用{content}",
                "学生能够解决涉及{content}的实际问题",
                "学生能够演示{content}的正确用法",
                "学生能够选择适当的{content}完成交际任务"
            ],
            "分析": [
                "学生能够分析{content}的组成部分和结构",
                "学生能够识别{content}中的模式和规律",
                "学生能够区分{content}中的事实和观点",
                "学生能够检验{content}的内在逻辑"
            ],
            "评价": [
                "学生能够基于特定标准评估{content}的质量",
                "学生能够判断{content}的适当性和有效性",
                "学生能够批判性地审视{content}",
                "学生能够评价不同{content}的优缺点"
            ],
            "创造": [
                "学生能够创作原创的{content}",
                "学生能够设计解决{problem}的方案",
                "学生能够整合所学知识开发新的{content}",
                "学生能够构建创新的{content}"
            ]
        }
        
        # 每个认知层次的评估方法
        self.assessment_methods = {
            "记忆": [
                "选择题",
                "填空题",
                "匹配题",
                "简答题",
                "闪卡测试",
                "词汇拼写测试"
            ],
            "理解": [
                "解释题",
                "概括题",
                "举例题",
                "翻译题",
                "图表解释",
                "问答题"
            ],
            "应用": [
                "情境问题解决",
                "角色扮演",
                "实际操作任务",
                "案例分析",
                "模拟活动",
                "应用项目"
            ],
            "分析": [
                "分类题",
                "比较题",
                "关系分析",
                "结构分析",
                "错误分析",
                "文本分析"
            ],
            "评价": [
                "评论题",
                "评估报告",
                "辩论活动",
                "批判性分析",
                "同伴评价",
                "自我评估"
            ],
            "创造": [
                "创作项目",
                "设计任务",
                "研究计划",
                "作品创作",
                "解决方案开发",
                "创新实验"
            ]
        }
        
        # 每个认知层次的教师AI素养表现
        self.teacher_ai_literacy = {
            "记忆": [
                "了解AI记忆辅助工具的功能和特点",
                "掌握使用AI工具辅助记忆的基本方法",
                "能够选择适合不同记忆任务的AI工具",
                "理解记忆规律与AI算法的结合点",
                "能设计简单的AI辅助记忆活动"
            ],
            "理解": [
                "了解AI内容生成技术的原理和特点",
                "掌握提示工程技巧生成解释性内容",
                "能够评估AI生成解释的准确性和适用性",
                "调整AI输出以适应不同学习者的认知水平",
                "能设计多模态理解支持活动"
            ],
            "应用": [
                "了解情境模拟和AI对话系统的功能",
                "掌握设计真实应用场景的技巧",
                "能够创建交互式语言应用练习",
                "整合AI工具支持实时应用反馈",
                "设计虚拟情境中的语言应用任务"
            ],
            "分析": [
                "了解文本分析和语言结构可视化技术",
                "掌握使用AI工具进行语言对比分析",
                "能够引导学生使用AI进行模式识别",
                "设计促进分析思维的AI辅助活动",
                "整合多种AI工具进行深度分析"
            ],
            "评价": [
                "了解AI评估工具和同伴评价系统",
                "掌握设置评价标准和评估指标的方法",
                "能够指导学生使用AI进行自我评价",
                "分析AI评估结果并提供补充指导",
                "设计平衡AI评估和人工评价的活动"
            ],
            "创造": [
                "了解AI协作创作和创意支持工具",
                "掌握使用AI辅助创意生成的技巧",
                "能够引导学生创造性使用AI",
                "设计鼓励原创性思维的AI辅助活动",
                "整合多种创作工具构建创意平台"
            ]
        }
        
        # 教学设计库
        self.design_library = {}
    
    def generate_teaching_objectives(self, 
                                     cognitive_level: str, 
                                     content: str,
                                     count: int = 3) -> List[str]:
        """生成特定认知层次的教学目标"""
        if cognitive_level not in self.cognitive_levels:
            raise ValueError(f"认知层次必须是以下之一: {', '.join(self.cognitive_levels)}")
        
        templates = self.objective_templates[cognitive_level]
        
        # 根据内容类型替换模板中的占位符
        objectives = []
        for i in range(min(count, len(templates))):
            template = templates[i % len(templates)]
            
            # 处理可能的其他占位符
            if "{related_content}" in template:
                if cognitive_level == "理解":
                    related_contents = {
                        "汉字": "拼音",
                        "词汇": "同义词",
                        "语法": "句型",
                        "成语": "俗语",
                        "课文": "相关文章",
                        "对话": "情境表达"
                    }
                    related = related_contents.get(content, "相关内容")
                    objective = template.format(content=content, related_content=related)
                else:
                    objective = template.format(content=content)
            elif "{attribute}" in template:
                if cognitive_level == "记忆":
                    attributes = {
                        "汉字": "发音",
                        "词汇": "意义",
                        "语法": "用法",
                        "成语": "典故",
                        "课文": "主题",
                        "对话": "功能"
                    }
                    attribute = attributes.get(content, "特征")
                    objective = template.format(content=content, attribute=attribute)
                else:
                    objective = template.format(content=content)
            elif "{problem}" in template:
                if cognitive_level == "创造":
                    problems = {
                        "汉字": "汉字记忆难题",
                        "词汇": "词汇应用困难",
                        "语法": "语法理解障碍",
                        "成语": "成语使用不当",
                        "课文": "阅读理解问题",
                        "对话": "交际困境"
                    }
                    problem = problems.get(content, "学习难题")
                    objective = template.format(problem=problem)
                else:
                    objective = template.format(content=content)
            else:
                objective = template.format(content=content)
            
            objectives.append(objective)
        
        return objectives
    
    def suggest_ai_applications(self, 
                                cognitive_level: str, 
                                count: int = 3) -> List[str]:
        """建议特定认知层次的AI应用"""
        if cognitive_level not in self.cognitive_levels:
            raise ValueError(f"认知层次必须是以下之一: {', '.join(self.cognitive_levels)}")
        
        applications = self.ai_applications[cognitive_level]
        selected = random.sample(applications, min(count, len(applications)))
        
        return selected
    
    def suggest_assessment_methods(self, 
                                  cognitive_level: str, 
                                  count: int = 3) -> List[str]:
        """建议特定认知层次的评估方法"""
        if cognitive_level not in self.cognitive_levels:
            raise ValueError(f"认知层次必须是以下之一: {', '.join(self.cognitive_levels)}")
        
        methods = self.assessment_methods[cognitive_level]
        selected = random.sample(methods, min(count, len(methods)))
        
        return selected
    
    def get_teacher_ai_literacy(self, 
                                cognitive_level: str) -> List[str]:
        """获取特定认知层次教师AI素养表现"""
        if cognitive_level not in self.cognitive_levels:
            raise ValueError(f"认知层次必须是以下之一: {', '.join(self.cognitive_levels)}")
        
        return self.teacher_ai_literacy[cognitive_level]
    
    def create_lesson_design(self, 
                             title: str,
                             content_type: str,
                             content: str,
                             primary_level: str,
                             secondary_levels: List[str] = None,
                             hsk_level: int = 3,
                             duration: int = 45) -> Dict:
        """创建教学设计"""
        # 验证认知层次
        if primary_level not in self.cognitive_levels:
            raise ValueError(f"主要认知层次必须是以下之一: {', '.join(self.cognitive_levels)}")
        
        if secondary_levels:
            for level in secondary_levels:
                if level not in self.cognitive_levels:
                    raise ValueError(f"次要认知层次必须是以下之一: {', '.join(self.cognitive_levels)}")
        
        # 确保次要认知层次不包含主要认知层次
        if secondary_levels and primary_level in secondary_levels:
            secondary_levels.remove(primary_level)
        
        # 如果没有提供次要认知层次，默认选择1-2个其他层次
        if not secondary_levels:
            available_levels = [l for l in self.cognitive_levels if l != primary_level]
            count = min(2, len(available_levels))
            secondary_levels = random.sample(available_levels, count)
        
        # 生成教学目标
        objectives = []
        objectives.extend(self.generate_teaching_objectives(primary_level, content, 2))
        for level in secondary_levels:
            objectives.extend(self.generate_teaching_objectives(level, content, 1))
        
        # 建议AI应用
        ai_applications = {}
        ai_applications[primary_level] = self.suggest_ai_applications(primary_level, 3)
        for level in secondary_levels:
            ai_applications[level] = self.suggest_ai_applications(level, 1)
        
        # 建议评估方法
        assessment_methods = {}
        assessment_methods[primary_level] = self.suggest_assessment_methods(primary_level, 2)
        for level in secondary_levels:
            assessment_methods[level] = self.suggest_assessment_methods(level, 1)
        
        # 提取教师AI素养要求
        teacher_ai_literacy = {}
        teacher_ai_literacy[primary_level] = self.get_teacher_ai_literacy(primary_level)
        for level in secondary_levels:
            teacher_ai_literacy[level] = self.get_teacher_ai_literacy(level)[:2]
        
        # 生成教学活动流程
        activities = []
        
        # 热身活动（通常是记忆或理解层次）
        warmup_level = "记忆" if primary_level != "记忆" else "理解"
        warmup_ai = random.choice(self.ai_applications[warmup_level])
        activities.append({
            "name": f"热身: {content_type}导入",
            "description": f"使用{warmup_ai}激活学生已有知识，导入新{content_type}学习。",
            "duration": 5,
            "cognitive_level": warmup_level,
            "ai_application": warmup_ai
        })
        
        # 主要活动（主要认知层次）
        main_ai_apps = ai_applications[primary_level]
        main_duration = 25 if len(secondary_levels) <= 1 else 20
        activities.append({
            "name": f"主活动: {primary_level}层次{content_type}学习",
            "description": f"学生通过{main_ai_apps[0]}和{main_ai_apps[1]}，在{primary_level}层次上学习{content}。",
            "duration": main_duration,
            "cognitive_level": primary_level,
            "ai_application": main_ai_apps
        })
        
        # 次要活动（次要认知层次）
        remaining_time = duration - 5 - main_duration - 5  # 减去热身、主活动和总结的时间
        per_activity_time = remaining_time // len(secondary_levels)
        
        for i, level in enumerate(secondary_levels):
            activities.append({
                "name": f"辅助活动{i+1}: {level}层次拓展",
                "description": f"学生通过{ai_applications[level][0]}，在{level}层次上拓展{content}学习。",
                "duration": per_activity_time,
                "cognitive_level": level,
                "ai_application": ai_applications[level][0]
            })
        
        # 总结活动（通常是评价层次）
        summary_level = "评价" if primary_level != "评价" else "创造"
        summary_ai = random.choice(self.ai_applications[summary_level])
        activities.append({
            "name": "总结与反思",
            "description": f"学生使用{summary_ai}对本课学习进行评估和总结。",
            "duration": 5,
            "cognitive_level": summary_level,
            "ai_application": summary_ai
        })
        
        # 创建完整教学设计
        design = {
            "title": title,
            "content_type": content_type,
            "content": content,
            "hsk_level": hsk_level,
            "duration": duration,
            "cognitive_levels": {
                "primary": primary_level,
                "secondary": secondary_levels
            },
            "objectives": objectives,
            "ai_applications": ai_applications,
            "assessment_methods": assessment_methods,
            "teacher_ai_literacy": teacher_ai_literacy,
            "activities": activities,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存到设计库
        design_id = f"{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.design_library[design_id] = design
        
        return design
    
    def export_design(self, design_id: str, format: str = "json") -> str:
        """导出教学设计"""
        if design_id not in self.design_library:
            raise ValueError(f"未找到设计ID: {design_id}")
        
        design = self.design_library[design_id]
        
        if format.lower() == "json":
            filename = f"{design_id}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(design, f, ensure_ascii=False, indent=4)
            return filename
        
        elif format.lower() == "markdown":
            filename = f"{design_id}.md"
            
            md_content = f"# {design['title']}\n\n"
            md_content += f"* HSK级别: {design['hsk_level']}\n"
            md_content += f"* 内容类型: {design['content_type']}\n"
            md_content += f"* 内容: {design['content']}\n"
            md_content += f"* 时长: {design['duration']}分钟\n"
            md_content += f"* 创建时间: {design['created_at']}\n\n"
            
            md_content += "## 认知层次\n\n"
            md_content += f"* 主要认知层次: {design['cognitive_levels']['primary']}\n"
            md_content += f"* 次要认知层次: {', '.join(design['cognitive_levels']['secondary'])}\n\n"
            
            md_content += "## 教学目标\n\n"
            for obj in design['objectives']:
                md_content += f"* {obj}\n"
            md_content += "\n"
            
            md_content += "## AI应用\n\n"
            for level, apps in design['ai_applications'].items():
                md_content += f"### {level}层次应用\n\n"
                for app in apps:
                    md_content += f"* {app}\n"
                md_content += "\n"
            
            md_content += "## 评估方法\n\n"
            for level, methods in design['assessment_methods'].items():
                md_content += f"### {level}层次评估\n\n"
                for method in methods:
                    md_content += f"* {method}\n"
                md_content += "\n"
            
            md_content += "## 教师AI素养要求\n\n"
            for level, skills in design['teacher_ai_literacy'].items():
                md_content += f"### {level}层次素养\n\n"
                for skill in skills:
                    md_content += f"* {skill}\n"
                md_content += "\n"
            
            md_content += "## 教学活动\n\n"
            for i, activity in enumerate(design['activities']):
                md_content += f"### 活动{i+1}: {activity['name']} ({activity['duration']}分钟)\n\n"
                md_content += f"* 认知层次: {activity['cognitive_level']}\n"
                md_content += f"* 描述: {activity['description']}\n"
                if isinstance(activity['ai_application'], list):
                    md_content += "* AI应用:\n"
                    for app in activity['ai_application']:
                        md_content += f"  - {app}\n"
                else:
                    md_content += f"* AI应用: {activity['ai_application']}\n"
                md_content += "\n"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            return filename
        
        elif format.lower() == "html":
            filename = f"{design_id}.html"
            
            html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{design['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .info-box {{ background-color: #f8f9fa; border-left: 4px solid #4285f4; padding: 15px; margin-bottom: 20px; }}
        .activity {{ background-color: #e9f5ff; border-radius: 5px; padding: 15px; margin-bottom: 15px; }}
        .level-primary {{ color: #4285f4; font-weight: bold; }}
        .level-secondary {{ color: #0f9d58; }}
        ul {{ padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{design['title']}</h1>
        
        <div class="info-box">
            <p><strong>HSK级别:</strong> {design['hsk_level']}</p>
            <p><strong>内容类型:</strong> {design['content_type']}</p>
            <p><strong>内容:</strong> {design['content']}</p>
            <p><strong>时长:</strong> {design['duration']}分钟</p>
            <p><strong>创建时间:</strong> {design['created_at']}</p>
        </div>
        
        <h2>认知层次</h2>
        <p><span class="level-primary">主要认知层次:</span> {design['cognitive_levels']['primary']}</p>
        <p><span class="level-secondary">次要认知层次:</span> {', '.join(design['cognitive_levels']['secondary'])}</p>
        
        <h2>教学目标</h2>
        <ul>
"""
            
            for obj in design['objectives']:
                html_content += f"            <li>{obj}</li>\n"
            
            html_content += """        </ul>
        
        <h2>AI应用</h2>
"""
            
            for level, apps in design['ai_applications'].items():
                html_content += f"        <h3>{level}层次应用</h3>\n        <ul>\n"
                for app in apps:
                    html_content += f"            <li>{app}</li>\n"
                html_content += "        </ul>\n"
            
            html_content += """        <h2>评估方法</h2>
"""
            
            for level, methods in design['assessment_methods'].items():
                html_content += f"        <h3>{level}层次评估</h3>\n        <ul>\n"
                for method in methods:
                    html_content += f"            <li>{method}</li>\n"
                html_content += "        </ul>\n"
            
            html_content += """        <h2>教师AI素养要求</h2>
"""
            
            for level, skills in design['teacher_ai_literacy'].items():
                html_content += f"        <h3>{level}层次素养</h3>\n        <ul>\n"
                for skill in skills:
                    html_content += f"            <li>{skill}</li>\n"
                html_content += "        </ul>\n"
            
            html_content += """        <h2>教学活动</h2>
"""
            
            for i, activity in enumerate(design['activities']):
                html_content += f"""        <div class="activity">
            <h3>活动{i+1}: {activity['name']} ({activity['duration']}分钟)</h3>
            <p><strong>认知层次:</strong> {activity['cognitive_level']}</p>
            <p><strong>描述:</strong> {activity['description']}</p>
            <p><strong>AI应用:</strong></p>
            <ul>
"""
                if isinstance(activity['ai_application'], list):
                    for app in activity['ai_application']:
                        html_content += f"                <li>{app}</li>\n"
                else:
                    html_content += f"                <li>{activity['ai_application']}</li>\n"
                
                html_content += """            </ul>
        </div>
"""
            
            html_content += """    </div>
</body>
</html>
"""
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            return filename
        
        else:
            raise ValueError(f"不支持的格式: {format}，支持的格式有: json, markdown, html")
    
    def visualize_design(self, design_id: str, output_file: str = None) -> str:
        """可视化教学设计"""
        if design_id not in self.design_library:
            raise ValueError(f"未找到设计ID: {design_id}")
        
        design = self.design_library[design_id]
        
        plt.figure(figsize=(15, 10))
        
        # 1. 饼图：认知层次分布
        ax1 = plt.subplot(2, 2, 1)
        
        # 计算各认知层次在活动中的时间分布
        level_duration = {}
        for activity in design['activities']:
            level = activity['cognitive_level']
            if level not in level_duration:
                level_duration[level] = 0
            level_duration[level] += activity['duration']
        
        # 准备饼图数据
        levels = list(level_duration.keys())
        durations = list(level_duration.values())
        
        # 高亮主要认知层次
        colors = ['lightblue'] * len(levels)
        highlight_index = levels.index(design['cognitive_levels']['primary']) if design['cognitive_levels']['primary'] in levels else None
        if highlight_index is not None:
            colors[highlight_index] = 'royalblue'
        
        # 绘制饼图
        wedges, texts, autotexts = ax1.pie(
            durations, 
            labels=levels, 
            colors=colors,
            autopct='%1.1f%%', 
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # 调整文本
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('black')
        
        ax1.set_title('认知层次时间分布')
        
        # 2. 水平条形图：AI应用分布
        ax2 = plt.subplot(2, 2, 2)
        
        # 准备条形图数据
        all_apps = []
        for level, apps in design['ai_applications'].items():
            for app in apps:
                if app not in all_apps:
                    all_apps.append(app)
        
        # 计算每个AI应用的使用次数
        app_counts = {}
        for app in all_apps:
            app_counts[app] = 0
        
        for activity in design['activities']:
            if isinstance(activity['ai_application'], list):
                for app in activity['ai_application']:
                    if app in app_counts:
                        app_counts[app] += 1
            else:
                app = activity['ai_application']
                if app in app_counts:
                    app_counts[app] += 1
        
        # 排序显示
        apps = sorted(all_apps, key=lambda x: app_counts[x], reverse=True)
        counts = [app_counts[app] for app in apps]
        
        # 绘制条形图
        bars = ax2.barh(apps, counts, color='lightgreen')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.0f}", va='center')
        
        ax2.set_title('AI应用使用频率')
        ax2.set_xlabel('使用次数')
        
        # 3. 时间线图：教学活动流程
        ax3 = plt.subplot(2, 1, 2)
        
        # 准备时间线数据
        activities = design['activities']
        activity_names = [act['name'] for act in activities]
        start_times = [0]
        for i in range(1, len(activities)):
            start_times.append(start_times[i-1] + activities[i-1]['duration'])
        durations = [act['duration'] for act in activities]
        
        # 设定颜色映射
        cmap = {
            "记忆": "royalblue",
            "理解": "lightblue",
            "应用": "lightgreen",
            "分析": "gold",
            "评价": "orange",
            "创造": "tomato"
        }
        
        colors = [cmap[act['cognitive_level']] for act in activities]
        
        # 绘制甘特图样式的时间线
        for i, (name, start, duration, color) in enumerate(zip(activity_names, start_times, durations, colors)):
            ax3.barh(i, duration, left=start, height=0.5, color=color, alpha=0.8, edgecolor='black')
            # 在条形中间添加文本标签
            ax3.text(start + duration/2, i, name, ha='center', va='center', fontsize=8)
        
        # 设置y轴标签和刻度
        ax3.set_yticks(range(len(activity_names)))
        ax3.set_yticklabels(["活动"+str(i+1) for i in range(len(activity_names))])
        
        # 设置x轴为时间刻度
        ax3.set_xticks(range(0, design['duration']+1, 5))
        ax3.set_xlabel('时间 (分钟)')
        
        # 添加图例
        handles = [plt.Rectangle((0,0),1,1, color=cmap[level]) for level in self.cognitive_levels]
        ax3.legend(handles, self.cognitive_levels, loc='upper right')
        
        ax3.set_title('教学活动时间线')
        ax3.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        if output_file:
            plt.savefig(output_file)
        else:
            output_file = f"{design_id}_visualization.png"
            plt.savefig(output_file)
        
        plt.close()
        
        return output_file
    
    def list_designs(self) -> List[Dict]:
        """列出所有教学设计"""
        design_list = []
        for design_id, design in self.design_library.items():
            design_list.append({
                "id": design_id,
                "title": design['title'],
                "content": design['content'],
                "primary_level": design['cognitive_levels']['primary'],
                "hsk_level": design['hsk_level'],
                "created_at": design['created_at']
            })
        
        return design_list
    
    def save_library(self, filename: str = "bloom_ai_design_library.json") -> str:
        """保存设计库"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.design_library, f, ensure_ascii=False, indent=4)
        
        return filename
    
    def load_library(self, filename: str = "bloom_ai_design_library.json") -> None:
        """加载设计库"""
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                self.design_library = json.load(f)


def main():
    """主函数示例"""
    # 初始化设计模型
    model = BloomAITeachingDesign()
    
    # 创建新教学设计
    design = model.create_lesson_design(
        title="汉字构成与意义的多角度分析",
        content_type="汉字",
        content="形声字",
        primary_level="分析",
        hsk_level=4,
        duration=45
    )
    
    # 获取设计ID
    design_id = list(model.design_library.keys())[-1]
    
    # 导出为Markdown
    md_file = model.export_design(design_id, "markdown")
    print(f"教学设计已导出为Markdown: {md_file}")
    
    # 可视化设计
    vis_file = model.visualize_design(design_id)
    print(f"教学设计可视化已生成: {vis_file}")
    
    # 创建另一个教学设计
    design2 = model.create_lesson_design(
        title="中国传统节日的文化内涵创造性表达",
        content_type="文化主题",
        content="春节",
        primary_level="创造",
        secondary_levels=["理解", "评价"],
        hsk_level=5,
        duration=60
    )
    
    # 获取新设计ID
    design_id2 = list(model.design_library.keys())[-1]
    
    # 导出为HTML
    html_file = model.export_design(design_id2, "html")
    print(f"教学设计已导出为HTML: {html_file}")
    
    # 列出所有设计
    designs = model.list_designs()
    print("\n已创建的教学设计:")
    for d in designs:
        print(f"- {d['title']} (ID: {d['id']})")
    
    # 保存设计库
    lib_file = model.save_library()
    print(f"\n设计库已保存: {lib_file}")


if __name__ == "__main__":
    main()
