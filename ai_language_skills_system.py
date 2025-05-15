# AI辅助语言能力培养系统
# 针对听、说、读、写四项语言技能的AI辅助培养方法

import os
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AILanguageSkillsSystem:
    """AI辅助语言能力培养系统"""
    
    def __init__(self):
        """初始化AI辅助语言能力培养系统"""
        # 语言技能类型
        self.skill_types = ["听力", "口语", "阅读", "写作"]
        
        # 不同HSK级别的能力要求
        self.hsk_requirements = {
            1: "能够理解和使用简单的词语和句子，满足具体的交际需求",
            2: "能够用汉语进行简单和直接的日常交流",
            3: "能够用汉语完成日常生活、学习和工作中的基本交际任务",
            4: "能够就广泛的话题进行交流，流利地表达自己的观点",
            5: "能够阅读中文报刊杂志，欣赏中文影视作品，用中文进行较为正式的演讲",
            6: "能够轻松地理解各种信息，能够用中文有效地表达自己的见解"
        }
        
        # 各技能的AI辅助工具
        self.ai_tools = {
            "听力": [
                {
                    "name": "多样化听力材料生成器",
                    "description": "基于学习者水平生成不同主题、场景和难度的听力材料",
                    "features": ["语速调节", "口音多样化", "背景噪音控制", "语言难度自适应"],
                    "suitable_levels": [1, 2, 3, 4, 5, 6]
                },
                {
                    "name": "智能字幕系统",
                    "description": "提供可调节的智能字幕支持，包括全字幕、关键词字幕或无字幕模式",
                    "features": ["字幕显示控制", "生词自动标注", "关键句突出显示", "复述提示"],
                    "suitable_levels": [1, 2, 3, 4, 5]
                },
                {
                    "name": "听力理解辅助工具",
                    "description": "提供听力内容的背景知识、文化解释和语境分析",
                    "features": ["文化背景解释", "语境分析", "隐含意义提示", "关键信息突出"],
                    "suitable_levels": [3, 4, 5, 6]
                },
                {
                    "name": "听力笔记助手",
                    "description": "辅助学习者进行听力笔记，提供关键词提示和结构化笔记模板",
                    "features": ["关键词识别", "笔记结构建议", "信息组织辅助", "要点回顾"],
                    "suitable_levels": [4, 5, 6]
                },
                {
                    "name": "听力错误分析系统",
                    "description": "分析学习者在听力理解中的常见错误模式，提供针对性指导",
                    "features": ["错误模式识别", "听力策略建议", "弱项识别", "进步追踪"],
                    "suitable_levels": [2, 3, 4, 5, 6]
                }
            ],
            "口语": [
                {
                    "name": "AI对话伙伴",
                    "description": "提供交互式口语练习，模拟各种真实场景的对话",
                    "features": ["情境对话模拟", "语速调节", "难度渐进", "个性化话题"],
                    "suitable_levels": [1, 2, 3, 4, 5, 6]
                },
                {
                    "name": "语音评估系统",
                    "description": "对发音、语调和流利度进行实时评估和反馈",
                    "features": ["声调评估", "发音纠正", "语流分析", "流利度评价"],
                    "suitable_levels": [1, 2, 3, 4, 5, 6]
                },
                {
                    "name": "情境模拟系统",
                    "description": "创建虚拟场景进行沉浸式口语练习，如面试、会议、购物等",
                    "features": ["真实场景模拟", "角色扮演", "情景变化", "压力调节"],
                    "suitable_levels": [3, 4, 5, 6]
                },
                {
                    "name": "口语流利度训练器",
                    "description": "通过节奏训练、连读练习和反应速度游戏提高口语流利度",
                    "features": ["节奏训练", "连读练习", "反应速度游戏", "停顿控制"],
                    "suitable_levels": [3, 4, 5, 6]
                },
                {
                    "name": "表达优化建议器",
                    "description": "提供更地道、更高级的表达方式建议，提升口语表达质量",
                    "features": ["表达升级建议", "地道用语推荐", "语体风格调整", "表达变体展示"],
                    "suitable_levels": [4, 5, 6]
                }
            ],
            "阅读": [
                {
                    "name": "自适应阅读材料库",
                    "description": "根据学习者水平自动调整文本难度的阅读材料系统",
                    "features": ["难度自适应", "兴趣匹配", "进度感知", "主题多样化"],
                    "suitable_levels": [1, 2, 3, 4, 5, 6]
                },
                {
                    "name": "智能阅读辅助工具",
                    "description": "提供词汇解释、语法分析和背景知识的阅读支持工具",
                    "features": ["即点即查", "语境释义", "语法提示", "文化注解"],
                    "suitable_levels": [1, 2, 3, 4, 5]
                },
                {
                    "name": "阅读理解训练系统",
                    "description": "生成多层次阅读理解问题，从字面理解到推理和评价",
                    "features": ["多层次问题", "理解深度递进", "答案解析", "思维引导"],
                    "suitable_levels": [2, 3, 4, 5, 6]
                },
                {
                    "name": "阅读速度训练器",
                    "description": "通过调整文本显示速度和范围，训练阅读速度和理解效率",
                    "features": ["速度调节", "眼动训练", "理解测试", "进度追踪"],
                    "suitable_levels": [3, 4, 5, 6]
                },
                {
                    "name": "文本分析可视化工具",
                    "description": "将文本结构、关键信息和逻辑关系可视化，辅助深度阅读",
                    "features": ["结构可视化", "关键词突出", "逻辑关系图示", "主题网络"],
                    "suitable_levels": [4, 5, 6]
                }
            ],
            "写作": [
                {
                    "name": "写作辅助系统",
                    "description": "提供语法、词汇和表达建议的实时写作辅助工具",
                    "features": ["语法检查", "词汇建议", "句式优化", "表达润色"],
                    "suitable_levels": [1, 2, 3, 4, 5, 6]
                },
                {
                    "name": "智能反馈工具",
                    "description": "对写作进行多维度评价和建议，包括内容、结构、语言和风格",
                    "features": ["内容评价", "结构分析", "语言评估", "风格建议"],
                    "suitable_levels": [2, 3, 4, 5, 6]
                },
                {
                    "name": "协作写作平台",
                    "description": "支持教师、同伴和AI共同参与的写作指导平台",
                    "features": ["多方反馈", "版本比较", "修改追踪", "协作编辑"],
                    "suitable_levels": [3, 4, 5, 6]
                },
                {
                    "name": "写作模板库",
                    "description": "提供各类写作的结构模板和表达框架，如书信、报告、论文等",
                    "features": ["多类型模板", "表达框架", "示例参考", "个性化调整"],
                    "suitable_levels": [2, 3, 4, 5]
                },
                {
                    "name": "创意写作激发器",
                    "description": "通过提示、图像和情境帮助克服写作障碍，激发创意",
                    "features": ["创意提示", "灵感激发", "写作引导", "障碍突破"],
                    "suitable_levels": [3, 4, 5, 6]
                }
            ]
        }
        
        # 各技能的学习活动模板
        self.activity_templates = {
            "听力": [
                {
                    "name": "差异辨识训练",
                    "description": "听辨相似音、音变和声调的细微差异，提高音辨能力",
                    "steps": [
                        "播放包含易混音素/声调的音频",
                        "学生选择正确选项或填写听到的内容",
                        "AI提供即时反馈和解析",
                        "逐步增加难度和干扰因素"
                    ],
                    "ai_tool_types": ["多样化听力材料生成器", "听力错误分析系统"],
                    "suitable_levels": [1, 2, 3]
                },
                {
                    "name": "预测性听力",
                    "description": "在听前预测内容，听中验证预测，听后反思差异",
                    "steps": [
                        "AI提供主题和背景信息",
                        "学生预测可能听到的内容",
                        "分段聆听，验证和调整预测",
                        "完成后比较预测与实际内容"
                    ],
                    "ai_tool_types": ["听力理解辅助工具", "智能字幕系统"],
                    "suitable_levels": [2, 3, 4]
                },
                {
                    "name": "选择性听力",
                    "description": "有目的地听取特定信息，忽略无关细节",
                    "steps": [
                        "AI设定听取目标（如时间、地点、数字等）",
                        "学生在听力中识别和记录目标信息",
                        "AI检查记录的准确性并提供反馈",
                        "增加信息密度和干扰因素"
                    ],
                    "ai_tool_types": ["多样化听力材料生成器", "听力笔记助手"],
                    "suitable_levels": [2, 3, 4, 5]
                },
                {
                    "name": "全球听力",
                    "description": "理解整体信息和说话者意图，把握主旨和态度",
                    "steps": [
                        "聆听较长对话或演讲",
                        "回答关于主旨、目的和态度的问题",
                        "AI引导分析言外之意和隐含态度",
                        "讨论不同文化背景如何影响理解"
                    ],
                    "ai_tool_types": ["听力理解辅助工具", "多样化听力材料生成器"],
                    "suitable_levels": [3, 4, 5, 6]
                },
                {
                    "name": "听写转述训练",
                    "description": "通过听写和转述练习提高听力信息处理能力",
                    "steps": [
                        "学生听取内容并进行完整或选择性听写",
                        "AI评估听写准确性并提供改进建议",
                        "学生用自己的话转述内容要点",
                        "AI评估转述的完整性和准确性"
                    ],
                    "ai_tool_types": ["听力笔记助手", "听力错误分析系统"],
                    "suitable_levels": [3, 4, 5, 6]
                }
            ],
            "口语": [
                {
                    "name": "模仿跟读训练",
                    "description": "跟随标准发音模型进行模仿练习，强化发音基础",
                    "steps": [
                        "AI播放标准发音示范",
                        "学生模仿并录制自己的发音",
                        "AI分析对比并提供具体改进建议",
                        "反复练习直至达到目标准确度"
                    ],
                    "ai_tool_types": ["语音评估系统", "多样化听力材料生成器"],
                    "suitable_levels": [1, 2, 3]
                },
                {
                    "name": "情境对话练习",
                    "description": "在模拟真实场景中进行对话练习，培养交际能力",
                    "steps": [
                        "AI创建具体情境（如餐厅点餐、问路等）",
                        "学生与AI进行角色对话",
                        "AI根据学生反应调整对话难度和方向",
                        "对话结束后AI提供表现评价和建议"
                    ],
                    "ai_tool_types": ["AI对话伙伴", "情境模拟系统"],
                    "suitable_levels": [1, 2, 3, 4, 5, 6]
                },
                {
                    "name": "即兴表达挑战",
                    "description": "对即时话题进行即兴表达，提高思维反应速度",
                    "steps": [
                        "AI随机生成话题或情境",
                        "学生在有限时间内准备并作出响应",
                        "AI评估流利度、相关性和表达质量",
                        "提供表现反馈和改进建议"
                    ],
                    "ai_tool_types": ["口语流利度训练器", "表达优化建议器"],
                    "suitable_levels": [3, 4, 5, 6]
                },
                {
                    "name": "叙述与描述训练",
                    "description": "练习事件叙述、过程描述和图片描述等表达技能",
                    "steps": [
                        "AI提供叙述/描述任务（如描述图片、讲述故事）",
                        "学生完成口头表达任务",
                        "AI评估内容组织、词汇使用和表达清晰度",
                        "提供针对性的表达优化建议"
                    ],
                    "ai_tool_types": ["表达优化建议器", "AI对话伙伴"],
                    "suitable_levels": [2, 3, 4, 5]
                },
                {
                    "name": "辩论与说服练习",
                    "description": "通过辩论和说服性表达培养高级口语能力",
                    "steps": [
                        "AI提出辩题并分配立场",
                        "学生准备并陈述论点",
                        "AI作为对方进行反驳",
                        "学生进行反驳和总结陈词"
                    ],
                    "ai_tool_types": ["AI对话伙伴", "表达优化建议器"],
                    "suitable_levels": [4, 5, 6]
                }
            ],
            "阅读": [
                {
                    "name": "递进式阅读",
                    "description": "从易到难逐步提升阅读难度，建立阅读信心",
                    "steps": [
                        "AI根据学生水平提供适当难度的文本",
                        "学生阅读并完成理解任务",
                        "根据表现自动调整下一篇文章难度",
                        "定期回顾进步情况和阅读策略"
                    ],
                    "ai_tool_types": ["自适应阅读材料库", "智能阅读辅助工具"],
                    "suitable_levels": [1, 2, 3, 4]
                },
                {
                    "name": "扫描与略读训练",
                    "description": "练习快速查找特定信息(扫描)和把握要点(略读)的技能",
                    "steps": [
                        "AI设定具体的信息搜索任务",
                        "学生在限定时间内找出信息",
                        "AI评估准确性和速度",
                        "提供扫描和略读策略指导"
                    ],
                    "ai_tool_types": ["阅读速度训练器", "自适应阅读材料库"],
                    "suitable_levels": [2, 3, 4, 5]
                },
                {
                    "name": "深度阅读分析",
                    "description": "深入分析文本结构、论点和修辞手法，培养批判性阅读能力",
                    "steps": [
                        "学生阅读复杂文本",
                        "完成结构分析、作者意图和论证方式等任务",
                        "AI提供分析反馈和引导",
                        "讨论不同解读和评价观点"
                    ],
                    "ai_tool_types": ["文本分析可视化工具", "阅读理解训练系统"],
                    "suitable_levels": [4, 5, 6]
                },
                {
                    "name": "跨文本阅读",
                    "description": "比较和整合多个文本的信息，培养高阶阅读能力",
                    "steps": [
                        "AI提供多个相关文本",
                        "学生识别共同点、差异和互补信息",
                        "完成跨文本分析任务",
                        "AI评估信息整合能力"
                    ],
                    "ai_tool_types": ["文本分析可视化工具", "自适应阅读材料库"],
                    "suitable_levels": [4, 5, 6]
                },
                {
                    "name": "阅读策略训练",
                    "description": "系统性学习和应用各种阅读策略，提高阅读效率",
                    "steps": [
                        "AI介绍特定阅读策略（如预测、推断、概括等）",
                        "学生在指导下应用该策略",
                        "评估策略应用效果",
                        "在不同类型文本中练习该策略"
                    ],
                    "ai_tool_types": ["阅读理解训练系统", "智能阅读辅助工具"],
                    "suitable_levels": [2, 3, 4, 5]
                }
            ],
            "写作": [
                {
                    "name": "句型扩展练习",
                    "description": "从基础句型开始，逐步添加细节和复杂结构",
                    "steps": [
                        "从简单句开始，如'我喜欢水果'",
                        "AI引导逐步添加细节、修饰和从句",
                        "学生尝试不同的扩展方式",
                        "AI评估句子结构和表达效果"
                    ],
                    "ai_tool_types": ["写作辅助系统", "智能反馈工具"],
                    "suitable_levels": [1, 2, 3]
                },
                {
                    "name": "引导式写作",
                    "description": "通过结构化的提示和引导完成特定类型的写作任务",
                    "steps": [
                        "AI提供写作框架和必要表达",
                        "学生按照引导完成写作",
                        "AI提供分步反馈",
                        "逐渐减少引导，增加自主写作部分"
                    ],
                    "ai_tool_types": ["写作模板库", "写作辅助系统"],
                    "suitable_levels": [1, 2, 3, 4]
                },
                {
                    "name": "过程式写作",
                    "description": "经历完整的写作过程：头脑风暴、草稿、修改和定稿",
                    "steps": [
                        "AI引导头脑风暴和内容规划",
                        "学生创作初稿",
                        "AI提供结构、内容和语言建议",
                        "学生修改并完成终稿"
                    ],
                    "ai_tool_types": ["智能反馈工具", "协作写作平台"],
                    "suitable_levels": [3, 4, 5, 6]
                },
                {
                    "name": "体裁转换练习",
                    "description": "将同一内容转换为不同体裁，如将叙述转为对话、报告等",
                    "steps": [
                        "学生阅读源文本",
                        "AI说明目标体裁的特点和要求",
                        "学生完成体裁转换写作",
                        "AI评估体裁特点把握和内容保留"
                    ],
                    "ai_tool_types": ["写作模板库", "智能反馈工具"],
                    "suitable_levels": [3, 4, 5]
                },
                {
                    "name": "创意写作workshop",
                    "description": "探索创造性写作，如故事、诗歌、剧本等",
                    "steps": [
                        "AI提供创意写作提示和启发",
                        "学生自由发挥创作",
                        "AI和同伴提供建设性反馈",
                        "修改并分享创作成果"
                    ],
                    "ai_tool_types": ["创意写作激发器", "协作写作平台"],
                    "suitable_levels": [3, 4, 5, 6]
                }
            ]
        }
        
        # 学习活动库
        self.activity_library = {}
    
    def suggest_tools(self, 
                     skill_type: str, 
                     hsk_level: int,
                     count: int = 2) -> List[Dict]:
        """推荐适合特定技能和级别的AI工具"""
        
        if skill_type not in self.skill_types:
            raise ValueError(f"技能类型必须是以下之一: {', '.join(self.skill_types)}")
        
        if hsk_level not in self.hsk_requirements:
            raise ValueError(f"HSK级别必须是1-6之间的整数")
        
        # 过滤适合当前HSK级别的工具
        suitable_tools = [tool for tool in self.ai_tools[skill_type] 
                         if hsk_level in tool['suitable_levels']]
        
        # 如果没有完全匹配的工具，选择最接近的
        if not suitable_tools:
            all_tools = self.ai_tools[skill_type]
            suitable_tools = sorted(all_tools, 
                                   key=lambda tool: min([abs(hsk_level - level) 
                                                       for level in tool['suitable_levels']]))
        
        # 选择工具数量
        selected_tools = suitable_tools[:min(count, len(suitable_tools))]
        
        return selected_tools
    
    def suggest_activities(self, 
                          skill_type: str, 
                          hsk_level: int,
                          count: int = 2) -> List[Dict]:
        """推荐适合特定技能和级别的学习活动"""
        
        if skill_type not in self.skill_types:
            raise ValueError(f"技能类型必须是以下之一: {', '.join(self.skill_types)}")
        
        if hsk_level not in self.hsk_requirements:
            raise ValueError(f"HSK级别必须是1-6之间的整数")
        
        # 过滤适合当前HSK级别的活动
        suitable_activities = [activity for activity in self.activity_templates[skill_type] 
                              if hsk_level in activity['suitable_levels']]
        
        # 如果没有完全匹配的活动，选择最接近的
        if not suitable_activities:
            all_activities = self.activity_templates[skill_type]
            suitable_activities = sorted(all_activities, 
                                        key=lambda activity: min([abs(hsk_level - level) 
                                                                for level in activity['suitable_levels']]))
        
        # 选择活动数量
        selected_activities = suitable_activities[:min(count, len(suitable_activities))]
        
        return selected_activities
    
    def create_skill_plan(self, 
                         title: str,
                         skill_type: str,
                         hsk_level: int,
                         focus_area: str,
                         duration_weeks: int,
                         sessions_per_week: int = 2,
                         minutes_per_session: int = 30) -> Dict:
        """创建语言技能培养计划"""
        
        if skill_type not in self.skill_types:
            raise ValueError(f"技能类型必须是以下之一: {', '.join(self.skill_types)}")
        
        if hsk_level not in self.hsk_requirements:
            raise ValueError(f"HSK级别必须是1-6之间的整数")
        
        # 根据总周数和每周课时，确定需要的活动数量
        total_sessions = duration_weeks * sessions_per_week
        
        # 推荐AI工具
        recommended_tools = self.suggest_tools(skill_type, hsk_level, 3)
        
        # 生成学习目标
        objectives = []
        
        if skill_type == "听力":
            objectives = [
                f"能够理解{focus_area}相关的{hsk_level}级别听力材料",
                f"能够识别{focus_area}中的关键信息和细节",
                f"能够跟上正常语速的{focus_area}对话或讲解",
                f"能够理解不同语境下{focus_area}的表达差异"
            ]
        elif skill_type == "口语":
            objectives = [
                f"能够流利地讨论{focus_area}相关话题",
                f"能够在{focus_area}情境中使用恰当的表达",
                f"能够清晰表达对{focus_area}的观点和看法",
                f"能够使用适当的语调和节奏表达{focus_area}内容"
            ]
        elif skill_type == "阅读":
            objectives = [
                f"能够理解{focus_area}主题的{hsk_level}级别文本",
                f"能够识别{focus_area}文本中的主要观点和支持细节",
                f"能够根据上下文推断{focus_area}文本中的生词意义",
                f"能够批判性地分析{focus_area}文本的内容和结构"
            ]
        elif skill_type == "写作":
            objectives = [
                f"能够写作关于{focus_area}的结构完整的文章",
                f"能够使用{hsk_level}级别的词汇和语法表达{focus_area}内容",
                f"能够根据不同目的和读者调整{focus_area}的写作风格",
                f"能够有效组织和连贯地表达{focus_area}相关的想法"
            ]
        
        # 生成每周学习计划
        weekly_plans = []
        
        # 获取所有适合的活动模板
        all_suitable_activities = []
        for activity in self.activity_templates[skill_type]:
            if hsk_level in activity['suitable_levels']:
                all_suitable_activities.append(activity)
        
        # 如果模板不够，允许重复使用，但优先使用不同模板
        for week in range(1, duration_weeks + 1):
            sessions = []
            
            for session in range(1, sessions_per_week + 1):
                # 如果已经用完所有活动模板，就重复使用
                activity_index = (week - 1) * sessions_per_week + (session - 1)
                activity_template = all_suitable_activities[activity_index % len(all_suitable_activities)]
                
                # 根据活动模板选择合适的AI工具
                activity_tools = []
                for tool_type in activity_template['ai_tool_types'][:2]:  # 最多使用两种工具
                    matching_tools = [t for t in recommended_tools if t['name'] == tool_type]
                    if matching_tools:
                        activity_tools.append(matching_tools[0])
                    else:
                        # 如果没有完全匹配的工具名称，选择任意工具
                        if recommended_tools:
                            activity_tools.append(random.choice(recommended_tools))
                
                # 创建本次课的具体内容
                session_content = {
                    "week": week,
                    "session": session,
                    "activity_name": activity_template['name'],
                    "focus": f"{focus_area}中的{self._generate_specific_focus(skill_type, focus_area, week, session)}",
                    "activity_description": activity_template['description'],
                    "steps": activity_template['steps'],
                    "tools": activity_tools,
                    "duration": minutes_per_session
                }
                
                sessions.append(session_content)
            
            weekly_plan = {
                "week": week,
                "theme": f"{focus_area} - {self._generate_weekly_theme(skill_type, focus_area, week)}",
                "sessions": sessions
            }
            
            weekly_plans.append(weekly_plan)
        
        # 设计评估方法
        assessment_methods = []
        
        if skill_type == "听力":
            assessment_methods = [
                "听力理解测试，包括多项选择和开放性问题",
                "听写练习，评估关键词和细节的识别能力",
                "听后转述任务，评估整体理解和信息处理能力",
                "不同语速和口音的听力材料理解能力测试"
            ]
        elif skill_type == "口语":
            assessment_methods = [
                "基于任务的口语表现评估，如角色扮演和情境对话",
                "即兴演讲测试，评估流利度和组织能力",
                "发音和语调准确度评估",
                "口语表达丰富度和适当性评估"
            ]
        elif skill_type == "阅读":
            assessment_methods = [
                "阅读理解测试，包括多项选择和开放性问题",
                "阅读速度和理解效率测试",
                "文本分析任务，评估深度理解和批判性阅读能力",
                "生词理解策略测试，评估应对未知词汇的能力"
            ]
        elif skill_type == "写作":
            assessment_methods = [
                "基于提示的写作任务，评估内容、组织和语言质量",
                "特定体裁的写作评估，如叙述、描述或议论文",
                "修改任务，评估编辑和提升写作质量的能力",
                "写作过程评估，包括规划、起草和修改各阶段"
            ]
        
        # 创建完整的技能培养计划
        skill_plan = {
            "title": title,
            "skill_type": skill_type,
            "hsk_level": hsk_level,
            "focus_area": focus_area,
            "duration_weeks": duration_weeks,
            "sessions_per_week": sessions_per_week,
            "minutes_per_session": minutes_per_session,
            "learning_objectives": objectives,
            "recommended_tools": recommended_tools,
            "weekly_plans": weekly_plans,
            "assessment_methods": assessment_methods,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存到活动库
        plan_id = f"{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.activity_library[plan_id] = skill_plan
        
        return skill_plan
    
    def _generate_weekly_theme(self, skill_type: str, focus_area: str, week: int) -> str:
        """生成每周主题"""
        
        themes = {
            "听力": [
                "基础听辨训练", "日常对话理解", "语篇结构识别", 
                "语气与态度解读", "隐含信息理解", "跨文化听力"
            ],
            "口语": [
                "发音与流利度", "日常交际表达", "详细描述与讲解", 
                "观点表达与辩论", "正式场合表达", "文化适当性表达"
            ],
            "阅读": [
                "基础阅读理解", "文本结构分析", "细节与推理", 
                "作者意图识别", "批判性阅读", "跨文化阅读视角"
            ],
            "写作": [
                "句式与段落结构", "描述与叙述写作", "说明与议论写作", 
                "格式与体裁特点", "修辞手法应用", "写作风格发展"
            ]
        }
        
        # 选择主题，确保不超出主题列表长度
        theme_index = (week - 1) % len(themes[skill_type])
        
        return f"{themes[skill_type][theme_index]}"
    
    def _generate_specific_focus(self, skill_type: str, focus_area: str, week: int, session: int) -> str:
        """生成具体专注点"""
        
        if skill_type == "听力":
            focuses = [
                "数字和数量表达", "时间和日期表达", "位置和方向词", 
                "情感和态度词", "连接词和转折语", "专业术语",
                "成语和习惯表达", "委婉和间接表达"
            ]
        elif skill_type == "口语":
            focuses = [
                "打招呼和寒暄", "请求和建议", "同意和拒绝", 
                "描述人物和场景", "讲述经历和故事", "表达观点和态度",
                "协商和辩论", "演讲和正式表达"
            ]
        elif skill_type == "阅读":
            focuses = [
                "关键词识别", "主旨和细节", "指代和连接", 
                "推理和预测", "作者态度", "批判性分析",
                "不同文体特点", "文化内涵理解"
            ]
        elif skill_type == "写作":
            focuses = [
                "基本句型", "段落构建", "连贯性和衔接性", 
                "不同文体写作", "修辞手法", "语体和风格",
                "修改和编辑", "创意表达"
            ]
        
        # 计算唯一的索引，确保不会超出列表长度
        index = ((week - 1) * 2 + (session - 1)) % len(focuses)
        
        return focuses[index]
    
    def export_plan(self, plan_id: str, format: str = "json") -> str:
        """导出语言技能培养计划"""
        if plan_id not in self.activity_library:
            raise ValueError(f"未找到计划ID: {plan_id}")
        
        plan = self.activity_library[plan_id]
        
        if format.lower() == "json":
            filename = f"{plan_id}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=4)
            return filename
        
        elif format.lower() == "markdown":
            filename = f"{plan_id}.md"
            
            md_content = f"# {plan['title']}\n\n"
            md_content += f"* 技能类型: {plan['skill_type']}\n"
            md_content += f"* HSK级别: {plan['hsk_level']}\n"
            md_content += f"* 专注领域: {plan['focus_area']}\n"
            md_content += f"* 总时长: {plan['duration_weeks']}周 ({plan['sessions_per_week']}次/周, {plan['minutes_per_session']}分钟/次)\n"
            md_content += f"* 创建时间: {plan['created_at']}\n\n"
            
            md_content += "## 学习目标\n\n"
            for obj in plan['learning_objectives']:
                md_content += f"* {obj}\n"
            md_content += "\n"
            
            md_content += "## 推荐AI工具\n\n"
            for tool in plan['recommended_tools']:
                md_content += f"### {tool['name']}\n\n"
                md_content += f"{tool['description']}\n\n"
                md_content += "**功能特点：**\n"
                for feature in tool['features']:
                    md_content += f"* {feature}\n"
                md_content += "\n"
            
            md_content += "## 每周学习计划\n\n"
            for week_plan in plan['weekly_plans']:
                md_content += f"### 第{week_plan['week']}周: {week_plan['theme']}\n\n"
                
                for session in week_plan['sessions']:
                    md_content += f"#### 课时{session['session']}: {session['activity_name']} ({session['duration']}分钟)\n\n"
                    md_content += f"**专注点**: {session['focus']}\n\n"
                    md_content += f"{session['activity_description']}\n\n"
                    
                    md_content += "**步骤**:\n"
                    for step in session['steps']:
                        md_content += f"1. {step}\n"
                    md_content += "\n"
                    
                    if session['tools']:
                        md_content += "**使用工具**:\n"
                        for tool in session['tools']:
                            md_content += f"* {tool['name']}\n"
                    md_content += "\n"
            
            md_content += "## 评估方法\n\n"
            for method in plan['assessment_methods']:
                md_content += f"* {method}\n"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            return filename
        
        elif format.lower() == "html":
            filename = f"{plan_id}.html"
            
            # 为不同技能类型设置不同的主题颜色
            color_theme = {
                "听力": "#4285f4",  # Blue
                "口语": "#ea4335",  # Red
                "阅读": "#fbbc05",  # Yellow
                "写作": "#34a853"   # Green
            }
            
            theme_color = color_theme.get(plan['skill_type'], "#4285f4")
            
            html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{plan['title']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; color: #333; line-height: 1.6; }}
        h1, h2, h3, h4 {{ color: {theme_color}; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .header {{ background-color: {theme_color}; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .info-box {{ background-color: #f8f9fa; border-left: 4px solid {theme_color}; padding: 15px; margin-bottom: 20px; }}
        .week-card {{ background-color: #f1f3f4; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
        .session-card {{ background-color: white; border: 1px solid #e0e0e0; border-radius: 5px; padding: 15px; margin-bottom: 15px; }}
        .tool-tag {{ background-color: #e8f0fe; color: {theme_color}; padding: 3px 8px; border-radius: 3px; margin-right: 5px; font-size: 0.9em; display: inline-block; margin-bottom: 5px; }}
        .tool-box {{ background-color: #e8f0fe; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
        ul, ol {{ padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{plan['title']}</h1>
            <p><strong>技能类型:</strong> {plan['skill_type']} | <strong>HSK级别:</strong> {plan['hsk_level']}</p>
        </div>
        
        <div class="info-box">
            <p><strong>专注领域:</strong> {plan['focus_area']}</p>
            <p><strong>总时长:</strong> {plan['duration_weeks']}周 ({plan['sessions_per_week']}次/周, {plan['minutes_per_session']}分钟/次)</p>
            <p><strong>创建时间:</strong> {plan['created_at']}</p>
        </div>
        
        <h2>学习目标</h2>
        <ul>
"""
            
            for obj in plan['learning_objectives']:
                html_content += f"            <li>{obj}</li>\n"
            
            html_content += """        </ul>
        
        <h2>推荐AI工具</h2>
"""
            
            for tool in plan['recommended_tools']:
                html_content += f"""        <div class="tool-box">
            <h3>{tool['name']}</h3>
            <p>{tool['description']}</p>
            <p><strong>功能特点：</strong></p>
            <ul>
"""
                
                for feature in tool['features']:
                    html_content += f"                <li>{feature}</li>\n"
                
                html_content += """            </ul>
        </div>
"""
            
            html_content += """        <h2>每周学习计划</h2>
"""
            
            for week_plan in plan['weekly_plans']:
                html_content += f"""        <div class="week-card">
            <h3>第{week_plan['week']}周: {week_plan['theme']}</h3>
"""
                
                for session in week_plan['sessions']:
                    html_content += f"""            <div class="session-card">
                <h4>课时{session['session']}: {session['activity_name']} ({session['duration']}分钟)</h4>
                <p><strong>专注点:</strong> {session['focus']}</p>
                <p>{session['activity_description']}</p>
                <p><strong>步骤:</strong></p>
                <ol>
"""
                    
                    for step in session['steps']:
                        html_content += f"                    <li>{step}</li>\n"
                    
                    html_content += """                </ol>
"""
                    
                    if session['tools']:
                        html_content += """                <p><strong>使用工具:</strong></p>
                <div>
"""
                        
                        for tool in session['tools']:
                            html_content += f'                    <span class="tool-tag">{tool["name"]}</span>\n'
                        
                        html_content += """                </div>
"""
                    
                    html_content += """            </div>
"""
                
                html_content += """        </div>
"""
            
            html_content += """        <h2>评估方法</h2>
        <ul>
"""
            
            for method in plan['assessment_methods']:
                html_content += f"            <li>{method}</li>\n"
            
            html_content += """        </ul>
    </div>
</body>
</html>
"""
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            return filename
        
        else:
            raise ValueError(f"不支持的格式: {format}，支持的格式有: json, markdown, html")
    
    def visualize_plan(self, plan_id: str, output_file: str = None) -> str:
        """可视化语言技能培养计划"""
        if plan_id not in self.activity_library:
            raise ValueError(f"未找到计划ID: {plan_id}")
        
        plan = self.activity_library[plan_id]
        
        # 为不同技能类型设置不同的主题颜色
        color_theme = {
            "听力": "#4285f4",  # Blue
            "口语": "#ea4335",  # Red
            "阅读": "#fbbc05",  # Yellow
            "写作": "#34a853"   # Green
        }
        
        theme_color = color_theme.get(plan['skill_type'], "#4285f4")
        
        plt.figure(figsize=(15, 10))
        
        # 1. 课程结构甘特图
        ax1 = plt.subplot(2, 1, 1)
        
        # 准备数据
        weeks = [f"第{i}周" for i in range(1, plan['duration_weeks'] + 1)]
        sessions_per_week = plan['sessions_per_week']
        
        # 为每周的每个课时创建一个条形
        for week_idx, week in enumerate(weeks):
            for session_idx in range(sessions_per_week):
                # 获取该课时的活动名
                week_plan = plan['weekly_plans'][week_idx]
                session = week_plan['sessions'][session_idx]
                activity_name = session['activity_name']
                
                # 计算条形的位置和大小
                left = week_idx
                height = 1 / sessions_per_week
                bottom = 1 - (session_idx + 1) * height
                
                # 绘制条形
                ax1.barh(bottom + height/2, 0.8, left=left, height=height*0.8, 
                        color=theme_color, alpha=0.7 - 0.2 * (session_idx % 2))
                
                # 添加活动名称
                ax1.text(left + 0.4, bottom + height/2, activity_name, 
                        ha='center', va='center', fontsize=8)
        
        # 设置Y轴（课时）
        session_labels = []
        for i in range(sessions_per_week):
            session_labels.extend([f"课时{i+1}"] * plan['duration_weeks'])
        
        # 隐藏Y轴刻度
        ax1.set_yticks([])
        
        # 设置X轴（周）
        ax1.set_xticks(range(len(weeks)))
        ax1.set_xticklabels(weeks)
        
        # 添加网格线分隔每周
        ax1.grid(axis='x', linestyle='-', alpha=0.3)
        
        # 设置标题和标签
        ax1.set_title(f"{plan['title']} - 课程结构")
        
        # 2. 学习活动分布饼图
        ax2 = plt.subplot(2, 2, 3)
        
        # 统计每种活动类型的数量
        activity_counts = {}
        for week_plan in plan['weekly_plans']:
            for session in week_plan['sessions']:
                activity_name = session['activity_name']
                if activity_name in activity_counts:
                    activity_counts[activity_name] += 1
                else:
                    activity_counts[activity_name] = 1
        
        # 绘制饼图
        activities = list(activity_counts.keys())
        counts = list(activity_counts.values())
        
        if activities:  # 确保有活动数据
            wedges, texts, autotexts = ax2.pie(
                counts, 
                labels=activities, 
                autopct='%1.1f%%',
                textprops={'fontsize': 8},
                colors=plt.cm.tab10.colors[:len(activities)]
            )
            
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')
            
            ax2.set_title("学习活动分布")
        else:
            ax2.text(0.5, 0.5, "无活动数据", ha='center', va='center')
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        # 3. 工具使用热力图
        ax3 = plt.subplot(2, 2, 4)
        
        # 统计每种工具在每周的使用次数
        tool_usage = {}
        for i, week_plan in enumerate(plan['weekly_plans']):
            week_num = week_plan['week']
            for session in week_plan['sessions']:
                for tool in session['tools']:
                    tool_name = tool['name']
                    if tool_name not in tool_usage:
                        tool_usage[tool_name] = [0] * plan['duration_weeks']
                    tool_usage[tool_name][week_num - 1] += 1
        
        # 准备热力图数据
        tools = list(tool_usage.keys())
        data = []
        for tool in tools:
            data.append(tool_usage[tool])
        
        if tools:  # 确保有工具数据
            # 创建热力图
            im = ax3.imshow(data, cmap='YlGnBu')
            
            # 设置坐标轴
            ax3.set_xticks(range(plan['duration_weeks']))
            ax3.set_xticklabels([f"第{i+1}周" for i in range(plan['duration_weeks'])])
            ax3.set_yticks(range(len(tools)))
            ax3.set_yticklabels(tools)
            
            # 在每个单元格中添加数值
            for i in range(len(tools)):
                for j in range(plan['duration_weeks']):
                    text = ax3.text(j, i, data[i][j], ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=ax3, label="使用次数")
            ax3.set_title("AI工具使用频率")
        else:
            ax3.text(0.5, 0.5, "无工具使用数据", ha='center', va='center')
            ax3.set_xticks([])
            ax3.set_yticks([])
        
        # 添加主标题
        plt.suptitle(f"{plan['focus_area']} - {plan['skill_type']}能力培养计划 (HSK{plan['hsk_level']})", fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为主标题留出空间
        
        # 保存图表
        if output_file:
            plt.savefig(output_file)
        else:
            output_file = f"{plan_id}_visualization.png"
            plt.savefig(output_file)
        
        plt.close()
        
        return output_file
    
    def compare_skills(self, output_file: str = None) -> str:
        """比较四项语言技能的AI辅助培养特点"""
        plt.figure(figsize=(15, 10))
        
        # 1. 各技能适用工具数量比较
        ax1 = plt.subplot(2, 2, 1)
        
        # 统计每种技能下的工具数量
        tool_counts = {skill: len(tools) for skill, tools in self.ai_tools.items()}
        
        # 绘制条形图
        skills = list(self.skill_types)
        counts = [tool_counts[skill] for skill in skills]
        
        # 为不同技能设置不同颜色
        colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']
        
        ax1.bar(skills, counts, color=colors)
        ax1.set_ylabel('工具数量')
        ax1.set_title('各技能AI辅助工具数量比较')
        
        # 添加数值标签
        for i, v in enumerate(counts):
            ax1.text(i, v + 0.1, str(v), ha='center')
        
        # 2. 各技能活动模板数量比较
        ax2 = plt.subplot(2, 2, 2)
        
        # 统计每种技能下的活动模板数量
        activity_counts = {skill: len(activities) for skill, activities in self.activity_templates.items()}
        
        # 绘制条形图
        skills = list(self.skill_types)
        counts = [activity_counts[skill] for skill in skills]
        
        ax2.bar(skills, counts, color=colors)
        ax2.set_ylabel('活动模板数量')
        ax2.set_title('各技能学习活动模板数量比较')
        
        # 添加数值标签
        for i, v in enumerate(counts):
            ax2.text(i, v + 0.1, str(v), ha='center')
        
        # 3. 各技能HSK级别适用性分析
        ax3 = plt.subplot(2, 1, 2)
        
        # 分析各技能在不同HSK级别的适用性
        hsk_applicability = {skill: [0] * 6 for skill in self.skill_types}
        
        for skill in self.skill_types:
            for tool in self.ai_tools[skill]:
                for level in tool['suitable_levels']:
                    hsk_applicability[skill][level - 1] += 1
        
        # 准备绘图数据
        x = range(1, 7)  # HSK级别1-6
        
        for i, skill in enumerate(self.skill_types):
            ax3.plot(x, hsk_applicability[skill], marker='o', label=skill, color=colors[i], linewidth=2)
        
        ax3.set_xlabel('HSK级别')
        ax3.set_ylabel('适用工具数量')
        ax3.set_xticks(x)
        ax3.set_title('各技能在不同HSK级别的工具适用性')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # 添加主标题
        plt.suptitle('语言四项技能AI辅助培养比较分析', fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为主标题留出空间
        
        # 保存图表
        if output_file:
            plt.savefig(output_file)
        else:
            output_file = f"language_skills_comparison_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            plt.savefig(output_file)
        
        plt.close()
        
        return output_file
    
    def list_plans(self) -> List[Dict]:
        """列出所有语言技能培养计划"""
        plan_list = []
        for plan_id, plan in self.activity_library.items():
            plan_list.append({
                "id": plan_id,
                "title": plan['title'],
                "skill_type": plan['skill_type'],
                "hsk_level": plan['hsk_level'],
                "focus_area": plan['focus_area'],
                "duration_weeks": plan['duration_weeks'],
                "created_at": plan['created_at']
            })
        
        return plan_list
    
    def save_library(self, filename: str = "language_skills_plans.json") -> str:
        """保存计划库"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.activity_library, f, ensure_ascii=False, indent=4)
        
        return filename
    
    def load_library(self, filename: str = "language_skills_plans.json") -> None:
        """加载计划库"""
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                self.activity_library = json.load(f)


def main():
    """主函数示例"""
    # 初始化系统
    skills_system = AILanguageSkillsSystem()
    
    # 1. 创建听力培养计划
    listening_plan = skills_system.create_skill_plan(
        title="商务交流听力能力培养",
        skill_type="听力",
        hsk_level=4,
        focus_area="商务交流",
        duration_weeks=4,
        sessions_per_week=2,
        minutes_per_session=45
    )
    
    # 获取计划ID
    listening_id = list(skills_system.activity_library.keys())[-1]
    
    # 导出为Markdown
    md_file = skills_system.export_plan(listening_id, "markdown")
    print(f"听力培养计划已导出为Markdown: {md_file}")
    
    # 可视化计划
    vis_file = skills_system.visualize_plan(listening_id)
    print(f"计划可视化已生成: {vis_file}")
    
    # 2. 创建口语培养计划
    speaking_plan = skills_system.create_skill_plan(
        title="日常社交口语能力培养",
        skill_type="口语",
        hsk_level=3,
        focus_area="日常社交",
        duration_weeks=6,
        sessions_per_week=2,
        minutes_per_session=30
    )
    
    # 获取计划ID
    speaking_id = list(skills_system.activity_library.keys())[-1]
    
    # 导出为HTML
    html_file = skills_system.export_plan(speaking_id, "html")
    print(f"口语培养计划已导出为HTML: {html_file}")
    
    # 3. 创建阅读培养计划
    reading_plan = skills_system.create_skill_plan(
        title="学术文献阅读能力培养",
        skill_type="阅读",
        hsk_level=5,
        focus_area="学术文献",
        duration_weeks=8,
        sessions_per_week=2,
        minutes_per_session=60
    )
    
    # 4. 创建写作培养计划
    writing_plan = skills_system.create_skill_plan(
        title="创意写作能力培养",
        skill_type="写作",
        hsk_level=5,
        focus_area="创意写作",
        duration_weeks=6,
        sessions_per_week=1,
        minutes_per_session=90
    )
    
    # 比较四项语言技能
    comparison_file = skills_system.compare_skills()
    print(f"语言技能比较图已生成: {comparison_file}")
    
    # 列出所有计划
    plans = skills_system.list_plans()
    print("\n已创建的语言技能培养计划:")
    for p in plans:
        print(f"- {p['title']} ({p['skill_type']}, HSK{p['hsk_level']}, ID: {p['id']})")
    
    # 推荐听力工具
    listening_tools = skills_system.suggest_tools("听力", 4, 2)
    print("\n推荐的HSK4级听力工具:")
    for tool in listening_tools:
        print(f"- {tool['name']}: {tool['description']}")
    
    # 推荐口语活动
    speaking_activities = skills_system.suggest_activities("口语", 3, 2)
    print("\n推荐的HSK3级口语活动:")
    for activity in speaking_activities:
        print(f"- {activity['name']}: {activity['description']}")
    
    # 保存计划库
    lib_file = skills_system.save_library()
    print(f"\n计划库已保存: {lib_file}")


if __name__ == "__main__":
    main()
