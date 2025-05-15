# AI辅助的积极学习策略系统
# 实现四种积极学习策略：AI辅助探究式学习、沉浸式文化体验、协作性问题解决和自适应微学习

import json
import os
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt

class AIActiveLearningSystems:
    """AI辅助的积极学习策略系统"""
    
    def __init__(self):
        """初始化AI辅助积极学习策略系统"""
        # 学习策略类型
        self.strategy_types = [
            "AI辅助探究式学习", 
            "沉浸式文化体验", 
            "协作性问题解决", 
            "自适应微学习"
        ]
        
        # 各策略的核心特点
        self.strategy_features = {
            "AI辅助探究式学习": [
                "以学生为中心的研究活动",
                "AI辅助资源检索与分析",
                "促进深度思考与创造性问题解决",
                "提问-探究-分析-创造-分享的流程",
                "教师作为引导者而非知识传授者",
                "发展信息素养与批判性思维"
            ],
            "沉浸式文化体验": [
                "虚拟情境中的语言与文化学习",
                "多感官交互式学习环境",
                "角色扮演与文化身份体验",
                "情境构建-角色扮演-互动体验-反思总结的流程",
                "文化内涵与语言表达的整合",
                "跨文化理解与比较视角"
            ],
            "协作性问题解决": [
                "小组协作完成真实性任务",
                "AI辅助团队协作与资源共享",
                "基于问题的情境化学习",
                "问题识别-团队组建-方案设计-实施评估的流程",
                "多角度观点与集体智慧整合",
                "社交学习与沟通能力发展"
            ],
            "自适应微学习": [
                "碎片化学习单元个性化推送",
                "基于学习数据的实时调整",
                "即时反馈与进度追踪",
                "碎片学习-即时反馈-进度追踪-知识连接的流程",
                "利用碎片时间的高效学习",
                "认知负荷优化与学习效率提升"
            ]
        }
        
        # 各策略的AI工具
        self.ai_tools = {
            "AI辅助探究式学习": [
                "智能搜索引擎",
                "文本分析工具",
                "数据可视化助手",
                "知识图谱构建器",
                "概念关系分析器",
                "多语言资源翻译器",
                "交互式问题生成器"
            ],
            "沉浸式文化体验": [
                "虚拟情境生成器",
                "角色对话系统",
                "文化背景模拟器",
                "情感反应分析器",
                "文化习俗指导员",
                "虚拟文化导览",
                "多媒体内容创建工具"
            ],
            "协作性问题解决": [
                "协作工作平台",
                "团队贡献分析器",
                "资源匹配推荐系统",
                "进度监控仪表盘",
                "方案评估分析器",
                "思维导图工具",
                "多角度问题分析器"
            ],
            "自适应微学习": [
                "个性化内容推送系统",
                "学习路径优化器",
                "记忆曲线跟踪器",
                "实时评估反馈工具",
                "微内容生成器",
                "学习习惯分析器",
                "知识连接可视化工具"
            ]
        }
        
        # 各策略的实施步骤
        self.implementation_steps = {
            "AI辅助探究式学习": [
                "教师设计开放性探究问题",
                "学生使用AI工具收集多样化信息",
                "学生在AI支持下整理和分析信息",
                "学生基于分析结果形成自己的见解",
                "学生分享探究成果并进行同伴评价",
                "教师引导反思探究过程和结果"
            ],
            "沉浸式文化体验": [
                "教师设计文化情境和学习目标",
                "AI系统生成逼真的文化场景",
                "学生在虚拟环境中扮演特定角色",
                "学生与AI生成的虚拟角色进行互动",
                "学生体验文化习俗和交际规则",
                "学生分享体验感受并进行文化反思"
            ],
            "协作性问题解决": [
                "教师提出或引导选择真实问题",
                "AI系统协助组建互补性学习小组",
                "小组成员分配角色和任务",
                "学生利用AI辅助进行信息收集和方案设计",
                "小组将设计的方案付诸实践",
                "学生评估方案效果和团队协作过程"
            ],
            "自适应微学习": [
                "AI系统分析学习者水平和需求",
                "系统推送个性化微学习内容",
                "学习者完成微型练习并获得即时反馈",
                "系统记录学习表现并调整推送内容",
                "系统生成可视化学习进度地图",
                "系统帮助学习者连接碎片化知识"
            ]
        }
        
        # 各策略的评估方法
        self.assessment_methods = {
            "AI辅助探究式学习": [
                "探究报告评估",
                "知识构建过程分析",
                "问题解决能力评价",
                "信息素养表现评估",
                "创造性思维评价",
                "同伴互评与反馈"
            ],
            "沉浸式文化体验": [
                "文化理解测验",
                "角色扮演表现评估",
                "文化情境应对能力",
                "跨文化反思日志",
                "语言文化整合能力",
                "虚拟情境互动分析"
            ],
            "协作性问题解决": [
                "解决方案质量评估",
                "团队协作过程分析",
                "个人贡献评价",
                "方案实施效果评估",
                "问题分析深度评价",
                "团队反思报告"
            ],
            "自适应微学习": [
                "学习进度追踪",
                "知识掌握水平测试",
                "学习习惯分析",
                "长期记忆保持率",
                "知识应用能力评估",
                "学习效率分析"
            ]
        }
        
        # 各策略的适用场景
        self.suitable_scenarios = {
            "AI辅助探究式学习": [
                "高级汉语阅读理解课",
                "中国历史文化研究专题",
                "汉语语言特点分析课",
                "中文文学作品赏析",
                "新闻媒体与社会议题",
                "中国传统艺术研究"
            ],
            "沉浸式文化体验": [
                "中国传统节日体验课",
                "地域文化特色学习",
                "日常生活场景会话练习",
                "中国历史事件再现",
                "商务礼仪与交际文化",
                "民族习俗与文化差异"
            ],
            "协作性问题解决": [
                "文化活动策划项目",
                "中文内容创作任务",
                "跨文化交流活动设计",
                "语言学习资源开发",
                "社区服务学习项目",
                "中文媒体制作合作"
            ],
            "自适应微学习": [
                "汉字记忆与书写练习",
                "词汇量建设与巩固",
                "语法点逐步掌握",
                "发音与声调训练",
                "常用表达积累",
                "碎片时间学习需求"
            ]
        }
        
        # 学习策略项目库
        self.project_library = {}
    
    def suggest_strategy(self, 
                        learning_goal: str, 
                        student_level: str,
                        student_count: int,
                        available_time: int) -> str:
        """推荐适合的学习策略"""
        # 基于学习目标、学生水平、人数和可用时间推荐策略
        
        # 策略倾向性评分
        strategy_scores = {
            "AI辅助探究式学习": 0,
            "沉浸式文化体验": 0,
            "协作性问题解决": 0,
            "自适应微学习": 0
        }
        
        # 基于学习目标评分
        culture_keywords = ["文化", "习俗", "传统", "历史", "艺术", "价值观", "民族", "社会"]
        research_keywords = ["研究", "分析", "探索", "理解", "比较", "评价", "思考", "知识"]
        collaboration_keywords = ["合作", "项目", "解决", "设计", "创造", "开发", "团队", "共同"]
        skills_keywords = ["技能", "掌握", "练习", "记忆", "应用", "巩固", "提高", "强化"]
        
        for keyword in culture_keywords:
            if keyword in learning_goal:
                strategy_scores["沉浸式文化体验"] += 2
        
        for keyword in research_keywords:
            if keyword in learning_goal:
                strategy_scores["AI辅助探究式学习"] += 2
        
        for keyword in collaboration_keywords:
            if keyword in learning_goal:
                strategy_scores["协作性问题解决"] += 2
        
        for keyword in skills_keywords:
            if keyword in learning_goal:
                strategy_scores["自适应微学习"] += 2
        
        # 基于学生水平评分
        if student_level in ["初级", "HSK1", "HSK2"]:
            strategy_scores["自适应微学习"] += 3
            strategy_scores["沉浸式文化体验"] += 1
        elif student_level in ["中级", "HSK3", "HSK4"]:
            strategy_scores["沉浸式文化体验"] += 2
            strategy_scores["协作性问题解决"] += 2
            strategy_scores["自适应微学习"] += 1
        else:  # 高级
            strategy_scores["AI辅助探究式学习"] += 3
            strategy_scores["协作性问题解决"] += 2
        
        # 基于学生人数评分
        if student_count <= 1:
            strategy_scores["自适应微学习"] += 3
            strategy_scores["AI辅助探究式学习"] += 1
        elif student_count <= 5:
            strategy_scores["协作性问题解决"] += 3
            strategy_scores["AI辅助探究式学习"] += 2
        else:  # 大班教学
            strategy_scores["沉浸式文化体验"] += 2
            strategy_scores["自适应微学习"] += 2
        
        # 基于可用时间评分
        if available_time <= 20:
            strategy_scores["自适应微学习"] += 3
        elif available_time <= 45:
            strategy_scores["沉浸式文化体验"] += 2
            strategy_scores["协作性问题解决"] += 1
        else:  # 长时间课程
            strategy_scores["AI辅助探究式学习"] += 3
            strategy_scores["协作性问题解决"] += 2
        
        # 找出得分最高的策略
        recommended_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        return recommended_strategy
    
    def create_project(self, 
                      title: str,
                      strategy_type: str,
                      topic: str,
                      student_level: str,
                      duration: int,
                      learning_objectives: List[str]) -> Dict:
        """创建学习策略项目"""
        
        if strategy_type not in self.strategy_types:
            raise ValueError(f"策略类型必须是以下之一: {', '.join(self.strategy_types)}")
        
        # 选择适合的AI工具
        selected_tools = random.sample(self.ai_tools[strategy_type], 3)
        
        # 选择适合的评估方法
        selected_assessments = random.sample(self.assessment_methods[strategy_type], 2)
        
        # 创建项目活动流程
        activities = []
        steps = self.implementation_steps[strategy_type]
        
        # 分配时间（预留10%用于开始和总结）
        available_time = int(duration * 0.9)
        per_step_time = available_time // len(steps)
        remaining_time = available_time - per_step_time * len(steps)
        
        # 分配开始和总结时间
        intro_time = int(duration * 0.05)
        conclusion_time = duration - available_time - intro_time
        
        # 添加介绍活动
        activities.append({
            "name": "项目介绍与目标设定",
            "description": f"教师介绍{topic}探究项目，明确学习目标和活动流程。",
            "duration": intro_time,
            "tools": []
        })
        
        # 添加主要活动步骤
        for i, step in enumerate(steps):
            # 为关键步骤分配更多时间
            if i == 1 or i == 3:  # 假设第2和第4步是关键步骤
                step_duration = per_step_time + (remaining_time // 2)
                remaining_time -= (remaining_time // 2)
            else:
                step_duration = per_step_time
            
            # 选择该步骤使用的工具
            step_tools = []
            if i < len(selected_tools):
                step_tools = [selected_tools[i]]
            
            activities.append({
                "name": f"步骤{i+1}: {step}",
                "description": step,
                "duration": step_duration,
                "tools": step_tools
            })
        
        # 添加总结活动
        activities.append({
            "name": "项目总结与反思",
            "description": f"学生总结学习成果，分享收获，教师引导反思学习过程。",
            "duration": conclusion_time,
            "tools": []
        })
        
        # 创建项目
        project = {
            "title": title,
            "strategy_type": strategy_type,
            "topic": topic,
            "student_level": student_level,
            "duration": duration,
            "learning_objectives": learning_objectives,
            "ai_tools": selected_tools,
            "assessment_methods": selected_assessments,
            "activities": activities,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存到项目库
        project_id = f"{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.project_library[project_id] = project
        
        return project
    
    def export_project(self, project_id: str, format: str = "json") -> str:
        """导出学习项目"""
        if project_id not in self.project_library:
            raise ValueError(f"未找到项目ID: {project_id}")
        
        project = self.project_library[project_id]
        
        if format.lower() == "json":
            filename = f"{project_id}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(project, f, ensure_ascii=False, indent=4)
            return filename
        
        elif format.lower() == "markdown":
            filename = f"{project_id}.md"
            
            md_content = f"# {project['title']}\n\n"
            md_content += f"* 策略类型: {project['strategy_type']}\n"
            md_content += f"* 主题: {project['topic']}\n"
            md_content += f"* 学生水平: {project['student_level']}\n"
            md_content += f"* 时长: {project['duration']}分钟\n"
            md_content += f"* 创建时间: {project['created_at']}\n\n"
            
            md_content += "## 学习目标\n\n"
            for obj in project['learning_objectives']:
                md_content += f"* {obj}\n"
            md_content += "\n"
            
            md_content += "## AI工具\n\n"
            for tool in project['ai_tools']:
                md_content += f"* {tool}\n"
            md_content += "\n"
            
            md_content += "## 评估方法\n\n"
            for method in project['assessment_methods']:
                md_content += f"* {method}\n"
            md_content += "\n"
            
            md_content += "## 活动流程\n\n"
            for i, activity in enumerate(project['activities']):
                md_content += f"### {activity['name']} ({activity['duration']}分钟)\n\n"
                md_content += f"* 描述: {activity['description']}\n"
                if activity['tools']:
                    md_content += "* 使用工具:\n"
                    for tool in activity['tools']:
                        md_content += f"  - {tool}\n"
                md_content += "\n"
            
            md_content += "## 策略特点\n\n"
            for feature in self.strategy_features[project['strategy_type']]:
                md_content += f"* {feature}\n"
            md_content += "\n"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            return filename
        
        elif format.lower() == "html":
            filename = f"{project_id}.html"
            
            # 为不同策略类型设置不同的主题颜色
            color_theme = {
                "AI辅助探究式学习": "#4285f4",  # Google Blue
                "沉浸式文化体验": "#ea4335",  # Google Red
                "协作性问题解决": "#fbbc05",  # Google Yellow
                "自适应微学习": "#34a853"   # Google Green
            }
            
            theme_color = color_theme.get(project['strategy_type'], "#4285f4")
            
            html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project['title']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; color: #333; line-height: 1.6; }}
        h1, h2, h3 {{ color: {theme_color}; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .header {{ background-color: {theme_color}; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .info-box {{ background-color: #f8f9fa; border-left: 4px solid {theme_color}; padding: 15px; margin-bottom: 20px; }}
        .activity {{ background-color: #f1f3f4; border-radius: 5px; padding: 15px; margin-bottom: 15px; position: relative; }}
        .activity-duration {{ position: absolute; top: 15px; right: 15px; background-color: {theme_color}; color: white; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; }}
        .tool-tag {{ background-color: #e8f0fe; color: {theme_color}; padding: 3px 8px; border-radius: 3px; margin-right: 5px; font-size: 0.9em; display: inline-block; margin-bottom: 5px; }}
        .features {{ background-color: #e8f0fe; padding: 15px; border-radius: 5px; }}
        ul {{ padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{project['title']}</h1>
            <p><strong>策略类型:</strong> {project['strategy_type']}</p>
        </div>
        
        <div class="info-box">
            <p><strong>主题:</strong> {project['topic']}</p>
            <p><strong>学生水平:</strong> {project['student_level']}</p>
            <p><strong>时长:</strong> {project['duration']}分钟</p>
            <p><strong>创建时间:</strong> {project['created_at']}</p>
        </div>
        
        <h2>学习目标</h2>
        <ul>
"""
            
            for obj in project['learning_objectives']:
                html_content += f"            <li>{obj}</li>\n"
            
            html_content += """        </ul>
        
        <h2>AI工具</h2>
        <div>
"""
            
            for tool in project['ai_tools']:
                html_content += f'            <span class="tool-tag">{tool}</span>\n'
            
            html_content += """        </div>
        <br>
        
        <h2>评估方法</h2>
        <ul>
"""
            
            for method in project['assessment_methods']:
                html_content += f"            <li>{method}</li>\n"
            
            html_content += """        </ul>
        
        <h2>活动流程</h2>
"""
            
            for activity in project['activities']:
                html_content += f"""        <div class="activity">
            <div class="activity-duration">{activity['duration']}分钟</div>
            <h3>{activity['name']}</h3>
            <p>{activity['description']}</p>
"""
                
                if activity['tools']:
                    html_content += "            <div>\n"
                    for tool in activity['tools']:
                        html_content += f'                <span class="tool-tag">{tool}</span>\n'
                    html_content += "            </div>\n"
                
                html_content += "        </div>\n"
            
            html_content += f"""        <h2>策略特点</h2>
        <div class="features">
            <ul>
"""
            
            for feature in self.strategy_features[project['strategy_type']]:
                html_content += f"                <li>{feature}</li>\n"
            
            html_content += """            </ul>
        </div>
    </div>
</body>
</html>
"""
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            return filename
        
        else:
            raise ValueError(f"不支持的格式: {format}，支持的格式有: json, markdown, html")
    
    def visualize_project(self, project_id: str, output_file: str = None) -> str:
        """可视化学习项目"""
        if project_id not in self.project_library:
            raise ValueError(f"未找到项目ID: {project_id}")
        
        project = self.project_library[project_id]
        
        # 为不同策略类型设置不同的主题颜色
        color_theme = {
            "AI辅助探究式学习": "#4285f4",  # Google Blue
            "沉浸式文化体验": "#ea4335",  # Google Red
            "协作性问题解决": "#fbbc05",  # Google Yellow
            "自适应微学习": "#34a853"   # Google Green
        }
        
        theme_color = color_theme.get(project['strategy_type'], "#4285f4")
        
        plt.figure(figsize=(15, 10))
        
        # 1. 时间线图：活动流程
        ax1 = plt.subplot(2, 1, 1)
        
        # 准备时间线数据
        activities = project['activities']
        activity_names = [act['name'].replace("步骤", "步骤\n") for act in activities]
        start_times = [0]
        for i in range(1, len(activities)):
            start_times.append(start_times[i-1] + activities[i-1]['duration'])
        durations = [act['duration'] for act in activities]
        
        # 绘制甘特图样式的时间线
        for i, (name, start, duration) in enumerate(zip(activity_names, start_times, durations)):
            ax1.barh(i, duration, left=start, height=0.5, color=theme_color, alpha=0.8, edgecolor='black')
            # 在条形中间添加文本标签
            text_color = 'white' if 'project_id' in name else 'black'
            ax1.text(start + duration/2, i, name, ha='center', va='center', fontsize=8, color=text_color)
        
        # 设置y轴标签和刻度
        ax1.set_yticks(range(len(activity_names)))
        ax1.set_yticklabels(activity_names)
        
        # 设置x轴为时间刻度
        ax1.set_xticks(range(0, project['duration']+1, 5))
        ax1.set_xlabel('时间 (分钟)')
        
        ax1.set_title('活动流程时间线')
        ax1.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 2. 工具使用与活动关系
        ax2 = plt.subplot(2, 2, 3)
        
        # 准备数据
        all_tools = []
        for activity in activities:
            for tool in activity['tools']:
                if tool not in all_tools:
                    all_tools.append(tool)
        
        if all_tools:  # 只有当有工具时才绘制
            # 创建工具-活动矩阵
            activity_indices = list(range(len(activities)))
            tool_activity_matrix = []
            
            for tool in all_tools:
                row = []
                for activity in activities:
                    if tool in activity['tools']:
                        row.append(1)
                    else:
                        row.append(0)
                tool_activity_matrix.append(row)
            
            # 绘制热图
            ax2.imshow(tool_activity_matrix, cmap='YlGnBu', aspect='auto')
            
            # 设置标签
            ax2.set_yticks(range(len(all_tools)))
            ax2.set_yticklabels(all_tools)
            
            # 设置x轴标签（活动名称）
            short_names = [f"活动{i+1}" for i in range(len(activities))]
            ax2.set_xticks(range(len(activities)))
            ax2.set_xticklabels(short_names, rotation=45, ha='right')
            
            # 在热图上添加数值标签
            for i in range(len(all_tools)):
                for j in range(len(activities)):
                    text = "●" if tool_activity_matrix[i][j] == 1 else ""
                    ax2.text(j, i, text, ha='center', va='center')
            
            ax2.set_title('工具在活动中的使用')
        else:
            ax2.text(0.5, 0.5, "无工具使用数据", ha='center', va='center')
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title('工具在活动中的使用')
        
        # 3. 策略特点可视化
        ax3 = plt.subplot(2, 2, 4)
        
        # 准备数据
        features = self.strategy_features[project['strategy_type']]
        
        if features:  # 只有当有特点时才绘制
            # 绘制水平条形图
            y_pos = range(len(features))
            ax3.barh(y_pos, [1] * len(features), color=theme_color, alpha=0.6)
            
            # 设置标签
            ax3.set_yticks(y_pos)
            # 处理长文本，确保可读性
            wrapped_features = ['\n'.join([feature[i:i+30] for i in range(0, len(feature), 30)]) for feature in features]
            ax3.set_yticklabels(wrapped_features)
            
            # 移除x轴刻度
            ax3.set_xticks([])
            
            ax3.set_title('策略核心特点')
        else:
            ax3.text(0.5, 0.5, "无策略特点数据", ha='center', va='center')
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_title('策略核心特点')
        
        # 添加主标题
        plt.suptitle(project['title'], fontsize=16, color=theme_color, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为主标题留出空间
        
        # 保存图表
        if output_file:
            plt.savefig(output_file)
        else:
            output_file = f"{project_id}_visualization.png"
            plt.savefig(output_file)
        
        plt.close()
        
        return output_file
    
    def compare_strategies(self, output_file: str = None) -> str:
        """比较不同的学习策略"""
        plt.figure(figsize=(15, 10))
        
        # 获取策略特点数据
        strategy_features_count = {strategy: len(features) for strategy, features in self.strategy_features.items()}
        
        # 1. 策略特点数量比较
        ax1 = plt.subplot(2, 2, 1)
        
        # 绘制条形图
        strategies = list(self.strategy_types)
        feature_counts = [strategy_features_count[s] for s in strategies]
        
        # 为不同策略设置不同颜色
        colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']
        
        ax1.bar(strategies, feature_counts, color=colors)
        ax1.set_ylabel('特点数量')
        ax1.set_title('各策略特点数量比较')
        
        # 旋转x轴标签以便更好地显示
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
        
        # 2. 各策略实施步骤数量比较
        ax2 = plt.subplot(2, 2, 2)
        
        # 获取步骤数量数据
        step_counts = {strategy: len(steps) for strategy, steps in self.implementation_steps.items()}
        strategies = list(self.strategy_types)
        steps = [step_counts[s] for s in strategies]
        
        ax2.bar(strategies, steps, color=colors)
        ax2.set_ylabel('步骤数量')
        ax2.set_title('各策略实施步骤数量比较')
        
        # 旋转x轴标签
        plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
        
        # 3. 策略特点文字云比较
        ax3 = plt.subplot(2, 1, 2)
        
        # 合并所有策略特点
        all_features = []
        for strategy, features in self.strategy_features.items():
            for feature in features:
                all_features.append((strategy, feature))
        
        # 为每个策略分配y位置
        strategy_positions = {
            "AI辅助探究式学习": 4,
            "沉浸式文化体验": 3,
            "协作性问题解决": 2,
            "自适应微学习": 1
        }
        
        # 为每个特点分配x位置
        max_features = max(len(features) for features in self.strategy_features.values())
        
        # 绘制特点文本
        for strategy, feature in all_features:
            # 找出该特点在该策略中的索引
            feature_index = self.strategy_features[strategy].index(feature)
            
            # 计算x位置（均匀分布特点）
            strategy_feature_count = len(self.strategy_features[strategy])
            x_pos = (feature_index + 1) * (max_features / (strategy_feature_count + 1))
            
            # 获取y位置
            y_pos = strategy_positions[strategy]
            
            # 确定颜色
            color_index = list(strategy_positions.keys()).index(strategy)
            color = colors[color_index]
            
            # 添加特点文本
            # 截断长文本以提高可读性
            short_feature = feature[:40] + '...' if len(feature) > 40 else feature
            ax3.text(x_pos, y_pos, short_feature, ha='center', va='center', 
                    fontsize=8, color=color, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # 设置坐标轴
        ax3.set_xlim(0, max_features + 1)
        ax3.set_ylim(0.5, 4.5)
        
        # 添加策略标签
        for strategy, position in strategy_positions.items():
            ax3.text(0, position, strategy, ha='right', va='center', fontsize=10,
                    weight='bold', color=colors[list(strategy_positions.keys()).index(strategy)])
        
        # 隐藏坐标轴
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        
        ax3.set_title('各策略关键特点比较')
        
        plt.tight_layout()
        
        # 保存图表
        if output_file:
            plt.savefig(output_file)
        else:
            output_file = f"strategy_comparison_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            plt.savefig(output_file)
        
        plt.close()
        
        return output_file
    
    def list_projects(self) -> List[Dict]:
        """列出所有学习项目"""
        project_list = []
        for project_id, project in self.project_library.items():
            project_list.append({
                "id": project_id,
                "title": project['title'],
                "strategy_type": project['strategy_type'],
                "topic": project['topic'],
                "student_level": project['student_level'],
                "duration": project['duration'],
                "created_at": project['created_at']
            })
        
        return project_list
    
    def save_library(self, filename: str = "active_learning_projects.json") -> str:
        """保存项目库"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.project_library, f, ensure_ascii=False, indent=4)
        
        return filename
    
    def load_library(self, filename: str = "active_learning_projects.json") -> None:
        """加载项目库"""
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                self.project_library = json.load(f)


def main():
    """主函数示例"""
    # 初始化系统
    learning_system = AIActiveLearningSystems()
    
    # 1. 创建探究式学习项目
    inquiry_project = learning_system.create_project(
        title="中国方言地理分布探究",
        strategy_type="AI辅助探究式学习",
        topic="中国方言",
        student_level="高级",
        duration=90,
        learning_objectives=[
            "学生能够识别中国主要方言区的地理分布",
            "学生能够分析方言形成的历史和地理因素",
            "学生能够比较不同方言的语音和词汇特点",
            "学生能够运用AI工具进行语言资料的收集和分析"
        ]
    )
    
    # 获取项目ID
    inquiry_id = list(learning_system.project_library.keys())[-1]
    
    # 导出为Markdown
    md_file = learning_system.export_project(inquiry_id, "markdown")
    print(f"探究式学习项目已导出为Markdown: {md_file}")
    
    # 可视化项目
    vis_file = learning_system.visualize_project(inquiry_id)
    print(f"项目可视化已生成: {vis_file}")
    
    # 2. 创建沉浸式文化体验项目
    immersive_project = learning_system.create_project(
        title="春节庆祝虚拟体验",
        strategy_type="沉浸式文化体验",
        topic="中国春节",
        student_level="中级",
        duration=60,
        learning_objectives=[
            "学生能够识别和使用与春节相关的词汇和表达",
            "学生能够体验春节庆祝的主要活动和习俗",
            "学生能够理解春节背后的文化内涵和价值观",
            "学生能够比较中国春节与自己文化中的节日"
        ]
    )
    
    # 获取项目ID
    immersive_id = list(learning_system.project_library.keys())[-1]
    
    # 导出为HTML
    html_file = learning_system.export_project(immersive_id, "html")
    print(f"沉浸式文化体验项目已导出为HTML: {html_file}")
    
    # 3. 创建协作性问题解决项目
    collab_project = learning_system.create_project(
        title="中文文化活动策划",
        strategy_type="协作性问题解决",
        topic="文化活动策划",
        student_level="中高级",
        duration=120,
        learning_objectives=[
            "学生能够运用中文进行团队协作和任务分配",
            "学生能够设计符合中国文化特色的活动方案",
            "学生能够评估活动方案的可行性和文化适当性",
            "学生能够用中文展示和说明活动策划成果"
        ]
    )
    
    # 4. 创建自适应微学习项目
    micro_project = learning_system.create_project(
        title="HSK4级词汇强化训练",
        strategy_type="自适应微学习",
        topic="HSK4词汇",
        student_level="中级",
        duration=30,
        learning_objectives=[
            "学生能够记忆并正确使用HSK4级核心词汇",
            "学生能够根据个人记忆特点优化学习策略",
            "学生能够在真实语境中识别和应用目标词汇",
            "学生能够建立词汇间的语义网络和联系"
        ]
    )
    
    # 比较不同学习策略
    comparison_file = learning_system.compare_strategies()
    print(f"学习策略比较图已生成: {comparison_file}")
    
    # 列出所有项目
    projects = learning_system.list_projects()
    print("\n已创建的学习项目:")
    for p in projects:
        print(f"- {p['title']} ({p['strategy_type']}, ID: {p['id']})")
    
    # 推荐学习策略
    recommended = learning_system.suggest_strategy(
        learning_goal="提高学生对中国传统节日的文化理解和交际能力",
        student_level="中级",
        student_count=15,
        available_time=60
    )
    print(f"\n基于学习需求推荐的策略: {recommended}")
    
    # 保存项目库
    lib_file = learning_system.save_library()
    print(f"\n项目库已保存: {lib_file}")


if __name__ == "__main__":
    main()
