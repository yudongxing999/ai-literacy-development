# 教师AI素养评估系统
# 基于"AI基础认知、AI工具应用、AI教学设计和AI伦理与批判"四维度评价框架

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from typing import Dict, List, Any, Optional

class TeacherAILiteracyAssessment:
    """教师AI素养评估系统"""
    
    def __init__(self):
        """初始化教师AI素养评估系统"""
        self.dimensions = ["AI基础认知", "AI工具应用", "AI教学设计", "AI伦理与批判"]
        
        # 维度下的指标和评分标准
        self.indicators = {
            "AI基础认知": [
                "AI基础知识理解",
                "AI教育应用认识",
                "AI技术发展趋势把握"
            ],
            "AI工具应用": [
                "通用AI工具使用",
                "语言教学专用AI工具应用",
                "AI工具环境配置"
            ],
            "AI教学设计": [
                "AI增强的教学活动设计",
                "个性化学习路径创建",
                "数据驱动教学实施"
            ],
            "AI伦理与批判": [
                "AI伦理问题认识",
                "数据隐私与安全保护",
                "算法偏见与公平性审视"
            ]
        }
        
        # 三级发展水平
        self.levels = ["初级(获取)", "中级(深化)", "高级(创造)"]
        
        # 发展水平对应的评分范围
        self.level_scores = {
            "初级(获取)": (1, 3),
            "中级(深化)": (4, 7),
            "高级(创造)": (8, 10)
        }
        
        # 评分标准详细说明
        self.rubrics = self._load_rubrics()
        
        # 用于存储评估结果
        self.assessment_results = {}
    
    def _load_rubrics(self) -> Dict:
        """加载评估标准"""
        # 实际应用中，这里可以从配置文件加载
        rubrics = {}
        
        # AI基础认知维度的评分标准
        rubrics["AI基础知识理解"] = {
            "初级(获取)": "了解AI基本概念和类型，能够解释机器学习、深度学习等基础概念",
            "中级(深化)": "理解AI技术原理和局限性，能分析不同AI技术的适用范围",
            "高级(创造)": "掌握AI系统运作机制，能对AI教育系统进行全面评估"
        }
        
        rubrics["AI教育应用认识"] = {
            "初级(获取)": "认识常见AI教育应用类型，了解其基本功能",
            "中级(深化)": "分析AI应用在教育中的价值，能评估应用的教学适用性",
            "高级(创造)": "评估AI在教育中的变革潜力，能预见技术发展趋势"
        }
        
        rubrics["AI技术发展趋势把握"] = {
            "初级(获取)": "关注AI教育案例和新闻，了解新技术的教育应用",
            "中级(深化)": "跟踪AI研究发展趋势，理解技术进步对教育的影响",
            "高级(创造)": "参与AI教育对话和创新，能对AI发展方向提出见解"
        }
        
        # AI工具应用维度的评分标准
        rubrics["通用AI工具使用"] = {
            "初级(获取)": "能使用基础AI工具，如大语言模型、翻译工具等",
            "中级(深化)": "熟练操作多种AI工具，能根据需求选择适当工具",
            "高级(创造)": "能定制AI工具功能，优化工具在教学中的应用"
        }
        
        rubrics["语言教学专用AI工具应用"] = {
            "初级(获取)": "能应用预设的AI语言教学模板和资源",
            "中级(深化)": "能整合多种AI工具组合应用于语言教学",
            "高级(创造)": "能开发AI应用流程，创建专业化的语言教学工具链"
        }
        
        rubrics["AI工具环境配置"] = {
            "初级(获取)": "能尝试简单的AI提示工程，使用基本提示词",
            "中级(深化)": "能优化提示获取更好的结果，调整参数提高输出质量",
            "高级(创造)": "能创新提示工程方法，开发高级提示策略和模板"
        }
        
        # AI教学设计维度的评分标准
        rubrics["AI增强的教学活动设计"] = {
            "初级(获取)": "能在既有教学中融入简单的AI工具和资源",
            "中级(深化)": "能设计AI增强的教学活动，优化教学流程",
            "高级(创造)": "能创新AI教学模式，开发原创的教学方法"
        }
        
        rubrics["个性化学习路径创建"] = {
            "初级(获取)": "能应用AI生成的个性化教学内容和资源",
            "中级(深化)": "能创建基于AI的个性化学习路径和教学策略",
            "高级(创造)": "能构建自适应学习生态系统，实现深度个性化"
        }
        
        rubrics["数据驱动教学实施"] = {
            "初级(获取)": "能使用AI辅助评估工具分析学习数据",
            "中级(深化)": "能实施数据驱动教学，基于分析结果调整教学",
            "高级(创造)": "能领导AI教学变革，建立数据驱动的教学体系"
        }
        
        # AI伦理与批判维度的评分标准
        rubrics["AI伦理问题认识"] = {
            "初级(获取)": "认识基本AI伦理问题，了解主要争议",
            "中级(深化)": "能分析AI应用的伦理影响，识别潜在风险",
            "高级(创造)": "能参与AI伦理规范制定，引导负责任使用"
        }
        
        rubrics["数据隐私与安全保护"] = {
            "初级(获取)": "了解数据隐私的重要性，认识基本保护措施",
            "中级(深化)": "能保护学习者数据安全，实施隐私保护策略",
            "高级(创造)": "能引导学生和同事负责任地使用AI，建立数据保护机制"
        }
        
        rubrics["算法偏见与公平性审视"] = {
            "初级(获取)": "遵循AI使用规范，注意可能的偏见问题",
            "中级(深化)": "能审视算法偏见，识别AI系统中的不公平现象",
            "高级(创造)": "能批判性评估AI系统，纠正算法偏见，确保公平性"
        }
        
        return rubrics
    
    def create_assessment_form(self, output_file: str = "教师AI素养评估表.csv") -> str:
        """生成评估表格"""
        rows = []
        
        for dimension in self.dimensions:
            for indicator in self.indicators[dimension]:
                for level in self.levels:
                    description = self.rubrics[indicator][level]
                    rows.append({
                        "维度": dimension,
                        "指标": indicator,
                        "发展水平": level,
                        "描述": description,
                        "评分(1-10)": ""
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        
        return output_file
    
    def process_assessment(self, assessment_file: str, teacher_id: str) -> Dict:
        """处理评估表格，计算结果"""
        df = pd.read_csv(assessment_file, encoding="utf-8-sig")
        
        # 确保评分列已填写
        if df["评分(1-10)"].isna().any():
            raise ValueError("评分表未完全填写，请确保所有指标都已评分")
        
        # 转换评分为数值型
        df["评分(1-10)"] = df["评分(1-10)"].astype(int)
        
        # 验证评分范围
        if (df["评分(1-10)"] < 1).any() or (df["评分(1-10)"] > 10).any():
            raise ValueError("评分范围应为1-10")
        
        # 计算各维度得分
        dimension_scores = {}
        for dimension in self.dimensions:
            dimension_data = df[df["维度"] == dimension]
            dimension_scores[dimension] = dimension_data["评分(1-10)"].mean()
        
        # 计算各指标得分
        indicator_scores = {}
        for dimension in self.dimensions:
            for indicator in self.indicators[dimension]:
                indicator_data = df[df["指标"] == indicator]
                indicator_scores[indicator] = indicator_data["评分(1-10)"].mean()
        
        # 确定各维度的发展水平
        dimension_levels = {}
        for dimension, score in dimension_scores.items():
            if score <= self.level_scores["初级(获取)"][1]:
                dimension_levels[dimension] = "初级(获取)"
            elif score <= self.level_scores["中级(深化)"][1]:
                dimension_levels[dimension] = "中级(深化)"
            else:
                dimension_levels[dimension] = "高级(创造)"
        
        # 确定整体发展水平
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        if overall_score <= self.level_scores["初级(获取)"][1]:
            overall_level = "初级(获取)"
        elif overall_score <= self.level_scores["中级(深化)"][1]:
            overall_level = "中级(深化)"
        else:
            overall_level = "高级(创造)"
        
        # 生成评估结果
        assessment_result = {
            "teacher_id": teacher_id,
            "assessment_date": datetime.now().strftime("%Y-%m-%d"),
            "dimension_scores": dimension_scores,
            "indicator_scores": indicator_scores,
            "dimension_levels": dimension_levels,
            "overall_score": overall_score,
            "overall_level": overall_level
        }
        
        # 保存评估结果
        self.assessment_results[teacher_id] = assessment_result
        
        return assessment_result
    
    def generate_report(self, teacher_id: str, output_file: str = None) -> str:
        """生成评估报告"""
        if teacher_id not in self.assessment_results:
            raise ValueError(f"未找到教师 {teacher_id} 的评估结果")
        
        result = self.assessment_results[teacher_id]
        
        # 生成报告内容
        report = f"# 教师AI素养评估报告\n\n"
        report += f"**教师ID**: {teacher_id}\n"
        report += f"**评估日期**: {result['assessment_date']}\n"
        report += f"**整体评分**: {result['overall_score']:.2f}/10\n"
        report += f"**整体水平**: {result['overall_level']}\n\n"
        
        report += "## 维度评分\n\n"
        for dimension, score in result['dimension_scores'].items():
            report += f"- **{dimension}**: {score:.2f}/10 ({result['dimension_levels'][dimension]})\n"
        
        report += "\n## 指标详情\n\n"
        for dimension in self.dimensions:
            report += f"### {dimension}\n\n"
            for indicator in self.indicators[dimension]:
                score = result['indicator_scores'][indicator]
                level = "初级(获取)" if score <= 3 else ("中级(深化)" if score <= 7 else "高级(创造)")
                report += f"- **{indicator}**: {score:.2f}/10 ({level})\n"
                report += f"  - {self.rubrics[indicator][level]}\n"
            report += "\n"
        
        report += "## 发展建议\n\n"
        # 基于评估结果生成针对性建议
        for dimension in self.dimensions:
            level = result['dimension_levels'][dimension]
            report += f"### {dimension}\n\n"
            
            if level == "初级(获取)":
                if dimension == "AI基础认知":
                    report += "- 参加AI基础知识培训课程，了解AI基本概念和原理\n"
                    report += "- 定期阅读AI教育应用案例，积累实践经验\n"
                    report += "- 关注AI教育领域的新闻和发展动态\n"
                elif dimension == "AI工具应用":
                    report += "- 尝试使用常见AI教学工具，如ChatGPT、智能翻译等\n"
                    report += "- 参与基础AI工具操作培训，熟悉基本功能\n"
                    report += "- 学习简单的提示工程技巧，提高工具使用效率\n"
                elif dimension == "AI教学设计":
                    report += "- 参考现有AI教学案例，在课堂中尝试简单应用\n"
                    report += "- 学习如何选择适合教学内容的AI工具\n"
                    report += "- 尝试使用AI生成的教学资源辅助教学\n"
                elif dimension == "AI伦理与批判":
                    report += "- 学习AI伦理基本原则和概念\n"
                    report += "- 了解数据隐私保护的基本知识\n"
                    report += "- 参与AI伦理意识培训活动\n"
            
            elif level == "中级(深化)":
                if dimension == "AI基础认知":
                    report += "- 深入学习AI技术原理，理解不同技术的优缺点\n"
                    report += "- 分析AI教育应用案例的实施过程和效果\n"
                    report += "- 跟踪AI研究前沿，了解技术发展趋势\n"
                elif dimension == "AI工具应用":
                    report += "- 探索多种AI工具的组合应用，提高教学效率\n"
                    report += "- 学习高级提示工程技巧，优化AI输出质量\n"
                    report += "- 参与AI工具应用的教师社区，分享经验\n"
                elif dimension == "AI教学设计":
                    report += "- 设计AI增强的教学活动，优化教学流程\n"
                    report += "- 尝试创建个性化学习路径，适应不同学生需求\n"
                    report += "- 学习使用学习数据分析工具，实践数据驱动教学\n"
                elif dimension == "AI伦理与批判":
                    report += "- 分析AI应用的伦理影响，识别潜在风险\n"
                    report += "- 制定教学中的数据保护策略，确保学生隐私安全\n"
                    report += "- 学习识别AI系统中的算法偏见，提高批判意识\n"
            
            else:  # 高级(创造)
                if dimension == "AI基础认知":
                    report += "- 参与AI教育领域的研究和创新活动\n"
                    report += "- 分享AI教育应用的专业见解和经验\n"
                    report += "- 预测AI技术对教育的长期影响，参与战略规划\n"
                elif dimension == "AI工具应用":
                    report += "- 探索AI工具的定制和优化，满足特定教学需求\n"
                    report += "- 开发创新的AI应用流程，提高教学效率\n"
                    report += "- 指导同事使用AI工具，促进团队发展\n"
                elif dimension == "AI教学设计":
                    report += "- 创新AI教学模式，开发原创的教学方法\n"
                    report += "- 构建自适应学习生态系统，实现深度个性化\n"
                    report += "- 领导AI教学变革，建立数据驱动的教学体系\n"
                elif dimension == "AI伦理与批判":
                    report += "- 参与AI伦理规范制定，引导负责任使用\n"
                    report += "- 建立数据保护和算法审核机制\n"
                    report += "- 引导学生和同事形成AI批判思维，负责任地使用AI\n"
            
            report += "\n"
        
        # 保存报告
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            return output_file
        else:
            report_file = f"教师AI素养评估报告_{teacher_id}_{result['assessment_date']}.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            return report_file
    
    def visualize_assessment(self, teacher_id: str, output_file: str = None) -> str:
        """可视化评估结果"""
        if teacher_id not in self.assessment_results:
            raise ValueError(f"未找到教师 {teacher_id} 的评估结果")
        
        result = self.assessment_results[teacher_id]
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 1. 雷达图：维度评分
        ax1 = plt.subplot(2, 2, 1, polar=True)
        
        # 准备雷达图数据
        categories = self.dimensions
        values = [result['dimension_scores'][dim] for dim in categories]
        
        # 闭合雷达图
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 绘制雷达图
        ax1.plot(angles, values, linewidth=2)
        ax1.fill(angles, values, alpha=0.2)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories[:-1])
        ax1.set_ylim(0, 10)
        ax1.set_title('AI素养维度评分')
        
        # 2. 条形图：指标得分
        ax2 = plt.subplot(2, 2, 2)
        
        # 准备条形图数据
        all_indicators = []
        indicator_scores = []
        indicator_colors = []
        
        color_map = {
            "AI基础认知": "blue",
            "AI工具应用": "green",
            "AI教学设计": "orange",
            "AI伦理与批判": "red"
        }
        
        for dimension in self.dimensions:
            for indicator in self.indicators[dimension]:
                all_indicators.append(indicator)
                indicator_scores.append(result['indicator_scores'][indicator])
                indicator_colors.append(color_map[dimension])
        
        # 绘制条形图
        bars = ax2.barh(all_indicators, indicator_scores, color=indicator_colors)
        ax2.set_xlim(0, 10)
        ax2.set_title('指标评分详情')
        ax2.set_xlabel('得分')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=dim) for dim, color in color_map.items()]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        # 3. 饼图：维度发展水平分布
        ax3 = plt.subplot(2, 2, 3)
        
        # 准备饼图数据
        level_counts = {"初级(获取)": 0, "中级(深化)": 0, "高级(创造)": 0}
        for level in result['dimension_levels'].values():
            level_counts[level] += 1
        
        # 绘制饼图
        labels = [f"{level}: {count}" for level, count in level_counts.items() if count > 0]
        sizes = [count for count in level_counts.values() if count > 0]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.axis('equal')
        ax3.set_title('发展水平分布')
        
        # 4. 文本框：总结和建议
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        summary_text = (
            f"评估日期: {result['assessment_date']}\n"
            f"整体评分: {result['overall_score']:.2f}/10\n"
            f"整体水平: {result['overall_level']}\n\n"
            "优势维度:\n"
        )
        
        # 找出最高分的维度
        max_score = max(result['dimension_scores'].values())
        for dimension, score in result['dimension_scores'].items():
            if score == max_score:
                summary_text += f"- {dimension}: {score:.2f}\n"
        
        summary_text += "\n需提升维度:\n"
        
        # 找出最低分的维度
        min_score = min(result['dimension_scores'].values())
        for dimension, score in result['dimension_scores'].items():
            if score == min_score:
                summary_text += f"- {dimension}: {score:.2f}\n"
        
        ax4.text(0, 0.5, summary_text, va='center', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        if output_file:
            plt.savefig(output_file)
        else:
            output_file = f"教师AI素养评估图表_{teacher_id}_{result['assessment_date']}.png"
            plt.savefig(output_file)
        
        plt.close()
        
        return output_file
    
    def compare_assessments(self, teacher_ids: List[str], output_file: str = None) -> str:
        """比较多位教师的评估结果"""
        # 检查所有教师ID是否有评估结果
        for teacher_id in teacher_ids:
            if teacher_id not in self.assessment_results:
                raise ValueError(f"未找到教师 {teacher_id} 的评估结果")
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 1. 雷达图：维度评分比较
        ax1 = plt.subplot(2, 2, 1, polar=True)
        
        # 准备雷达图数据
        categories = self.dimensions
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        
        # 闭合雷达图
        categories = categories + [categories[0]]
        angles = angles + [angles[0]]
        
        # 为每位教师绘制一条线
        for i, teacher_id in enumerate(teacher_ids):
            result = self.assessment_results[teacher_id]
            values = [result['dimension_scores'][dim] for dim in self.dimensions]
            values = values + [values[0]]  # 闭合
            
            ax1.plot(angles, values, linewidth=2, label=f"教师 {teacher_id}")
            ax1.fill(angles, values, alpha=0.1)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories[:-1])
        ax1.set_ylim(0, 10)
        ax1.set_title('维度评分比较')
        ax1.legend(loc='upper right')
        
        # 2. 条形图：整体评分比较
        ax2 = plt.subplot(2, 2, 2)
        
        # 准备条形图数据
        overall_scores = [self.assessment_results[tid]['overall_score'] for tid in teacher_ids]
        
        # 绘制条形图
        bars = ax2.bar(teacher_ids, overall_scores)
        ax2.set_ylim(0, 10)
        ax2.set_title('整体评分比较')
        ax2.set_xlabel('教师ID')
        ax2.set_ylabel('整体评分')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 3. 堆叠条形图：各维度得分比较
        ax3 = plt.subplot(2, 2, 3)
        
        # 准备数据
        x = np.arange(len(teacher_ids))
        width = 0.2
        
        # 为每个维度绘制一组条形
        for i, dimension in enumerate(self.dimensions):
            dimension_scores = [self.assessment_results[tid]['dimension_scores'][dimension] for tid in teacher_ids]
            ax3.bar(x + i*width, dimension_scores, width, label=dimension)
        
        ax3.set_xticks(x + width * (len(self.dimensions) - 1) / 2)
        ax3.set_xticklabels(teacher_ids)
        ax3.set_ylim(0, 10)
        ax3.set_title('各维度得分比较')
        ax3.set_xlabel('教师ID')
        ax3.set_ylabel('维度评分')
        ax3.legend()
        
        # 4. 发展水平比较
        ax4 = plt.subplot(2, 2, 4)
        
        # 准备数据
        level_counts = {level: [] for level in self.levels}
        
        for teacher_id in teacher_ids:
            result = self.assessment_results[teacher_id]
            for level in self.levels:
                count = sum(1 for l in result['dimension_levels'].values() if l == level)
                level_counts[level].append(count)
        
        # 绘制堆叠条形图
        bottom = np.zeros(len(teacher_ids))
        for level, counts in level_counts.items():
            ax4.bar(teacher_ids, counts, bottom=bottom, label=level)
            bottom += np.array(counts)
        
        ax4.set_ylim(0, len(self.dimensions))
        ax4.set_title('发展水平分布比较')
        ax4.set_xlabel('教师ID')
        ax4.set_ylabel('维度数量')
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存图表
        if output_file:
            plt.savefig(output_file)
        else:
            output_file = f"教师AI素养评估比较_{'-'.join(teacher_ids)}_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(output_file)
        
        plt.close()
        
        return output_file
    
    def export_results(self, output_file: str = "教师AI素养评估结果.json") -> str:
        """导出所有评估结果"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.assessment_results, f, ensure_ascii=False, indent=4)
        
        return output_file
    
    def import_results(self, input_file: str) -> None:
        """导入评估结果"""
        with open(input_file, "r", encoding="utf-8") as f:
            self.assessment_results = json.load(f)


def main():
    """主函数示例"""
    # 初始化评估系统
    assessment_system = TeacherAILiteracyAssessment()
    
    # 创建评估表
    form_file = assessment_system.create_assessment_form()
    print(f"评估表已生成：{form_file}")
    print("请填写评估表后再次运行程序进行分析")
    
    # 模拟填写评估表并处理
    # 在实际应用中，这部分应该由用户手动填写表格，然后程序读取
    # 这里仅作演示
    
    # 假设已有填写好的评估表
    try:
        teacher_id = "T12345"
        assessment_file = "教师AI素养评估表_已填写.csv"
        
        # 如果文件不存在，则生成一个模拟的填写表
        if not os.path.exists(assessment_file):
            df = pd.read_csv(form_file, encoding="utf-8-sig")
            # 模拟填写评分
            np.random.seed(42)  # 确保结果可重现
            df["评分(1-10)"] = np.random.randint(1, 11, size=len(df))
            df.to_csv(assessment_file, index=False, encoding="utf-8-sig")
            print(f"已生成模拟评估数据：{assessment_file}")
        
        # 处理评估结果
        result = assessment_system.process_assessment(assessment_file, teacher_id)
        print(f"评估处理完成，整体评分：{result['overall_score']:.2f}，整体水平：{result['overall_level']}")
        
        # 生成评估报告
        report_file = assessment_system.generate_report(teacher_id)
        print(f"评估报告已生成：{report_file}")
        
        # 可视化评估结果
        chart_file = assessment_system.visualize_assessment(teacher_id)
        print(f"评估图表已生成：{chart_file}")
        
        # 再添加一位教师的评估结果（模拟）
        teacher_id2 = "T67890"
        df = pd.read_csv(form_file, encoding="utf-8-sig")
        np.random.seed(43)  # 不同的种子，生成不同的评分
        df["评分(1-10)"] = np.random.randint(1, 11, size=len(df))
        assessment_file2 = "教师AI素养评估表_已填写2.csv"
        df.to_csv(assessment_file2, index=False, encoding="utf-8-sig")
        
        result2 = assessment_system.process_assessment(assessment_file2, teacher_id2)
        
        # 比较两位教师的评估结果
        compare_file = assessment_system.compare_assessments([teacher_id, teacher_id2])
        print(f"教师评估比较已生成：{compare_file}")
        
        # 导出评估结果
        export_file = assessment_system.export_results()
        print(f"评估结果已导出：{export_file}")
        
    except Exception as e:
        print(f"处理评估数据时出错: {str(e)}")


if __name__ == "__main__":
    main()
