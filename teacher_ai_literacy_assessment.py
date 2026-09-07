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
        self.indicators = {
            "AI基础认知": ["AI基础知识理解", "AI教育应用认识", "AI技术发展趋势把握"],
            "AI工具应用": ["通用AI工具使用", "语言教学专用AI工具应用", "AI工具环境配置"],
            "AI教学设计": ["AI增强的教学活动设计", "个性化学习路径创建", "数据驱动教学实施"],
            "AI伦理与批判": ["AI伦理问题认识", "数据隐私与安全保护", "算法偏见与公平性审视"]
        }
        self.levels = ["初级(获取)", "中级(深化)", "高级(创造)"]
        self.level_scores = {"初级(获取)": (1, 3), "中级(深化)": (4, 7), "高级(创造)": (8, 10)}
        self.rubrics = self._load_rubrics()
        self.assessment_results = {}
    
    def _load_rubrics(self) -> Dict:
        rubrics = {}
        for ind in [i for v in self.indicators.values() for i in v]:
            rubrics[ind] = {
                "初级(获取)": f"{ind}初级水平",
                "中级(深化)": f"{ind}中级水平",
                "高级(创造)": f"{ind}高级水平",
            }
        return rubrics
    
    def create_assessment_form(self, output_file: str = "教师AI素养评估表.csv") -> str:
        rows = []
        for dimension in self.dimensions:
            for indicator in self.indicators[dimension]:
                for level in self.levels:
                    rows.append({"维度": dimension, "指标": indicator, "发展水平": level, "描述": self.rubrics[indicator][level], "评分(1-10)": ""})
        pd.DataFrame(rows).to_csv(output_file, index=False, encoding="utf-8-sig")
        return output_file
    
    def process_assessment(self, assessment_file: str, teacher_id: str) -> Dict:
        df = pd.read_csv(assessment_file, encoding="utf-8-sig")
        if df["评分(1-10)"].isna().any():
            raise ValueError("评分表未完全填写，请确保所有指标都已评分")
        df["评分(1-10)"] = df["评分(1-10)"].astype(int)
        if (df["评分(1-10)"] < 1).any() or (df["评分(1-10)"] > 10).any():
            raise ValueError("评分范围应为1-10")
        dimension_scores = {d: df[df["维度"] == d]["评分(1-10)"].mean() for d in self.dimensions}
        indicator_scores = {i: df[df["指标"] == i]["评分(1-10)"].mean() for d in self.dimensions for i in self.indicators[d]}
        dimension_levels = {}
        for dimension, score in dimension_scores.items():
            if score <= self.level_scores["初级(获取)"][1]:
                dimension_levels[dimension] = "初级(获取)"
            elif score <= self.level_scores["中级(深化)"][1]:
                dimension_levels[dimension] = "中级(深化)"
            else:
                dimension_levels[dimension] = "高级(创造)"
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        if overall_score <= self.level_scores["初级(获取)"][1]:
            overall_level = "初级(获取)"
        elif overall_score <= self.level_scores["中级(深化)"][1]:
            overall_level = "中级(深化)"
        else:
            overall_level = "高级(创造)"
        assessment_result = {
            "teacher_id": teacher_id,
            "assessment_date": datetime.now().strftime("%Y-%m-%d"),
            "dimension_scores": dimension_scores,
            "indicator_scores": indicator_scores,
            "dimension_levels": dimension_levels,
            "overall_score": overall_score,
            "overall_level": overall_level,
        }
        self.assessment_results[teacher_id] = assessment_result
        return assessment_result
    
    def generate_report(self, teacher_id: str, output_file: str = None) -> str:
        if teacher_id not in self.assessment_results:
            raise ValueError(f"未找到教师 {teacher_id} 的评估结果")
        result = self.assessment_results[teacher_id]
        report = f"# 教师AI素养评估报告\n\n**教师ID**: {teacher_id}\n**整体评分**: {result['overall_score']:.2f}/10\n**整体水平**: {result['overall_level']}\n\n"
        report += "## 维度评分\n\n"
        for dimension, score in result['dimension_scores'].items():
            report += f"- **{dimension}**: {score:.2f}/10 ({result['dimension_levels'][dimension]})\n"
        report += "\n## 发展建议\n\n"
        for dimension in self.dimensions:
            level = result['dimension_levels'][dimension]
            report += f"### {dimension}\n\n- 当前水平: {level}\n\n"
        if output_file is None:
            output_file = f"教师AI素养评估报告_{teacher_id}_{result['assessment_date']}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        return output_file
    
    def visualize_assessment(self, teacher_id: str, output_file: str = None) -> str:
        if teacher_id not in self.assessment_results:
            raise ValueError(f"未找到教师 {teacher_id} 的评估结果")
        result = self.assessment_results[teacher_id]
        plt.figure(figsize=(15, 10))
        ax1 = plt.subplot(2, 2, 1, polar=True)
        categories = list(self.dimensions)
        values = [result['dimension_scores'][dim] for dim in categories]
        # 先按原始类别数计算角度，再闭合雷达图（与 compare_assessments 一致）
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        categories = categories + [categories[0]]
        values = values + [values[0]]
        angles = angles + [angles[0]]
        ax1.plot(angles, values, linewidth=2)
        ax1.fill(angles, values, alpha=0.2)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories[:-1])
        ax1.set_ylim(0, 10)
        ax1.set_title('AI素养维度评分')
        ax2 = plt.subplot(2, 2, 2)
        all_indicators, indicator_scores, indicator_colors = [], [], []
        color_map = {"AI基础认知": "blue", "AI工具应用": "green", "AI教学设计": "orange", "AI伦理与批判": "red"}
        for dimension in self.dimensions:
            for indicator in self.indicators[dimension]:
                all_indicators.append(indicator)
                indicator_scores.append(result['indicator_scores'][indicator])
                indicator_colors.append(color_map[dimension])
        ax2.barh(all_indicators, indicator_scores, color=indicator_colors)
        ax2.set_xlim(0, 10)
        ax2.set_title('指标评分详情')
        ax3 = plt.subplot(2, 2, 3)
        level_counts = {"初级(获取)": 0, "中级(深化)": 0, "高级(创造)": 0}
        for level in result['dimension_levels'].values():
            level_counts[level] += 1
        labels = [f"{level}: {count}" for level, count in level_counts.items() if count > 0]
        sizes = [count for count in level_counts.values() if count > 0]
        ax3.pie(sizes, labels=labels, colors=['#ff9999', '#66b3ff', '#99ff99'], autopct='%1.1f%%', startangle=90)
        ax3.set_title('发展水平分布')
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        ax4.text(0, 0.5, f"整体评分: {result['overall_score']:.2f}/10\n整体水平: {result['overall_level']}", va='center', fontsize=10)
        plt.tight_layout()
        if output_file is None:
            output_file = f"教师AI素养评估图表_{teacher_id}_{result['assessment_date']}.png"
        plt.savefig(output_file)
        plt.close()
        return output_file
    
    def compare_assessments(self, teacher_ids: List[str], output_file: str = None) -> str:
        for teacher_id in teacher_ids:
            if teacher_id not in self.assessment_results:
                raise ValueError(f"未找到教师 {teacher_id} 的评估结果")
        plt.figure(figsize=(15, 10))
        ax1 = plt.subplot(2, 2, 1, polar=True)
        categories = self.dimensions
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        categories = categories + [categories[0]]
        angles = angles + [angles[0]]
        for teacher_id in teacher_ids:
            result = self.assessment_results[teacher_id]
            values = [result['dimension_scores'][dim] for dim in self.dimensions]
            values = values + [values[0]]
            ax1.plot(angles, values, linewidth=2, label=f"教师 {teacher_id}")
            ax1.fill(angles, values, alpha=0.1)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories[:-1])
        ax1.set_ylim(0, 10)
        ax1.set_title('维度评分比较')
        ax1.legend(loc='upper right')
        ax2 = plt.subplot(2, 2, 2)
        overall_scores = [self.assessment_results[tid]['overall_score'] for tid in teacher_ids]
        ax2.bar(teacher_ids, overall_scores)
        ax2.set_ylim(0, 10)
        ax2.set_title('整体评分比较')
        plt.tight_layout()
        if output_file is None:
            output_file = f"教师AI素养评估比较_{'-'.join(teacher_ids)}_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(output_file)
        plt.close()
        return output_file
    
    def export_results(self, output_file: str = "教师AI素养评估结果.json") -> str:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.assessment_results, f, ensure_ascii=False, indent=4)
        return output_file
    
    def import_results(self, input_file: str) -> None:
        with open(input_file, "r", encoding="utf-8") as f:
            self.assessment_results = json.load(f)


def main():
    assessment_system = TeacherAILiteracyAssessment()
    form_file = assessment_system.create_assessment_form()
    print(f"评估表已生成：{form_file}")


if __name__ == "__main__":
    main()
