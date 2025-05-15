


# 演示使用示例
if __name__ == "__main__":
    # 创建个性化学习系统
    system = PersonalizedLearningSystem()
    
    # 模拟用户ID和日期范围
    user_id = 1500
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    # 为单个用户执行完整流程
    result = system.process_user(user_id, start_date, end_date)
    
    # 打印结果摘要
    print(f"用户 {user_id} 处理完成")
    print(f"决策支持: {result['decision_support_summary']['intervention_suggestions']} 项干预建议")
    print(f"资源调整: {result['implementation_summary'].get('resources_adjusted', 0)} 项")
    print(f"策略应用: {result['implementation_summary'].get('strategies_applied', 0)} 项")
    print(f"支持活动: {result['implementation_summary'].get('support_activities', 0)} 项")
    
    # 展示系统状态
    status = system.get_system_status()
    print(f"\n系统状态: {status['system_status']}")
    print(f"上次更新: {status['last_update']}")
    print(f"活跃用户: {status['system_metrics']['active_users']}")
"""
AI数据分析支持个性化学习模型 - 实现"采集-分析-决策-实施"的闭环系统

该系统包含四个主要模块：
1. 数据采集模块：从多种来源收集学习数据
2. 数据分析模块：多层次分析学习数据
3. 决策支持模块：生成教学决策建议
4. 教学实施模块：支持个性化教学实施
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import pickle
import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("personalized_learning_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PersonalizedLearningSystem")

# 数据库连接配置（示例）
DB_CONFIG = {
    "host": "localhost",
    "user": "user",
    "password": "password",
    "database": "learning_analytics"
}

class DataCollector:
    """数据采集模块，负责从多种来源收集学习数据"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据采集器
        
        Args:
            config: 配置信息，包括数据源连接信息
        """
        self.config = config
        self.data_sources = {}
        logger.info("数据采集模块初始化完成")
    
    def register_data_source(self, source_name: str, connection_info: Dict[str, Any]) -> None:
        """
        注册数据源
        
        Args:
            source_name: 数据源名称
            connection_info: 连接信息
        """
        self.data_sources[source_name] = connection_info
        logger.info(f"数据源 {source_name} 注册成功")
    
    def collect_lms_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        从学习管理系统收集数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含学习管理系统数据的DataFrame
        """
        # 实际实现中，这里会包含连接LMS的代码，如API调用或数据库查询
        logger.info(f"从LMS收集 {start_date} 至 {end_date} 的数据")
        
        # 模拟数据，实际应用中应从真实系统获取
        data = {
            'student_id': np.random.randint(1000, 2000, 100),
            'course_id': np.random.randint(1, 10, 100),
            'module_completion': np.random.random(100),
            'time_spent': np.random.randint(10, 200, 100),
            'score': np.random.randint(0, 100, 100),
            'timestamp': [end_date - timedelta(days=np.random.randint(0, 30)) for _ in range(100)]
        }
        return pd.DataFrame(data)
    
    def collect_mobile_app_data(self, user_ids: List[int]) -> pd.DataFrame:
        """
        从移动学习应用收集数据
        
        Args:
            user_ids: 用户ID列表
            
        Returns:
            包含移动应用学习数据的DataFrame
        """
        logger.info(f"从移动应用收集 {len(user_ids)} 个用户的数据")
        
        # 模拟数据
        data = {
            'user_id': np.random.choice(user_ids, 200),
            'feature': np.random.choice(['vocabulary', 'grammar', 'reading', 'listening'], 200),
            'duration': np.random.randint(1, 60, 200),
            'completion': np.random.random(200),
            'timestamp': [datetime.now() - timedelta(hours=np.random.randint(1, 72)) for _ in range(200)]
        }
        return pd.DataFrame(data)
    
    def collect_ai_assistant_data(self, user_ids: List[int]) -> pd.DataFrame:
        """
        从AI会话助手收集交互数据
        
        Args:
            user_ids: 用户ID列表
            
        Returns:
            包含AI助手交互数据的DataFrame
        """
        logger.info(f"从AI助手收集 {len(user_ids)} 个用户的交互数据")
        
        # 模拟会话数据
        conversation_types = ['grammar_question', 'vocabulary_clarification', 
                            'culture_inquiry', 'practice_dialogue', 'translation_help']
        
        data = {
            'user_id': np.random.choice(user_ids, 150),
            'conversation_type': np.random.choice(conversation_types, 150),
            'message_count': np.random.randint(2, 20, 150),
            'duration': np.random.randint(30, 600, 150),
            'sentiment_score': np.random.random(150) * 2 - 1,  # -1 to 1
            'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 14)) for _ in range(150)]
        }
        return pd.DataFrame(data)
    
    def collect_assessment_data(self, user_ids: List[int]) -> pd.DataFrame:
        """
        从学习评估系统收集数据
        
        Args:
            user_ids: 用户ID列表
            
        Returns:
            包含评估数据的DataFrame
        """
        logger.info(f"从学习评估系统收集 {len(user_ids)} 个用户的评估数据")
        
        # 定义技能类型和级别
        skill_types = ['reading', 'writing', 'listening', 'speaking']
        levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        
        # 模拟评估数据
        data = {
            'user_id': np.random.choice(user_ids, 200),
            'skill_type': np.random.choice(skill_types, 200),
            'level': np.random.choice(levels, 200),
            'score': np.random.randint(40, 100, 200),
            'errors': np.random.randint(0, 15, 200),
            'completion_time': np.random.randint(5, 60, 200),
            'test_date': [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(200)]
        }
        return pd.DataFrame(data)
    
    def collect_all_data(self, user_ids: List[int], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        收集所有来源的数据
        
        Args:
            user_ids: 用户ID列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含所有数据源数据的字典
        """
        logger.info(f"开始收集 {len(user_ids)} 个用户从 {start_date} 到 {end_date} 的所有数据")
        
        all_data = {
            "lms_data": self.collect_lms_data(start_date, end_date),
            "mobile_app_data": self.collect_mobile_app_data(user_ids),
            "ai_assistant_data": self.collect_ai_assistant_data(user_ids),
            "assessment_data": self.collect_assessment_data(user_ids)
        }
        
        logger.info("所有数据收集完成")
        return all_data


class DataAnalyzer:
    """数据分析模块，负责对收集的数据进行多层次分析"""
    
    def __init__(self):
        """初始化数据分析器"""
        self.models = {}
        nltk.download('vader_lexicon', quiet=True)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        logger.info("数据分析模块初始化完成")
    
    def descriptive_analysis(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        描述性分析：呈现学习现状和表现
        
        Args:
            data_dict: 包含各数据源数据的字典
            
        Returns:
            描述性分析结果
        """
        logger.info("执行描述性分析")
        results = {}
        
        # 学习进度跟踪
        if 'lms_data' in data_dict:
            lms_data = data_dict['lms_data']
            results['progress_stats'] = {
                'avg_completion': lms_data['module_completion'].mean(),
                'completion_by_course': lms_data.groupby('course_id')['module_completion'].mean().to_dict(),
                'time_spent_total': lms_data['time_spent'].sum(),
                'time_spent_avg': lms_data['time_spent'].mean()
            }
        
        # 能力水平分布
        if 'assessment_data' in data_dict:
            assessment_data = data_dict['assessment_data']
            results['skill_distribution'] = {
                'skill_level_counts': assessment_data.groupby(['skill_type', 'level']).size().to_dict(),
                'avg_scores_by_skill': assessment_data.groupby('skill_type')['score'].mean().to_dict(),
                'overall_avg_score': assessment_data['score'].mean()
            }
        
        # 行为模式识别
        if 'mobile_app_data' in data_dict:
            mobile_data = data_dict['mobile_app_data']
            results['behavior_patterns'] = {
                'feature_preference': mobile_data.groupby('feature')['duration'].sum().to_dict(),
                'peak_usage_hours': mobile_data.groupby(mobile_data['timestamp'].dt.hour)['duration'].sum().to_dict(),
                'avg_session_duration': mobile_data['duration'].mean()
            }
        
        logger.info("描述性分析完成")
        return results
    
    def diagnostic_analysis(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        诊断性分析：揭示学习问题和原因
        
        Args:
            data_dict: 包含各数据源数据的字典
            
        Returns:
            诊断性分析结果
        """
        logger.info("执行诊断性分析")
        results = {}
        
        # 错误模式分析
        if 'assessment_data' in data_dict:
            assessment_data = data_dict['assessment_data']
            
            # 根据错误数量和得分的关系识别错误模式
            error_score_corr = assessment_data.groupby('skill_type')[['errors', 'score']].corr().iloc[0::2, 1].to_dict()
            
            # 识别错误热点（错误最多的技能类型和级别）
            error_hotspots = assessment_data.groupby(['skill_type', 'level'])['errors'].mean().nlargest(3).to_dict()
            
            results['error_patterns'] = {
                'error_score_correlation': error_score_corr,
                'error_hotspots': error_hotspots,
                'high_error_users': assessment_data.groupby('user_id')['errors'].sum().nlargest(5).to_dict()
            }
        
        # 学习障碍诊断
        if 'lms_data' in data_dict and 'assessment_data' in data_dict:
            lms_data = data_dict['lms_data']
            assessment_data = data_dict['assessment_data']
            
            # 识别完成率低但学习时间长的课程（可能表示学习困难）
            course_difficulty = lms_data.groupby('course_id')[['module_completion', 'time_spent']].mean()
            course_difficulty['efficiency'] = course_difficulty['module_completion'] / course_difficulty['time_spent']
            difficult_courses = course_difficulty.sort_values('efficiency').head(3).to_dict('index')
            
            # 识别分数与模块完成率不一致的情况
            merged_data = pd.merge(
                lms_data[['student_id', 'module_completion']],
                assessment_data[['user_id', 'score']],
                left_on='student_id',
                right_on='user_id',
                how='inner'
            )
            merged_data['completion_score_gap'] = merged_data['module_completion'] * 100 - merged_data['score']
            learning_gaps = merged_data[abs(merged_data['completion_score_gap']) > 30]
            
            results['learning_barriers'] = {
                'difficult_courses': difficult_courses,
                'completion_score_gaps': len(learning_gaps),
                'gap_examples': learning_gaps.head(5).to_dict('records') if not learning_gaps.empty else []
            }
        
        # 差异化分析
        if 'mobile_app_data' in data_dict and 'assessment_data' in data_dict:
            mobile_data = data_dict['mobile_app_data']
            assessment_data = data_dict['assessment_data']
            
            # 根据应用使用模式和评估结果对学生进行聚类
            user_features = mobile_data.groupby('user_id').agg({
                'duration': ['sum', 'mean'],
                'completion': 'mean'
            })
            user_features.columns = ['total_duration', 'avg_duration', 'avg_completion']
            
            user_scores = assessment_data.groupby('user_id')['score'].mean().reset_index()
            
            user_profiles = pd.merge(
                user_features.reset_index(),
                user_scores,
                on='user_id',
                how='outer'
            ).fillna(user_features.mean())
            
            # 标准化数据
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(user_profiles.drop('user_id', axis=1))
            
            # K均值聚类
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # 将聚类结果添加到用户数据
            user_profiles['cluster'] = clusters
            
            # 分析每个集群的特征
            cluster_profiles = user_profiles.groupby('cluster').mean().to_dict('index')
            
            results['differentiation_analysis'] = {
                'cluster_profiles': cluster_profiles,
                'cluster_sizes': user_profiles['cluster'].value_counts().to_dict(),
                'cluster_model': kmeans,
                'scaler': scaler
            }
        
        logger.info("诊断性分析完成")
        return results
    
    def predictive_analysis(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        预测性分析：预测未来学习趋势
        
        Args:
            data_dict: 包含各数据源数据的字典
            
        Returns:
            预测性分析结果
        """
        logger.info("执行预测性分析")
        results = {}
        
        # 学习成果预测
        if 'lms_data' in data_dict and 'assessment_data' in data_dict:
            lms_data = data_dict['lms_data']
            assessment_data = data_dict['assessment_data']
            
            # 准备训练数据
            merged_data = pd.merge(
                lms_data[['student_id', 'module_completion', 'time_spent']],
                assessment_data[['user_id', 'score']],
                left_on='student_id',
                right_on='user_id',
                how='inner'
            )
            
            if not merged_data.empty and len(merged_data) > 10:
                X = merged_data[['module_completion', 'time_spent']]
                y = merged_data['score']
                
                # 训练随机森林回归模型预测分数
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # 保存模型
                self.models['score_prediction'] = model
                
                # 生成预测结果
                merged_data['predicted_score'] = model.predict(X)
                mse = mean_squared_error(merged_data['score'], merged_data['predicted_score'])
                
                results['outcome_prediction'] = {
                    'model_accuracy': {'mse': mse, 'rmse': np.sqrt(mse)},
                    'feature_importance': dict(zip(['module_completion', 'time_spent'], model.feature_importances_)),
                    'sample_predictions': merged_data[['student_id', 'score', 'predicted_score']].head(10).to_dict('records')
                }
        
        # 学习风险预警
        if 'lms_data' in data_dict and 'mobile_app_data' in data_dict:
            lms_data = data_dict['lms_data']
            mobile_data = data_dict['mobile_app_data']
            
            # 合并数据
            user_lms = lms_data.groupby('student_id').agg({
                'module_completion': 'mean',
                'time_spent': 'sum',
                'score': 'mean'
            }).reset_index()
            
            user_mobile = mobile_data.groupby('user_id').agg({
                'duration': 'sum',
                'completion': 'mean'
            }).reset_index()
            
            user_data = pd.merge(
                user_lms,
                user_mobile,
                left_on='student_id',
                right_on='user_id',
                how='outer'
            ).fillna(0)
            
            # 定义风险标签（示例：低分数+低完成率=高风险）
            risk_threshold = 0.4
            user_data['at_risk'] = ((user_data['module_completion'] < risk_threshold) & 
                                  (user_data['score'] < 60)).astype(int)
            
            # 训练风险预测模型
            if len(user_data) > 10:
                X = user_data[['module_completion', 'time_spent', 'duration', 'completion']]
                y = user_data['at_risk']
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # 保存模型
                self.models['risk_prediction'] = model
                
                # 生成风险预测
                user_data['risk_probability'] = model.predict_proba(X)[:, 1]
                
                # 找出高风险学生
                high_risk_users = user_data[user_data['risk_probability'] > 0.7]
                
                results['risk_warning'] = {
                    'model_accuracy': accuracy_score(y, model.predict(X)),
                    'feature_importance': dict(zip(['module_completion', 'time_spent', 'duration', 'completion'], 
                                              model.feature_importances_)),
                    'high_risk_count': len(high_risk_users),
                    'high_risk_users': high_risk_users[['student_id', 'risk_probability']].to_dict('records') if not high_risk_users.empty else []
                }
        
        # 发展轨迹分析
        if 'assessment_data' in data_dict:
            assessment_data = data_dict['assessment_data']
            assessment_data = assessment_data.copy()
            
            # 确保测试日期是日期类型
            assessment_data['test_date'] = pd.to_datetime(assessment_data['test_date'])
            
            # 按用户和日期对数据进行排序
            assessment_data.sort_values(['user_id', 'test_date'], inplace=True)
            
            # 为每个用户的每个技能类型创建时间序列
            user_skill_data = assessment_data.groupby(['user_id', 'skill_type']).agg(list).reset_index()
            
            # 示例：分析几个用户的分数变化趋势
            trend_users = user_skill_data['user_id'].unique()[:3]
            trend_analysis = {}
            
            for user_id in trend_users:
                user_data = user_skill_data[user_skill_data['user_id'] == user_id]
                user_trends = {}
                
                for _, row in user_data.iterrows():
                    skill = row['skill_type']
                    scores = row['score']
                    dates = row['test_date']
                    
                    # 计算简单的线性趋势
                    if len(scores) > 1:
                        x = np.arange(len(scores))
                        y = np.array(scores)
                        trend = np.polyfit(x, y, 1)[0]  # 一次多项式拟合的斜率
                        user_trends[skill] = {
                            'trend': trend,
                            'start_score': scores[0],
                            'current_score': scores[-1],
                            'score_change': scores[-1] - scores[0]
                        }
                
                trend_analysis[str(user_id)] = user_trends
            
            results['trajectory_analysis'] = {
                'user_trends': trend_analysis
            }
        
        logger.info("预测性分析完成")
        return results
    
    def prescriptive_analysis(self, data_dict: Dict[str, pd.DataFrame], 
                            diagnostic_results: Dict[str, Any],
                            predictive_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范性分析：提供优化学习的建议
        
        Args:
            data_dict: 包含各数据源数据的字典
            diagnostic_results: 诊断性分析结果
            predictive_results: 预测性分析结果
            
        Returns:
            规范性分析结果
        """
        logger.info("执行规范性分析")
        results = {}
        
        # 个性化学习路径
        if 'assessment_data' in data_dict and 'differentiation_analysis' in diagnostic_results:
            assessment_data = data_dict['assessment_data']
            
            # 获取聚类模型
            cluster_model = diagnostic_results['differentiation_analysis'].get('cluster_model')
            scaler = diagnostic_results['differentiation_analysis'].get('scaler')
            
            if cluster_model and scaler:
                # 为每个用户生成学习路径建议
                user_paths = {}
                
                # 获取代表性用户（每个聚类的几个用户）
                representative_users = []
                
                for cluster_id in range(3):  # 假设有3个聚类
                    # 在实际应用中，这里应该是基于聚类结果选择的代表性用户
                    # 为了示例，我们只选择几个随机用户
                    sample_users = assessment_data['user_id'].sample(min(3, len(assessment_data))).tolist()
                    representative_users.extend(sample_users)
                
                for user_id in representative_users:
                    user_assessments = assessment_data[assessment_data['user_id'] == user_id]
                    
                    # 基于用户的技能水平推荐学习路径
                    if not user_assessments.empty:
                        # 识别用户的弱项和强项
                        skill_scores = user_assessments.groupby('skill_type')['score'].mean()
                        weakest_skill = skill_scores.idxmin() if not skill_scores.empty else None
                        strongest_skill = skill_scores.idxmax() if not skill_scores.empty else None
                        
                        # 确定用户的平均水平
                        average_level = user_assessments['level'].mode().iloc[0] if not user_assessments['level'].empty else None
                        
                        # 根据用户特点生成路径建议
                        path_suggestions = {
                            "focus_areas": [weakest_skill] if weakest_skill else [],
                            "recommended_level": average_level,
                            "leverage_strengths": [strongest_skill] if strongest_skill else [],
                            "suggested_modules": [
                                f"{weakest_skill}_fundamentals" if weakest_skill else "general_chinese",
                                f"interactive_{strongest_skill}" if strongest_skill else "practice_dialogue",
                                "cultural_context"
                            ]
                        }
                        
                        user_paths[str(user_id)] = path_suggestions
                
                results['learning_paths'] = {
                    'user_specific_paths': user_paths,
                    'path_generation_method': "Based on skill assessment and cluster analysis"
                }
        
        # 学习策略建议
        # 基于用户行为和评估数据推荐学习策略
        if 'mobile_app_data' in data_dict and 'assessment_data' in data_dict:
            mobile_data = data_dict['mobile_app_data']
            assessment_data = data_dict['assessment_data']
            
            # 分析不同学习行为与成绩的关系
            user_mobile = mobile_data.groupby('user_id').agg({
                'duration': 'sum',
                'feature': lambda x: list(x),
                'completion': 'mean'
            }).reset_index()
            
            user_scores = assessment_data.groupby('user_id')['score'].mean().reset_index()
            
            merged_behavior = pd.merge(user_mobile, user_scores, on='user_id', how='inner')
            
            # 识别成功学习者的行为模式
            successful_learners = merged_behavior[merged_behavior['score'] > 80]
            average_learners = merged_behavior[(merged_behavior['score'] > 60) & (merged_behavior['score'] <= 80)]
            struggling_learners = merged_behavior[merged_behavior['score'] <= 60]
            
            # 提取各组的学习特征
            def extract_group_features(group_df):
                if group_df.empty:
                    return {}
                
                # 提取特征使用频率
                features = []
                for feature_list in group_df['feature']:
                    features.extend(feature_list)
                
                feature_counts = pd.Series(features).value_counts().to_dict()
                total_features = sum(feature_counts.values())
                feature_freq = {k: v/total_features for k, v in feature_counts.items()}
                
                return {
                    'avg_duration': group_df['duration'].mean(),
                    'avg_completion': group_df['completion'].mean(),
                    'feature_frequency': feature_freq,
                    'avg_score': group_df['score'].mean(),
                    'sample_size': len(group_df)
                }
            
            successful_features = extract_group_features(successful_learners)
            average_features = extract_group_features(average_learners)
            struggling_features = extract_group_features(struggling_learners)
            
            # 根据分析生成策略建议
            strategy_recommendations = {}
            
            # 为挣扎的学习者推荐策略
            if successful_features and struggling_features:
                # 比较成功学习者和挣扎学习者的差异
                duration_gap = successful_features.get('avg_duration', 0) - struggling_features.get('avg_duration', 0)
                completion_gap = successful_features.get('avg_completion', 0) - struggling_features.get('avg_completion', 0)
                
                # 基于差异的建议
                recommendations = []
                
                if duration_gap > 0:
                    recommendations.append(f"增加学习时间（成功学习者平均多花{duration_gap:.1f}分钟）")
                
                if completion_gap > 0:
                    recommendations.append(f"提高模块完成率（目标完成率{successful_features.get('avg_completion', 0):.1%}）")
                
                # 比较特征使用
                successful_freq = successful_features.get('feature_frequency', {})
                struggling_freq = struggling_features.get('feature_frequency', {})
                
                for feature, freq in successful_freq.items():
                    struggling_feature_freq = struggling_freq.get(feature, 0)
                    if freq > struggling_feature_freq * 1.5:  # 成功学习者使用频率高50%以上
                        recommendations.append(f"增加{feature}学习（成功学习者使用频率为{freq:.1%}，是挣扎学习者的{freq/max(struggling_feature_freq, 0.001):.1f}倍）")
                
                strategy_recommendations['struggling_learners'] = recommendations
            
            # 为平均学习者推荐提升策略
            if successful_features and average_features:
                recommendations = []
                
                # 分析差异并生成建议
                duration_gap = successful_features.get('avg_duration', 0) - average_features.get('avg_duration', 0)
                completion_gap = successful_features.get('avg_completion', 0) - average_features.get('avg_completion', 0)
                
                if duration_gap > 0:
                    recommendations.append(f"适当增加学习时间（+{duration_gap:.1f}分钟）")
                
                if completion_gap > 0:
                    recommendations.append(f"提高课程完成质量（目标完成率{successful_features.get('avg_completion', 0):.1%}）")
                
                # 比较特征使用
                successful_freq = successful_features.get('feature_frequency', {})
                average_freq = average_features.get('feature_frequency', {})
                
                for feature, freq in successful_freq.items():
                    avg_feature_freq = average_freq.get(feature, 0)
                    if freq > avg_feature_freq * 1.2:  # 成功学习者使用频率高20%以上
                        recommendations.append(f"增加{feature}学习（目标频率{freq:.1%}）")
                
                strategy_recommendations['average_learners'] = recommendations
            
            results['learning_strategies'] = {
                'learner_profiles': {
                    'successful': successful_features,
                    'average': average_features,
                    'struggling': struggling_features
                },
                'strategy_recommendations': strategy_recommendations
            }
        
        # 资源匹配推荐
        if 'assessment_data' in data_dict:
            assessment_data = data_dict['assessment_data']
            
            # 创建用户-技能矩阵
            user_skills = assessment_data.pivot_table(
                index='user_id', 
                columns='skill_type', 
                values='score',
                aggfunc='mean'
            ).fillna(0)
            
            # 模拟资源库（在实际系统中，这应该是一个真实的资源数据库）
            resource_library = {
                'reading': [
                    {'id': 'r001', 'title': '初级阅读理解训练', 'level': 'A1', 'tags': ['beginner', 'comprehension']},
                    {'id': 'r002', 'title': '中级文章阅读与分析', 'level': 'B1', 'tags': ['intermediate', 'analysis']},
                    {'id': 'r003', 'title': '高级文学作品赏析', 'level': 'C1', 'tags': ['advanced', 'literature']}
                ],
                'writing': [
                    {'id': 'w001', 'title': '基础汉字书写', 'level': 'A1', 'tags': ['beginner', 'characters']},
                    {'id': 'w002', 'title': '中级写作技巧', 'level': 'B1', 'tags': ['intermediate', 'composition']},
                    {'id': 'w003', 'title': '高级论文写作', 'level': 'C1', 'tags': ['advanced', 'essay']}
                ],
                'listening': [
                    {'id': 'l001', 'title': '基础听力训练', 'level': 'A1', 'tags': ['beginner', 'comprehension']},
                    {'id': 'l002', 'title': '中级听力与笔记', 'level': 'B1', 'tags': ['intermediate', 'note-taking']},
                    {'id': 'l003', 'title': '高级听力理解', 'level': 'C1', 'tags': ['advanced', 'analysis']}
                ],
                'speaking': [
                    {'id': 's001', 'title': '基础口语对话', 'level': 'A1', 'tags': ['beginner', 'conversation']},
                    {'id': 's002', 'title': '中级口语表达', 'level': 'B1', 'tags': ['intermediate', 'expression']},
                    {'id': 's003', 'title': '高级演讲技巧', 'level': 'C1', 'tags': ['advanced', 'presentation']}
                ]
            }
            
            # 为每个用户匹配资源
            user_resources = {}
            
            for user_id in user_skills.index[:5]:  # 为了示例，只处理前5个用户
                user_profile = user_skills.loc[user_id]
                
                # 识别用户的弱项和强项
                weakest_skill = user_profile.idxmin()
                strongest_skill = user_profile.idxmax()
                
                # 确定用户各技能的水平
                user_levels = {}
                for skill, score in user_profile.items():
                    if score < 60:
                        user_levels[skill] = 'A1'
                    elif score < 80:
                        user_levels[skill] = 'B1'
                    else:
                        user_levels[skill] = 'C1'
                
                # 推荐资源
                recommended_resources = []
                
                # 为弱项技能推荐资源
                weak_level = user_levels.get(weakest_skill, 'A1')
                for resource in resource_library.get(weakest_skill, []):
                    if resource['level'] == weak_level:
                        recommended_resources.append({
                            'resource_id': resource['id'],
                            'title': resource['title'],
                            'skill_type': weakest_skill,
                            'reason': 'improvement_needed',
                            'priority': 'high'
                        })
                
                # 为强项技能推荐提升资源
                strong_level = user_levels.get(strongest_skill, 'A1')
                next_level = {'A1': 'B1', 'B1': 'C1', 'C1': 'C1'}.get(strong_level, 'B1')
                
                for resource in resource_library.get(strongest_skill, []):
                    if resource['level'] == next_level:
                        recommended_resources.append({
                            'resource_id': resource['id'],
                            'title': resource['title'],
                            'skill_type': strongest_skill,
                            'reason': 'skill_advancement',
                            'priority': 'medium'
                        })
                
                # 为其他技能推荐维持资源
                for skill, level in user_levels.items():
                    if skill not in [weakest_skill, strongest_skill]:
                        for resource in resource_library.get(skill, []):
                            if resource['level'] == level:
                                recommended_resources.append({
                                    'resource_id': resource['id'],
                                    'title': resource['title'],
                                    'skill_type': skill,
                                    'reason': 'skill_maintenance',
                                    'priority': 'low'
                                })
                                break  # 每个维持技能只推荐一个资源
                
                user_resources[str(user_id)] = recommended_resources
            
            results['resource_recommendations'] = {
                'user_resources': user_resources,
                'recommendation_method': "Based on skill assessment and level mapping"
            }
        
        logger.info("规范性分析完成")
        return results
    
    def analyze_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        分析所有数据，生成综合分析报告
        
        Args:
            data_dict: 包含各数据源数据的字典
            
        Returns:
            综合分析结果
        """
        logger.info("开始全面数据分析")
        
        # 执行四种类型的分析
        descriptive_results = self.descriptive_analysis(data_dict)
        diagnostic_results = self.diagnostic_analysis(data_dict)
        predictive_results = self.predictive_analysis(data_dict)
        prescriptive_results = self.prescriptive_analysis(data_dict, diagnostic_results, predictive_results)
        
        # 整合所有分析结果
        analysis_results = {
            'descriptive_analysis': descriptive_results,
            'diagnostic_analysis': diagnostic_results,
            'predictive_analysis': predictive_results,
            'prescriptive_analysis': prescriptive_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info("全面数据分析完成")
        return analysis_results


class DecisionSupport:
    """决策支持模块，负责将分析结果转化为教学决策建议"""
    
    def __init__(self):
        """初始化决策支持模块"""
        self.visualizations = {}
        logger.info("决策支持模块初始化完成")
    
    def generate_learning_report(self, user_id: int, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成学习情况报告
        
        Args:
            user_id: 用户ID
            analysis_results: 分析结果
            
        Returns:
            学习情况报告
        """
        logger.info(f"生成用户 {user_id} 的学习情况报告")
        
        # 创建学习报告结构
        report = {
            'user_id': user_id,
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'overall_summary': {},
            'skill_assessment': {},
            'learning_behavior': {},
            'progress_trends': {},
            'visualizations': {}
        }
        
        # 提取学习者整体状况
        descriptive = analysis_results.get('descriptive_analysis', {})
        diagnostic = analysis_results.get('diagnostic_analysis', {})
        predictive = analysis_results.get('predictive_analysis', {})
        
        # 生成整体摘要
        if 'skill_distribution' in descriptive:
            skill_data = descriptive['skill_distribution']
            overall_score = skill_data.get('overall_avg_score', 0)
            
            progress_status = "优秀" if overall_score >= 90 else \
                            "良好" if overall_score >= 80 else \
                            "中等" if overall_score >= 70 else \
                            "及格" if overall_score >= 60 else "需要改进"
            
            report['overall_summary'] = {
                'overall_score': overall_score,
                'progress_status': progress_status,
                'key_strengths': [],
                'improvement_areas': []
            }
        
        # 提取技能评估信息
        if 'skill_distribution' in descriptive:
            skill_scores = descriptive['skill_distribution'].get('avg_scores_by_skill', {})
            
            if skill_scores:
                # 排序技能分数
                sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)
                
                # 填充技能评估
                report['skill_assessment'] = {
                    'skill_scores': skill_scores,
                    'strongest_skill': sorted_skills[0][0] if sorted_skills else None,
                    'weakest_skill': sorted_skills[-1][0] if sorted_skills else None,
                    'skill_levels': {}
                }
                
                # 确定各技能级别
                for skill, score in skill_scores.items():
                    if score >= 90:
                        level = "高级 (C1-C2)"
                    elif score >= 75:
                        level = "中上级 (B2)"
                    elif score >= 60:
                        level = "中级 (B1)"
                    elif score >= 45:
                        level = "初中级 (A2)"
                    else:
                        level = "初级 (A1)"
                    
                    report['skill_assessment']['skill_levels'][skill] = level
                
                # 添加到整体摘要
                if sorted_skills:
                    report['overall_summary']['key_strengths'].append(sorted_skills[0][0])
                    if len(sorted_skills) > 1 and sorted_skills[1][1] > 75:
                        report['overall_summary']['key_strengths'].append(sorted_skills[1][0])
                        
                    report['overall_summary']['improvement_areas'].append(sorted_skills[-1][0])
                    if len(sorted_skills) > 1 and sorted_skills[-2][1] < 60:
                        report['overall_summary']['improvement_areas'].append(sorted_skills[-2][0])
        
        # 提取学习行为信息
        if 'behavior_patterns' in descriptive:
            behavior_data = descriptive['behavior_patterns']
            feature_preference = behavior_data.get('feature_preference', {})
            
            if feature_preference:
                # 按学习时间排序特征偏好
                sorted_features = sorted(feature_preference.items(), key=lambda x: x[1], reverse=True)
                
                # 填充学习行为信息
                report['learning_behavior'] = {
                    'preferred_features': [item[0] for item in sorted_features[:2]] if len(sorted_features) > 1 else [sorted_features[0][0]] if sorted_features else [],
                    'least_used_features': [item[0] for item in sorted_features[-2:]] if len(sorted_features) > 1 else [sorted_features[-1][0]] if sorted_features else [],
                    'avg_session_duration': behavior_data.get('avg_session_duration', 0),
                    'peak_usage_time': max(behavior_data.get('peak_usage_hours', {}).items(), key=lambda x: x[1])[0] if behavior_data.get('peak_usage_hours') else None
                }
        
        # 提取进度趋势
        if 'trajectory_analysis' in predictive:
            trajectory_data = predictive['trajectory_analysis'].get('user_trends', {}).get(str(user_id), {})
            
            if trajectory_data:
                trends = {}
                
                for skill, data in trajectory_data.items():
                    trend_direction = "上升" if data.get('trend', 0) > 0 else \
                                    "稳定" if data.get('trend', 0) == 0 else "下降"
                    
                    trends[skill] = {
                        'direction': trend_direction,
                        'change': data.get('score_change', 0),
                        'trend_value': data.get('trend', 0)
                    }
                
                report['progress_trends'] = {
                    'skill_trends': trends,
                    'overall_trend': "积极" if sum(data.get('trend', 0) for data in trajectory_data.values()) > 0 else "需要关注"
                }
        
        # 生成和添加可视化（在实际系统中，这将创建真实的图表）
        # 这里我们只生成可视化的描述
        if 'skill_assessment' in report and report['skill_assessment'].get('skill_scores'):
            report['visualizations']['skill_radar'] = "技能雷达图：展示各项语言技能的评分分布"
            
        if 'progress_trends' in report and report['progress_trends'].get('skill_trends'):
            report['visualizations']['trend_chart'] = "趋势图：展示各项技能的分数变化趋势"
            
        if 'learning_behavior' in report:
            report['visualizations']['usage_pattern'] = "使用模式图：展示学习时间分布和特征偏好"
        
        logger.info(f"用户 {user_id} 的学习情况报告生成完成")
        return report
    
    def generate_intervention_suggestions(self, user_id: int, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成干预建议
        
        Args:
            user_id: 用户ID
            analysis_results: 分析结果
            
        Returns:
            干预建议列表
        """
        logger.info(f"生成用户 {user_id} 的干预建议")
        
        suggestions = []
        
        # 从分析结果中提取相关信息
        diagnostic = analysis_results.get('diagnostic_analysis', {})
        predictive = analysis_results.get('predictive_analysis', {})
        prescriptive = analysis_results.get('prescriptive_analysis', {})
        
        # 基于错误模式的干预建议
        if 'error_patterns' in diagnostic:
            error_data = diagnostic['error_patterns']
            hotspots = error_data.get('error_hotspots', {})
            
            for (skill_type, level), error_rate in hotspots.items():
                # 针对高错误率的技能和级别生成建议
                if error_rate > 5:  # 假设高错误率阈值为5
                    suggestion = {
                        'type': 'error_pattern',
                        'skill_type': skill_type,
                        'level': level,
                        'issue': f"{skill_type} {level}级别的错误率较高 ({error_rate:.1f})",
                        'suggestion': f"提供针对{skill_type}的强化练习，专注于{level}级别常见错误的纠正",
                        'priority': 'high' if error_rate > 8 else 'medium',
                        'resources': [f"{skill_type}_practice_{level}", f"{skill_type}_error_correction"]
                    }
                    suggestions.append(suggestion)
        
        # 基于学习障碍的干预建议
        if 'learning_barriers' in diagnostic:
            barrier_data = diagnostic['learning_barriers']
            difficult_courses = barrier_data.get('difficult_courses', {})
            
            for course_id, course_data in difficult_courses.items():
                efficiency = course_data.get('efficiency', 0)
                
                if efficiency < 0.01:  # 低效率阈值
                    suggestion = {
                        'type': 'learning_barrier',
                        'course_id': course_id,
                        'issue': f"课程{course_id}的学习效率低 (效率:{efficiency:.3f})",
                        'suggestion': f"检查课程{course_id}的难度设置，可能需要提供更多支持材料或先修内容",
                        'priority': 'medium',
                        'resources': [f"course_{course_id}_prerequisites", "learning_strategy_guide"]
                    }
                    suggestions.append(suggestion)
        
        # 基于风险预警的干预建议
        if 'risk_warning' in predictive:
            risk_data = predictive['risk_warning']
            high_risk_users = risk_data.get('high_risk_users', [])
            
            # 检查当前用户是否在高风险名单中
            user_risk = next((item for item in high_risk_users if item.get('student_id') == user_id), None)
            
            if user_risk:
                risk_probability = user_risk.get('risk_probability', 0)
                
                suggestion = {
                    'type': 'risk_warning',
                    'issue': f"学习风险预警 (风险概率:{risk_probability:.2f})",
                    'suggestion': "安排一对一辅导，制定学习计划，并增加检查点以跟进进度",
                    'priority': 'high' if risk_probability > 0.8 else 'medium',
                    'resources': ["study_plan_template", "one_on_one_tutoring", "progress_tracking_tool"]
                }
                suggestions.append(suggestion)
        
        # 基于学习策略的干预建议
        if 'learning_strategies' in prescriptive:
            strategy_data = prescriptive['learning_strategies']
            recommendations = strategy_data.get('strategy_recommendations', {})
            
            # 根据用户的学习效果选择建议组
            # 这里需要确定用户属于哪个学习者群体（优秀、一般或挣扎）
            # 在实际系统中，应该基于用户的实际成绩确定
            # 这里我们假设用户是一般学习者
            learner_group = 'average_learners'  # 可能的值: 'successful_learners', 'average_learners', 'struggling_learners'
            
            strategies = recommendations.get(learner_group, [])
            
            if strategies:
                for i, strategy in enumerate(strategies):
                    suggestion = {
                        'type': 'learning_strategy',
                        'issue': f"学习策略优化建议 #{i+1}",
                        'suggestion': strategy,
                        'priority': 'medium',
                        'resources': ["study_habits_guide", "time_management_tools"]
                    }
                    suggestions.append(suggestion)
        
        # 基于资源推荐的干预建议
        if 'resource_recommendations' in prescriptive:
            resource_data = prescriptive['resource_recommendations']
            user_resources = resource_data.get('user_resources', {}).get(str(user_id), [])
            
            if user_resources:
                high_priority_resources = [r for r in user_resources if r.get('priority') == 'high']
                
                if high_priority_resources:
                    resources_list = ", ".join([f"{r.get('title')} ({r.get('skill_type')})" for r in high_priority_resources])
                    
                    suggestion = {
                        'type': 'resource_recommendation',
                        'issue': "推荐优先学习资源",
                        'suggestion': f"专注于以下资源以提高薄弱项: {resources_list}",
                        'priority': 'medium',
                        'resources': [r.get('resource_id') for r in high_priority_resources]
                    }
                    suggestions.append(suggestion)
        
        # 对建议进行优先级排序
        suggestions.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}.get(x.get('priority', 'low'), 3))
        
        logger.info(f"用户 {user_id} 的干预建议生成完成，共 {len(suggestions)} 条")
        return suggestions
    
    def recommend_resources(self, user_id: int, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        推荐学习资源
        
        Args:
            user_id: 用户ID
            analysis_results: 分析结果
            
        Returns:
            推荐资源列表
        """
        logger.info(f"为用户 {user_id} 推荐学习资源")
        
        # 从分析结果中提取资源推荐
        prescriptive = analysis_results.get('prescriptive_analysis', {})
        resource_recommendations = prescriptive.get('resource_recommendations', {})
        user_resources = resource_recommendations.get('user_resources', {}).get(str(user_id), [])
        
        # 如果有直接的资源推荐，使用它们
        if user_resources:
            logger.info(f"用户 {user_id} 的资源推荐已存在，共 {len(user_resources)} 条")
            return user_resources
        
        # 否则，基于用户技能和学习行为创建推荐
        resources = []
        
        # 从分析结果中提取其他相关信息
        descriptive = analysis_results.get('descriptive_analysis', {})
        diagnostic = analysis_results.get('diagnostic_analysis', {})
        
        # 获取用户技能信息
        skill_data = descriptive.get('skill_distribution', {})
        skill_scores = skill_data.get('avg_scores_by_skill', {})
        
        # 模拟资源库（在实际系统中，这应该是真实的资源数据库）
        resource_library = {
            'reading': [
                {'id': 'r001', 'title': '初级阅读理解训练', 'level': 'A1', 'tags': ['beginner', 'comprehension']},
                {'id': 'r002', 'title': '中级文章阅读与分析', 'level': 'B1', 'tags': ['intermediate', 'analysis']},
                {'id': 'r003', 'title': '高级文学作品赏析', 'level': 'C1', 'tags': ['advanced', 'literature']}
            ],
            'writing': [
                {'id': 'w001', 'title': '基础汉字书写', 'level': 'A1', 'tags': ['beginner', 'characters']},
                {'id': 'w002', 'title': '中级写作技巧', 'level': 'B1', 'tags': ['intermediate', 'composition']},
                {'id': 'w003', 'title': '高级论文写作', 'level': 'C1', 'tags': ['advanced', 'essay']}
            ],
            'listening': [
                {'id': 'l001', 'title': '基础听力训练', 'level': 'A1', 'tags': ['beginner', 'comprehension']},
                {'id': 'l002', 'title': '中级听力与笔记', 'level': 'B1', 'tags': ['intermediate', 'note-taking']},
                {'id': 'l003', 'title': '高级听力理解', 'level': 'C1', 'tags': ['advanced', 'analysis']}
            ],
            'speaking': [
                {'id': 's001', 'title': '基础口语对话', 'level': 'A1', 'tags': ['beginner', 'conversation']},
                {'id': 's002', 'title': '中级口语表达', 'level': 'B1', 'tags': ['intermediate', 'expression']},
                {'id': 's003', 'title': '高级演讲技巧', 'level': 'C1', 'tags': ['advanced', 'presentation']}
            ]
        }
        
        if skill_scores:
            # 找出最弱和最强的技能
            weakest_skill = min(skill_scores.items(), key=lambda x: x[1])[0]
            strongest_skill = max(skill_scores.items(), key=lambda x: x[1])[0]
            
            # 为每个技能确定适当的级别
            skill_levels = {}
            for skill, score in skill_scores.items():
                if score < 60:
                    skill_levels[skill] = 'A1'
                elif score < 80:
                    skill_levels[skill] = 'B1'
                else:
                    skill_levels[skill] = 'C1'
            
            # 为弱项技能添加资源
            if weakest_skill in resource_library:
                weak_level = skill_levels.get(weakest_skill, 'A1')
                
                for resource in resource_library[weakest_skill]:
                    if resource['level'] == weak_level:
                        resources.append({
                            'resource_id': resource['id'],
                            'title': resource['title'],
                            'skill_type': weakest_skill,
                            'reason': '需要提高的领域',
                            'priority': 'high'
                        })
            
            # 为强项技能添加资源
            if strongest_skill in resource_library:
                strong_level = skill_levels.get(strongest_skill, 'A1')
                next_level = {'A1': 'B1', 'B1': 'C1', 'C1': 'C1'}.get(strong_level, 'B1')
                
                for resource in resource_library[strongest_skill]:
                    if resource['level'] == next_level:
                        resources.append({
                            'resource_id': resource['id'],
                            'title': resource['title'],
                            'skill_type': strongest_skill,
                            'reason': '提升优势技能',
                            'priority': 'medium'
                        })
            
            # 为其他技能添加资源
            for skill, level in skill_levels.items():
                if skill not in [weakest_skill, strongest_skill] and skill in resource_library:
                    for resource in resource_library[skill]:
                        if resource['level'] == level:
                            resources.append({
                                'resource_id': resource['id'],
                                'title': resource['title'],
                                'skill_type': skill,
                                'reason': '维持当前技能',
                                'priority': 'low'
                            })
                            break  # 每个技能只添加一个维持资源
        
        # 如果没有找到技能数据，添加一些通用资源
        if not resources:
            for skill_type, skill_resources in resource_library.items():
                resources.append({
                    'resource_id': skill_resources[0]['id'],
                    'title': skill_resources[0]['title'],
                    'skill_type': skill_type,
                    'reason': '建议学习资源',
                    'priority': 'medium'
                })
        
        logger.info(f"为用户 {user_id} 创建了 {len(resources)} 条资源推荐")
        return resources
    
    def generate_decision_support(self, user_id: int, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成决策支持包
        
        Args:
            user_id: 用户ID
            analysis_results: 分析结果
            
        Returns:
            决策支持包
        """
        logger.info(f"为用户 {user_id} 生成决策支持包")
        
        # 生成各类决策支持工具
        learning_report = self.generate_learning_report(user_id, analysis_results)
        intervention_suggestions = self.generate_intervention_suggestions(user_id, analysis_results)
        resource_recommendations = self.recommend_resources(user_id, analysis_results)
        
        # 整合为决策支持包
        decision_support = {
            'user_id': user_id,
            'generation_date': datetime.now().isoformat(),
            'learning_report': learning_report,
            'intervention_suggestions': intervention_suggestions,
            'resource_recommendations': resource_recommendations
        }
        
        logger.info(f"用户 {user_id} 的决策支持包生成完成")
        return decision_support


class TeachingImplementation:
    """教学实施模块，负责支持个性化教学干预"""
    
    def __init__(self):
        """初始化教学实施模块"""
        logger.info("教学实施模块初始化完成")
    
    def adjust_learning_resources(self, user_id: int, decision_support: Dict[str, Any]) -> Dict[str, Any]:
        """
        调整学习资源
        
        Args:
            user_id: 用户ID
            decision_support: 决策支持包
            
        Returns:
            资源调整结果
        """
        logger.info(f"为用户 {user_id} 调整学习资源")
        
        result = {
            'user_id': user_id,
            'adjustment_date': datetime.now().isoformat(),
            'adjusted_resources': []
        }
        
        # 从决策支持包中提取资源推荐
        resource_recommendations = decision_support.get('resource_recommendations', [])
        
        if not resource_recommendations:
            logger.warning(f"用户 {user_id} 没有资源推荐")
            return result
        
        # 根据优先级过滤和排序资源
        high_priority = [r for r in resource_recommendations if r.get('priority') == 'high']
        medium_priority = [r for r in resource_recommendations if r.get('priority') == 'medium']
        low_priority = [r for r in resource_recommendations if r.get('priority') == 'low']
        
        # 模拟资源调整逻辑
        # 在实际系统中，这里会与学习管理系统交互，更新用户的学习路径
        
        # 高优先级资源：添加到必修项目
        for resource in high_priority:
            result['adjusted_resources'].append({
                'resource_id': resource.get('resource_id'),
                'title': resource.get('title'),
                'adjustment': 'added_to_required',
                'status': 'active',
                'due_date': (datetime.now() + timedelta(days=7)).isoformat()
            })
        
        # 中优先级资源：添加到推荐项目
        for resource in medium_priority:
            result['adjusted_resources'].append({
                'resource_id': resource.get('resource_id'),
                'title': resource.get('title'),
                'adjustment': 'added_to_recommended',
                'status': 'available',
                'due_date': None
            })
        
        # 低优先级资源：添加到选修项目
        for resource in low_priority:
            result['adjusted_resources'].append({
                'resource_id': resource.get('resource_id'),
                'title': resource.get('title'),
                'adjustment': 'added_to_optional',
                'status': 'available',
                'due_date': None
            })
        
        # 添加资源调整的摘要信息
        result['summary'] = {
            'required_count': len(high_priority),
            'recommended_count': len(medium_priority),
            'optional_count': len(low_priority),
            'total_adjusted': len(high_priority) + len(medium_priority) + len(low_priority)
        }
        
        logger.info(f"用户 {user_id} 的学习资源调整完成，共调整 {result['summary']['total_adjusted']} 项")
        return result
    
    def optimize_teaching_strategies(self, user_id: int, decision_support: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化教学策略
        
        Args:
            user_id: 用户ID
            decision_support: 决策支持包
            
        Returns:
            教学策略优化结果
        """
        logger.info(f"为用户 {user_id} 优化教学策略")
        
        result = {
            'user_id': user_id,
            'optimization_date': datetime.now().isoformat(),
            'teaching_strategies': []
        }
        
        # 从决策支持包中提取信息
        learning_report = decision_support.get('learning_report', {})
        intervention_suggestions = decision_support.get('intervention_suggestions', [])
        
        # 分析学习者特征
        skill_assessment = learning_report.get('skill_assessment', {})
        learning_behavior = learning_report.get('learning_behavior', {})
        
        # 提取关键信息
        strongest_skill = skill_assessment.get('strongest_skill')
        weakest_skill = skill_assessment.get('weakest_skill')
        preferred_features = learning_behavior.get('preferred_features', [])
        
        # 为弱项技能设计策略
        if weakest_skill:
            # 基于弱项创建教学策略
            weak_strategies = []
            
            # 创建一些针对性策略
            if weakest_skill == 'reading':
                weak_strategies = [
                    "提供分级阅读材料，从易到难逐步提升",
                    "增加阅读理解练习，关注关键词识别",
                    "结合听读训练，增强语音与文字的连接"
                ]
            elif weakest_skill == 'writing':
                weak_strategies = [
                    "提供写作模板和范例，强化结构意识",
                    "增加汉字书写练习，注重笔顺和结构",
                    "实施渐进式写作任务，从句子到段落再到短文"
                ]
            elif weakest_skill == 'listening':
                weak_strategies = [
                    "提供多速度听力材料，从慢速开始",
                    "实施听写训练，关注语音识别",
                    "结合视频材料，利用视觉线索辅助理解"
                ]
            elif weakest_skill == 'speaking':
                weak_strategies = [
                    "提供对话模板和常用表达",
                    "实施口语模仿训练，关注发音和语调",
                    "设计情景对话练习，增强实用性"
                ]
            
            # 添加策略到结果
            for strategy in weak_strategies:
                result['teaching_strategies'].append({
                    'skill_focus': weakest_skill,
                    'strategy_type': 'remedial',
                    'description': strategy,
                    'priority': 'high'
                })
        
        # 利用强项技能促进学习
        if strongest_skill:
            # 基于强项创建教学策略
            strong_strategies = []
            
            # 创建一些利用强项的策略
            if strongest_skill == 'reading':
                strong_strategies = [
                    "通过阅读材料引入新语法和词汇",
                    "设计阅读后写作任务，从阅读过渡到写作",
                    "利用阅读材料进行口语讨论，促进口头表达"
                ]
            elif strongest_skill == 'writing':
                strong_strategies = [
                    "通过写作巩固新学语法点",
                    "设计写后朗读任务，促进口语表达",
                    "实施同伴写作评阅，加深理解"
                ]
            elif strongest_skill == 'listening':
                strong_strategies = [
                    "听后复述练习，促进口语表达",
                    "听写训练，促进写作能力",
                    "基于听力内容的讨论，拓展表达"
                ]
            elif strongest_skill == 'speaking':
                strong_strategies = [
                    "口头表达后记录，促进写作能力",
                    "朗读文本，连接口语和阅读",
                    "角色扮演对话，情境化语言使用"
                ]
            
            # 添加策略到结果
            for strategy in strong_strategies:
                result['teaching_strategies'].append({
                    'skill_focus': strongest_skill,
                    'strategy_type': 'leveraging_strength',
                    'description': strategy,
                    'priority': 'medium'
                })
        
        # 基于学习行为特点调整策略
        if preferred_features:
            behavior_strategies = []
            
            # 基于偏好特征创建策略
            for feature in preferred_features:
                if feature == 'vocabulary':
                    behavior_strategies.append({
                        'skill_focus': 'vocabulary_learning',
                        'strategy_type': 'preference_based',
                        'description': "提供主题词汇集，利用其词汇学习偏好",
                        'priority': 'medium'
                    })
                elif feature == 'grammar':
                    behavior_strategies.append({
                        'skill_focus': 'grammar_learning',
                        'strategy_type': 'preference_based',
                        'description': "提供结构化语法练习，利用其语法学习偏好",
                        'priority': 'medium'
                    })
                elif feature == 'reading':
                    behavior_strategies.append({
                        'skill_focus': 'reading',
                        'strategy_type': 'preference_based',
                        'description': "提供额外阅读材料，利用其阅读偏好",
                        'priority': 'medium'
                    })
                elif feature == 'listening':
                    behavior_strategies.append({
                        'skill_focus': 'listening',
                        'strategy_type': 'preference_based',
                        'description': "提供多样化听力材料，利用其听力学习偏好",
                        'priority': 'medium'
                    })
            
            # 添加策略到结果
            result['teaching_strategies'].extend(behavior_strategies)
        
        # 基于干预建议添加策略
        for suggestion in intervention_suggestions:
            if suggestion.get('type') == 'learning_strategy':
                result['teaching_strategies'].append({
                    'skill_focus': suggestion.get('skill_type', 'general'),
                    'strategy_type': 'intervention',
                    'description': suggestion.get('suggestion', ''),
                    'priority': suggestion.get('priority', 'medium')
                })
        
        # 添加策略优化的摘要信息
        result['summary'] = {
            'high_priority_strategies': len([s for s in result['teaching_strategies'] if s.get('priority') == 'high']),
            'medium_priority_strategies': len([s for s in result['teaching_strategies'] if s.get('priority') == 'medium']),
            'low_priority_strategies': len([s for s in result['teaching_strategies'] if s.get('priority') == 'low']),
            'total_strategies': len(result['teaching_strategies'])
        }
        
        logger.info(f"用户 {user_id} 的教学策略优化完成，共 {result['summary']['total_strategies']} 项策略")
        return result
    
    def provide_learning_support(self, user_id: int, decision_support: Dict[str, Any]) -> Dict[str, Any]:
        """
        提供学习支持
        
        Args:
            user_id: 用户ID
            decision_support: 决策支持包
            
        Returns:
            学习支持结果
        """
        logger.info(f"为用户 {user_id} 提供学习支持")
        
        result = {
            'user_id': user_id,
            'support_date': datetime.now().isoformat(),
            'support_activities': []
        }
        
        # 从决策支持包中提取信息
        learning_report = decision_support.get('learning_report', {})
        intervention_suggestions = decision_support.get('intervention_suggestions', [])
        
        # 提取学习者状况
        overall_summary = learning_report.get('overall_summary', {})
        progress_status = overall_summary.get('progress_status', '')
        improvement_areas = overall_summary.get('improvement_areas', [])
        
        # 根据学习者状况提供不同类型的支持
        # 1. 基础支持活动
        base_supports = [
            {
                'activity_type': 'regular_check_in',
                'title': '每周学习检查',
                'description': '安排每周15分钟的学习进度检查，回顾目标和成就',
                'frequency': 'weekly',
                'duration': 15,
                'support_level': 'basic'
            },
            {
                'activity_type': 'resource_guidance',
                'title': '学习资源指导',
                'description': '提供关于如何使用推荐学习资源的具体指导',
                'frequency': 'as_needed',
                'duration': 10,
                'support_level': 'basic'
            }
        ]
        result['support_activities'].extend(base_supports)
        
        # 2. 根据进度状态提供支持
        if progress_status in ['需要改进', '及格']:
            # 为需要更多支持的学习者提供强化支持
            intensive_supports = [
                {
                    'activity_type': 'tutoring_session',
                    'title': '一对一辅导课',
                    'description': '安排额外的一对一辅导课，针对性解决学习困难',
                    'frequency': 'bi-weekly',
                    'duration': 30,
                    'support_level': 'intensive'
                },
                {
                    'activity_type': 'study_plan',
                    'title': '详细学习计划',
                    'description': '制定详细的每日学习计划，包括目标、活动和检查点',
                    'frequency': 'daily',
                    'duration': 30,
                    'support_level': 'intensive'
                },
                {
                    'activity_type': 'skill_workshop',
                    'title': '学习技能工作坊',
                    'description': '参加学习策略工作坊，提升自我管理和学习效率',
                    'frequency': 'monthly',
                    'duration': 60,
                    'support_level': 'intensive'
                }
            ]
            result['support_activities'].extend(intensive_supports)
        elif progress_status in ['中等', '良好']:
            # 为中等程度学习者提供适度支持
            moderate_supports = [
                {
                    'activity_type': 'group_study',
                    'title': '小组学习活动',
                    'description': '参加主题学习小组，促进协作学习和同伴支持',
                    'frequency': 'weekly',
                    'duration': 45,
                    'support_level': 'moderate'
                },
                {
                    'activity_type': 'progress_review',
                    'title': '学习进度回顾',
                    'description': '定期回顾学习进度，调整学习策略和目标',
                    'frequency': 'bi-weekly',
                    'duration': 20,
                    'support_level': 'moderate'
                }
            ]
            result['support_activities'].extend(moderate_supports)
        else:  # 优秀
            # 为优秀学习者提供拓展支持
            enrichment_supports = [
                {
                    'activity_type': 'advanced_project',
                    'title': '高级学习项目',
                    'description': '参与挑战性的学习项目，拓展语言应用能力',
                    'frequency': 'monthly',
                    'duration': 90,
                    'support_level': 'enrichment'
                },
                {
                    'activity_type': 'peer_tutoring',
                    'title': '同伴辅导机会',
                    'description': '作为同伴辅导员，帮助其他学习者，巩固自身知识',
                    'frequency': 'weekly',
                    'duration': 30,
                    'support_level': 'enrichment'
                }
            ]
            result['support_activities'].extend(enrichment_supports)
        
        # 3. 针对具体改进领域的支持
        for area in improvement_areas:
            if area == 'reading':
                result['support_activities'].append({
                    'activity_type': 'targeted_support',
                    'title': '阅读理解强化',
                    'description': '提供阅读理解策略指导和练习资源，提高阅读能力',
                    'frequency': 'weekly',
                    'duration': 30,
                    'support_level': 'targeted'
                })
            elif area == 'writing':
                result['support_activities'].append({
                    'activity_type': 'targeted_support',
                    'title': '写作能力提升',
                    'description': '提供写作结构指导和反馈，改进写作质量',
                    'frequency': 'weekly',
                    'duration': 30,
                    'support_level': 'targeted'
                })
            elif area == 'listening':
                result['support_activities'].append({
                    'activity_type': 'targeted_support',
                    'title': '听力理解增强',
                    'description': '提供听力策略指导和练习材料，提高听力理解能力',
                    'frequency': 'weekly',
                    'duration': 30,
                    'support_level': 'targeted'
                })
            elif area == 'speaking':
                result['support_activities'].append({
                    'activity_type': 'targeted_support',
                    'title': '口语表达改进',
                    'description': '提供发音指导和会话练习机会，提升口语流利度',
                    'frequency': 'weekly',
                    'duration': 30,
                    'support_level': 'targeted'
                })
        
        # 4. 基于干预建议的额外支持
        for suggestion in intervention_suggestions:
            if suggestion.get('priority') == 'high':
                result['support_activities'].append({
                    'activity_type': 'intervention_support',
                    'title': f"干预支持: {suggestion.get('type', '一般')}",
                    'description': suggestion.get('suggestion', '提供针对性支持以解决学习问题'),
                    'frequency': 'as_needed',
                    'duration': 20,
                    'support_level': 'intervention'
                })
        
        # 添加支持活动的摘要信息
        support_levels = {}
        for activity in result['support_activities']:
            level = activity.get('support_level', 'other')
            support_levels[level] = support_levels.get(level, 0) + 1
        
        result['summary'] = {
            'support_levels': support_levels,
            'total_activities': len(result['support_activities']),
            'estimated_support_time': sum(activity.get('duration', 0) for activity in result['support_activities'])
        }
        
        logger.info(f"用户 {user_id} 的学习支持提供完成，共 {result['summary']['total_activities']} 项活动")
        return result
    
    def implement_personalized_teaching(self, user_id: int, decision_support: Dict[str, Any]) -> Dict[str, Any]:
        """
        实施个性化教学
        
        Args:
            user_id: 用户ID
            decision_support: 决策支持包
            
        Returns:
            个性化教学实施结果
        """
        logger.info(f"为用户 {user_id} 实施个性化教学")
        
        # 执行各个个性化教学环节
        resource_adjustment = self.adjust_learning_resources(user_id, decision_support)
        strategy_optimization = self.optimize_teaching_strategies(user_id, decision_support)
        learning_support = self.provide_learning_support(user_id, decision_support)
        
        # 整合所有实施结果
        implementation_result = {
            'user_id': user_id,
            'implementation_date': datetime.now().isoformat(),
            'resource_adjustment': resource_adjustment,
            'strategy_optimization': strategy_optimization,
            'learning_support': learning_support,
            'summary': {
                'resources_adjusted': resource_adjustment.get('summary', {}).get('total_adjusted', 0),
                'strategies_applied': strategy_optimization.get('summary', {}).get('total_strategies', 0),
                'support_activities': learning_support.get('summary', {}).get('total_activities', 0)
            }
        }
        
        logger.info(f"用户 {user_id} 的个性化教学实施完成")
        return implementation_result


class PersonalizedLearningSystem:
    """个性化学习系统，整合所有模块，实现数据驱动的个性化教学闭环"""
    
    def __init__(self, config: Dict[str, Any] = DB_CONFIG):
        """
        初始化个性化学习系统
        
        Args:
            config: 系统配置信息
        """
        self.config = config
        self.data_collector = DataCollector(config)
        self.data_analyzer = DataAnalyzer()
        self.decision_support = DecisionSupport()
        self.teaching_implementation = TeachingImplementation()
        logger.info("个性化学习系统初始化完成")
    
    def process_user(self, user_id: int, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """
        为单个用户执行完整的个性化学习处理流程
        
        Args:
            user_id: 用户ID
            start_date: 数据收集开始日期
            end_date: 数据收集结束日期
            
        Returns:
            处理结果
        """
        logger.info(f"开始为用户 {user_id} 执行个性化学习处理流程")
        
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # 1. 数据采集
        data_dict = self.data_collector.collect_all_data([user_id], start_date, end_date)
        
        # 2. 数据分析
        analysis_results = self.data_analyzer.analyze_data(data_dict)
        
        # 3. 决策支持
        decision_support = self.decision_support.generate_decision_support(user_id, analysis_results)
        
        # 4. 教学实施
        implementation_result = self.teaching_implementation.implement_personalized_teaching(user_id, decision_support)
        
        # 生成完整处理结果
        process_result = {
            'user_id': user_id,
            'process_date': datetime.now().isoformat(),
            'data_collection': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'data_sources': list(data_dict.keys())
            },
            'analysis_summary': {
                'descriptive_analysis': bool(analysis_results.get('descriptive_analysis')),
                'diagnostic_analysis': bool(analysis_results.get('diagnostic_analysis')),
                'predictive_analysis': bool(analysis_results.get('predictive_analysis')),
                'prescriptive_analysis': bool(analysis_results.get('prescriptive_analysis'))
            },
            'decision_support_summary': {
                'learning_report': bool(decision_support.get('learning_report')),
                'intervention_suggestions': len(decision_support.get('intervention_suggestions', [])),
                'resource_recommendations': len(decision_support.get('resource_recommendations', []))
            },
            'implementation_summary': implementation_result.get('summary', {})
        }
        
        logger.info(f"用户 {user_id} 的个性化学习处理流程完成")
        return process_result
    
    def process_batch(self, user_ids: List[int], start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """
        为多个用户执行批量处理
        
        Args:
            user_ids: 用户ID列表
            start_date: 数据收集开始日期
            end_date: 数据收集结束日期
            
        Returns:
            批量处理结果
        """
        logger.info(f"开始为 {len(user_ids)} 个用户执行批量处理")
        
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # 存储每个用户的处理结果
        user_results = {}
        
        # 1. 批量数据采集
        data_dict = self.data_collector.collect_all_data(user_ids, start_date, end_date)
        
        # 2. 对每个用户进行个性化处理
        for user_id in user_ids:
            try:
                # 过滤出当前用户的数据
                user_data = {
                    'lms_data': data_dict['lms_data'][data_dict['lms_data']['student_id'] == user_id],
                    'mobile_app_data': data_dict['mobile_app_data'][data_dict['mobile_app_data']['user_id'] == user_id],
                    'ai_assistant_data': data_dict['ai_assistant_data'][data_dict['ai_assistant_data']['user_id'] == user_id],
                    'assessment_data': data_dict['assessment_data'][data_dict['assessment_data']['user_id'] == user_id]
                }
                
                # 2. 数据分析
                analysis_results = self.data_analyzer.analyze_data(user_data)
                
                # 3. 决策支持
                decision_support = self.decision_support.generate_decision_support(user_id, analysis_results)
                
                # 4. 教学实施
                implementation_result = self.teaching_implementation.implement_personalized_teaching(user_id, decision_support)
                
                # 保存结果摘要
                user_results[user_id] = {
                    'status': 'success',
                    'analysis_summary': {
                        'descriptive_analysis': bool(analysis_results.get('descriptive_analysis')),
                        'diagnostic_analysis': bool(analysis_results.get('diagnostic_analysis')),
                        'predictive_analysis': bool(analysis_results.get('predictive_analysis')),
                        'prescriptive_analysis': bool(analysis_results.get('prescriptive_analysis'))
                    },
                    'implementation_summary': implementation_result.get('summary', {})
                }
                
            except Exception as e:
                logger.error(f"处理用户 {user_id} 时出错: {str(e)}")
                user_results[user_id] = {
                    'status': 'error',
                    'error_message': str(e)
                }
        
        # 生成批量处理结果摘要
        batch_result = {
            'process_date': datetime.now().isoformat(),
            'data_collection': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'user_count': len(user_ids)
            },
            'success_count': sum(1 for result in user_results.values() if result.get('status') == 'success'),
            'error_count': sum(1 for result in user_results.values() if result.get('status') == 'error'),
            'user_results': user_results
        }
        
        logger.info(f"批量处理完成: {batch_result['success_count']} 成功, {batch_result['error_count']} 失败")
        return batch_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态信息
        
        Returns:
            系统状态信息
        """
        # 在实际系统中，这里会返回真实的系统状态
        # 这里只返回模拟数据
        return {
            'system_status': 'online',
            'last_update': datetime.now().isoformat(),
            'data_collector_status': 'active',
            'data_analyzer_status': 'active',
            'decision_support_status': 'active',
            'teaching_implementation_status': 'active',
            'available_models': list(self.data_analyzer.models.keys()),
            'system_metrics': {
                'avg_processing_time': 2.5,  # 秒
                'active_users': 120,
                'total_processed_today': 350
            }
        }