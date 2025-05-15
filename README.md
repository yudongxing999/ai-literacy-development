# AI Literacy Development Framework for Chinese Language Education

This repository contains a comprehensive set of Python modules designed to support AI literacy development for international Chinese language teachers and enhance Chinese language teaching through AI integration. The framework is based on extensive research on AI applications in language education, teacher professional development, and innovative teaching strategies.

## 📋 Overview

This project provides tools and frameworks for:
- Assessing and developing teacher AI literacy
- Designing AI-enhanced teaching activities based on Bloom's Taxonomy
- Implementing active learning strategies with AI support
- Developing comprehensive language skills (listening, speaking, reading, writing) with AI assistance
- Creating personalized learning experiences through data analysis
- Generating intelligent teaching resources and providing automated feedback

The framework aims to bridge the gap between AI technology and pedagogical practice in international Chinese language education, promoting effective and ethical use of AI tools by language teachers.

## 🧩 Repository Components

### Core Modules

- **`teacher_ai_literacy_assessment.py`**: A four-dimensional assessment system for evaluating teacher AI literacy based on AI basic cognition, AI tool application, AI instructional design, and AI ethics.

- **`bloom_ai_teaching_design.py`**: Integration of Bloom's Cognitive Taxonomy with AI teaching applications, providing a framework for designing learning activities across all cognitive levels.

- **`ai_active_learning_strategies.py`**: Implementation of four active learning strategies: AI-assisted inquiry-based learning, immersive cultural experiences, collaborative problem-solving, and adaptive microlearning.

- **`ai_language_skills_system.py`**: A comprehensive system for developing listening, speaking, reading, and writing skills with AI support, including learning activity templates and tools for each skill.

- **`personalized-learning-system.py`**: A data-driven system implementing the "collection-analysis-decision-implementation" closed-loop model for personalized learning.

### Supplementary Modules

- **`# 国际中文教学资源智能生成系统核心代码.py`**: Core system for intelligent generation of Chinese teaching resources, including graded reading materials, situational dialogues, and grammar exercises.

- **`# 智能语言评估与反馈系统核心代码.py`**: Intelligent language assessment and feedback system for evaluating language performance and providing personalized guidance.

## 💻 Installation and Usage

### Prerequisites
- Python 3.8+
- Required libraries: pandas, numpy, matplotlib, scikit-learn, seaborn, nltk, tensorflow

### Installation

```bash
# Clone the repository
git clone https://github.com/yudongxing999/ai-literacy-development.git
cd ai-literacy-development

# Install required dependencies
pip install -r requirements.txt
```

### Usage Examples

#### Teacher AI Literacy Assessment
```python
from teacher_ai_literacy_assessment import TeacherAILiteracyAssessment

# Initialize the assessment system
assessment_system = TeacherAILiteracyAssessment()

# Create assessment form
form_file = assessment_system.create_assessment_form()
print(f"Assessment form created: {form_file}")

# Process assessment results for a teacher
result = assessment_system.process_assessment("completed_assessment.csv", "T12345")

# Generate assessment report
report_file = assessment_system.generate_report("T12345")
print(f"Assessment report generated: {report_file}")

# Visualize assessment results
chart_file = assessment_system.visualize_assessment("T12345")
print(f"Assessment visualization created: {chart_file}")
```

#### AI-Enhanced Language Teaching Design
```python
from bloom_ai_teaching_design import BloomAITeachingDesign

# Initialize the design model
model = BloomAITeachingDesign()

# Create a new teaching design
design = model.create_lesson_design(
    title="Character Structure Analysis", 
    content_type="Characters", 
    content="Phonetic-semantic compounds", 
    primary_level="Analysis",
    hsk_level=4,
    duration=45
)

# Export design as markdown
md_file = model.export_design(design_id, "markdown")
print(f"Teaching design exported as Markdown: {md_file}")

# Visualize the design
vis_file = model.visualize_design(design_id)
print(f"Design visualization created: {vis_file}")
```

#### Language Skills Development
```python
from ai_language_skills_system import AILanguageSkillsSystem

# Initialize the system
skills_system = AILanguageSkillsSystem()

# Create a listening skill development plan
listening_plan = skills_system.create_skill_plan(
    title="Business Communication Listening Skills",
    skill_type="听力",
    hsk_level=4,
    focus_area="Business Communication",
    duration_weeks=4,
    sessions_per_week=2,
    minutes_per_session=45
)

# Export plan to markdown
md_file = skills_system.export_plan(plan_id, "markdown")
print(f"Skill development plan exported: {md_file}")

# Visualize the plan
vis_file = skills_system.visualize_plan(plan_id)
print(f"Plan visualization created: {vis_file}")
```

## 🔍 Theoretical Background

This framework is based on extensive research on AI in education, particularly in language teaching contexts. Key theoretical foundations include:

1. **Four-Dimensional AI Literacy Framework**: Encompasses AI basic cognition, AI tool application, AI instructional design, and AI ethics and criticism.

2. **Bloom's Taxonomy Integration with AI**: Mapping AI applications to different cognitive levels (remembering, understanding, applying, analyzing, evaluating, creating).

3. **Data-Driven Personalized Learning**: Implementing a "collection-analysis-decision-implementation" closed-loop system for personalized education.

4. **Teacher Role Transformation**: Supporting teacher transition from knowledge transmitter to learning designer, from evaluator to learning facilitator, from independent worker to collaborator, and from technology user to technology leader.

5. **AI-Enhanced Active Learning Strategies**: Frameworks for inquiry-based learning, immersive cultural experiences, collaborative problem-solving, and adaptive microlearning.

For more detailed information, please refer to the academic paper "Teacher AI Literacy and the Future of Language Education" included in the repository.

## 📊 Visual Frameworks

The repository includes visualization tools for key frameworks:

- AI Literacy Assessment Framework
- Bloom's AI Teaching Design Model
- AI-Enhanced Language Skills Development
- AI-Supported Personalized Learning Model
- Teacher-AI Collaboration Model

## 🔄 Contributing

Contributions to this project are welcome! Please feel free to submit issues, fork the repository and submit pull requests.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This work is based on research in AI literacy for international Chinese language teachers and AI applications in language education. Special thanks to the researchers and educators who contributed to the theoretical foundations of this framework.

---

For questions or support, please open an issue in this repository.
