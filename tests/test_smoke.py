"""Smoke tests: import core modules that do not require API keys at import time."""


def test_import_teacher_ai_literacy_assessment():
    import teacher_ai_literacy_assessment  # noqa: F401


def test_import_bloom_ai_teaching_design():
    import bloom_ai_teaching_design  # noqa: F401


def test_import_ai_active_learning_strategies():
    import ai_active_learning_strategies  # noqa: F401


def test_import_ai_language_skills_system():
    import ai_language_skills_system  # noqa: F401
