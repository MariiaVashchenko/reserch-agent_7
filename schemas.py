"""
Pydantic-схеми для структурованих відповідей агентів.
"""

from typing import Literal
from pydantic import BaseModel, Field
class ResearchPlan(BaseModel):
    """План дослідження від Planner Agent."""
    goal: str = Field(description="Що ми намагаємося з'ясувати")
    search_queries: list[str] = Field(description="Конкретні запити для виконання")
    sources_to_check: list[str] = Field(description="'knowledge_base', 'web', або обидва")
    output_format: str = Field(description="Як має виглядати фінальний звіт")
class CritiqueResult(BaseModel):
    """Результат оцінки від Critic Agent."""
    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(description="Чи базуються дані на актуальних джерелах?")
    is_complete: bool = Field(description="Чи повністю покрито запит користувача?")
    is_well_structured: bool = Field(description="Чи логічно організовані знахідки?")
    strengths: list[str] = Field(description="Сильні сторони дослідження")
    gaps: list[str] = Field(description="Що пропущено, застаріло або погано структуровано")
    revision_requests: list[str] = Field(description="Що виправити, якщо verdict = REVISE")