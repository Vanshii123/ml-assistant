"""
EXPANDED Agent Models - Learning Goals for Everyone!
Not just AI/ML - includes Web, Mobile, Cloud, Security, Gaming, and more
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class LearningGoal(str, Enum):
    """Expanded learning goals - something for everyone!"""
    
    # Tech/Programming
    ML_INTERNSHIP = "ml_internship"
    DATA_SCIENCE_ROLE = "data_science"
    ML_ENGINEER_ROLE = "ml_engineer"
    WEB_DEVELOPMENT = "web_development"
    MOBILE_DEVELOPMENT = "mobile_development"
    GAME_DEVELOPMENT = "game_development"
    
    # DevOps/Cloud
    CLOUD_ENGINEERING = "cloud_engineering"
    DEVOPS_ENGINEER = "devops_engineer"
    
    # Security/Blockchain
    CYBER_SECURITY = "cyber_security"
    BLOCKCHAIN_DEV = "blockchain_development"
    
    # Other
    FULL_STACK_DEV = "full_stack_development"
    BACKEND_DEV = "backend_development"
    FRONTEND_DEV = "frontend_development"

class SkillLevel(str, Enum):
    """Current skill level"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class LearningStyle(str, Enum):
    """Learning preferences"""
    VISUAL = "visual"
    HANDS_ON = "hands_on"
    READING = "reading"
    BALANCED = "balanced"

class TimeConstraint(BaseModel):
    """Time availability"""
    hours_per_day: float = Field(ge=0.5, le=12)
    days_per_week: int = Field(ge=1, le=7)
    total_weeks: int = Field(ge=1, le=52)

class SkillGap(BaseModel):
    """Identified skill gap"""
    skill_name: str
    current_level: int = Field(ge=0, le=5)
    required_level: int = Field(ge=0, le=5)
    estimated_hours: float
    priority: str

class ResourceType(str, Enum):
    """Types of learning resources"""
    COURSE = "course"
    VIDEO = "video"
    BOOK = "book"
    PDF = "pdf"
    PRACTICE = "practice"
    PROJECT = "project"

class LearningResource(BaseModel):
    """A learning resource"""
    title: str
    type: ResourceType
    url: str
    estimated_hours: float
    difficulty: Optional[str] = "intermediate"

class WeeklyModule(BaseModel):
    """One week in the learning plan"""
    week_number: int
    title: str
    objective: str
    skills_covered: List[str]
    estimated_hours: float
    difficulty: str
    resources: List[LearningResource]
    reasoning: str

class LearningPlan(BaseModel):
    """Complete personalized learning plan"""
    modules: List[WeeklyModule]
    total_weeks: int
    total_hours: float
    success_probability: float
    planning_decisions: List[str]
    risks: List[str]
    checkpoints: List[str]

class UserProfile(BaseModel):
    """User's learning profile"""
    goal: LearningGoal
    current_level: SkillLevel
    current_skills: List[str] = []
    learning_style: LearningStyle = LearningStyle.BALANCED
    time_constraint: TimeConstraint

class AgentTrace(BaseModel):
    """Reasoning trace for debugging"""
    step_1_gaps: List[SkillGap] = []
    step_2_plan_decisions: List[str] = []
    step_3_resources: dict = {}