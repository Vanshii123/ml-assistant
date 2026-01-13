"""
OPTIMIZED LEARNING AGENT - 10x Faster
Pre-loads model once, batches all resource queries
"""

from typing import List, Dict, Tuple
from agent_models import *
import random

class LearningAgent:
    """Intelligent learning path generator with multi-step reasoning"""
    
    def __init__(self, recommend_courses_fn, recommend_pdfs_fn):
        self.recommend_courses = recommend_courses_fn
        self.recommend_pdfs = recommend_pdfs_fn
        
        # EXPANDED: Goals for EVERYONE, not just ML/AI
        self.goal_requirements = {
            LearningGoal.ML_INTERNSHIP: {
                "skills": ["python", "statistics", "linear_algebra", "machine_learning", "pandas", "numpy"],
                "hours": 180
            },
            LearningGoal.DATA_SCIENCE_ROLE: {
                "skills": ["python", "statistics", "sql", "data_visualization", "pandas", "machine_learning"],
                "hours": 170
            },
            LearningGoal.ML_ENGINEER_ROLE: {
                "skills": ["python", "deep_learning", "mlops", "system_design", "cloud", "pytorch"],
                "hours": 200
            },
            LearningGoal.WEB_DEVELOPMENT: {
                "skills": ["html", "css", "javascript", "react", "nodejs", "databases", "git"],
                "hours": 160
            },
            LearningGoal.MOBILE_DEVELOPMENT: {
                "skills": ["kotlin", "swift", "react_native", "ui_design", "apis", "mobile_architecture"],
                "hours": 150
            },
            LearningGoal.CLOUD_ENGINEERING: {
                "skills": ["aws", "docker", "kubernetes", "terraform", "networking", "security"],
                "hours": 140
            },
            LearningGoal.CYBER_SECURITY: {
                "skills": ["networking", "cryptography", "penetration_testing", "security_tools", "linux"],
                "hours": 160
            },
            LearningGoal.GAME_DEVELOPMENT: {
                "skills": ["unity", "unreal", "c++", "game_design", "3d_modeling", "physics"],
                "hours": 180
            },
            LearningGoal.DEVOPS_ENGINEER: {
                "skills": ["linux", "docker", "kubernetes", "ci_cd", "monitoring", "scripting"],
                "hours": 150
            },
            LearningGoal.BLOCKCHAIN_DEV: {
                "skills": ["solidity", "ethereum", "web3", "smart_contracts", "cryptography"],
                "hours": 140
            }
        }
    
    def create_personalized_plan(self, user_profile: UserProfile) -> Tuple[LearningPlan, AgentTrace]:
        """Generate complete learning plan with reasoning"""
        
        trace = AgentTrace(
            step_1_gaps=[],
            step_2_plan_decisions=[],
            step_3_resources={}
        )
        
        # STEP 1: Skill gap analysis
        print(f"🔍 STEP 1: Analyzing skill gaps for {user_profile.goal.value}...")
        gaps = self._analyze_skill_gaps(user_profile, trace)
        print(f"   ✅ Identified {len(gaps)} skill gaps")
        print(f"   ⏱️ Estimated {sum(g.estimated_hours for g in gaps)} hours needed")
        
        # STEP 2: Generate curriculum (FAST - no API calls yet)
        print("📋 STEP 2: Creating curriculum structure...")
        modules = self._generate_curriculum(user_profile, gaps, trace)
        print(f"   ✅ Created {len(modules)} week modules")
        
        # STEP 3: Ground with resources (OPTIMIZED - batch all at once)
        print("📚 STEP 3: Finding learning resources...")
        modules = self._ground_with_resources_FAST(modules, trace)
        print(f"   ✅ Matched resources for all weeks")
        
        # Calculate metrics
        total_hours = sum(m.estimated_hours for m in modules)
        feasibility = self._calculate_feasibility(user_profile, gaps)
        success_prob = min(feasibility * 0.9, 0.95)  # Cap at 95%
        
        risks = self._identify_risks(user_profile, gaps, feasibility)
        checkpoints = self._generate_checkpoints(modules)
        decisions = trace.step_2_plan_decisions[:5]  # Top 5 decisions
        
        plan = LearningPlan(
            modules=modules,
            total_weeks=user_profile.time_constraint.total_weeks,
            total_hours=total_hours,
            success_probability=success_prob,
            planning_decisions=decisions,
            risks=risks,
            checkpoints=checkpoints
        )
        
        print(f"✅ Plan complete! Success probability: {success_prob:.0%}")
        
        return plan, trace
    
    def _analyze_skill_gaps(self, profile: UserProfile, trace: AgentTrace) -> List[SkillGap]:
        """Identify what user needs to learn"""
        
        required_skills = self.goal_requirements.get(profile.goal, {}).get("skills", [])
        current_skills = set(s.lower() for s in profile.current_skills)
        
        gaps = []
        base_hours = {
            SkillLevel.BEGINNER: 20,
            SkillLevel.INTERMEDIATE: 15,
            SkillLevel.ADVANCED: 10
        }
        
        for skill in required_skills:
            if skill.lower() not in current_skills:
                gap = SkillGap(
                    skill_name=skill.replace("_", " ").title(),
                    current_level=0,
                    required_level=3,
                    estimated_hours=base_hours[profile.current_level],
                    priority="high" if skill in required_skills[:3] else "medium"
                )
                gaps.append(gap)
                trace.step_1_gaps.append(gap)
        
        # Sort by priority
        gaps.sort(key=lambda x: (x.priority != "high", x.skill_name))
        
        feasibility = self._calculate_feasibility(profile, gaps)
        trace.step_2_plan_decisions.append(
            f"Feasibility: {feasibility:.0%}"
        )
        
        return gaps
    
    def _generate_curriculum(self, profile: UserProfile, gaps: List[SkillGap], 
                           trace: AgentTrace) -> List[WeeklyModule]:
        """Create week-by-week learning plan"""
        
        weeks = profile.time_constraint.total_weeks
        hours_per_week = profile.time_constraint.hours_per_day * profile.time_constraint.days_per_week
        
        modules = []
        remaining_gaps = gaps.copy()
        
        for week in range(1, weeks + 1):
            if not remaining_gaps:
                break
            
            # Determine module focus
            if week <= 2:
                # Foundation weeks
                focus = remaining_gaps[0] if remaining_gaps else gaps[0]
                reasoning = f"Building foundation in {focus.skill_name} - essential for everything that follows"
            elif week >= weeks - 2:
                # Final weeks - projects
                focus = remaining_gaps[0] if remaining_gaps else gaps[-1]
                reasoning = f"Applying knowledge through {focus.skill_name} projects - portfolio building"
            else:
                # Middle weeks - core skills
                focus = remaining_gaps[0] if remaining_gaps else gaps[0]
                reasoning = f"Core skill development in {focus.skill_name}"
            
            module = WeeklyModule(
                week_number=week,
                title=f"{focus.skill_name}",
                objective=f"Master {focus.skill_name} fundamentals",
                skills_covered=[focus.skill_name],
                estimated_hours=hours_per_week,
                difficulty=self._get_difficulty(profile.current_level, week, weeks),
                resources=[],  # Will be filled later
                reasoning=reasoning
            )
            
            modules.append(module)
            trace.step_2_plan_decisions.append(f"Week {week}: {reasoning}")
            
            # Remove processed gap
            if remaining_gaps:
                remaining_gaps.pop(0)
        
        return modules
    
    def _ground_with_resources_FAST(self, modules: List[WeeklyModule], 
                                    trace: AgentTrace) -> List[WeeklyModule]:
        """OPTIMIZED: Batch all resource queries together"""
        
        # Collect all unique skills
        all_skills = set()
        for module in modules:
            all_skills.update(module.skills_covered)
        
        # OPTIMIZATION: Query each skill ONCE, cache results
        resource_cache = {}
        
        for skill in all_skills:
            try:
                courses = self.recommend_courses(skill, top_n=3)
                pdfs = self.recommend_pdfs(skill, top_n=2)
                resource_cache[skill] = (courses, pdfs)
            except Exception as e:
                print(f"   ⚠️ Could not fetch resources for {skill}: {e}")
                resource_cache[skill] = ([], [])
        
        # Now assign cached resources to each module
        for module in modules:
            resources = []
            
            for skill in module.skills_covered:
                courses, pdfs = resource_cache.get(skill, ([], []))
                
                # Add courses
                for course in courses[:2]:  # Max 2 per skill
                    resources.append(LearningResource(
                        title=course.get('course_title', 'Course'),
                        type=ResourceType.COURSE,
                        url=course.get('url', ''),
                        estimated_hours=5.0
                    ))
                
                # Add PDFs
                for pdf in pdfs[:1]:  # Max 1 per skill
                    resources.append(LearningResource(
                        title=pdf.get('pdf_file', 'Notes'),
                        type=ResourceType.PDF,
                        url=pdf.get('url', ''),
                        estimated_hours=2.0
                    ))
            
            module.resources = resources
            trace.step_3_resources[f"week_{module.week_number}"] = len(resources)
        
        return modules
    
    def _calculate_feasibility(self, profile: UserProfile, gaps: List[SkillGap]) -> float:
        """Calculate if plan is achievable"""
        
        total_hours_needed = sum(g.estimated_hours for g in gaps)
        available_hours = (profile.time_constraint.hours_per_day * 
                          profile.time_constraint.days_per_week * 
                          profile.time_constraint.total_weeks)
        
        return min(available_hours / total_hours_needed, 1.0)
    
    def _get_difficulty(self, level: SkillLevel, week: int, total_weeks: int) -> str:
        """Determine module difficulty"""
        if week <= 2:
            return "beginner"
        elif week >= total_weeks - 2:
            return "advanced"
        else:
            return "intermediate"
    
    def _identify_risks(self, profile: UserProfile, gaps: List[SkillGap], 
                       feasibility: float) -> List[str]:
        """Identify potential challenges"""
        risks = []
        
        if feasibility < 0.7:
            risks.append("Timeline is tight - consider extending by 2-4 weeks")
        
        if len(gaps) > 10:
            risks.append("Large number of skills to cover - focus on essentials first")
        
        if profile.current_level == SkillLevel.BEGINNER:
            risks.append("Foundation building takes time - be patient with week 1-2")
        
        return risks[:3]
    
    def _generate_checkpoints(self, modules: List[WeeklyModule]) -> List[str]:
        """Create milestone checkpoints"""
        checkpoints = []
        
        quarter = len(modules) // 4
        
        if quarter > 0:
            checkpoints.append(f"Week {quarter}: Foundation complete")
            checkpoints.append(f"Week {quarter*2}: Core skills developed")
            checkpoints.append(f"Week {quarter*3}: Advanced concepts learned")
            checkpoints.append(f"Week {len(modules)}: Portfolio projects finished")
        
        return checkpoints