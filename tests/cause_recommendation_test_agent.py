"""
Cause Recommendation Test Agent
Comprehensive testing agent for cause recommendation functionality with input-tool call-output pattern
"""

import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestScenario(Enum):
    PREFERENCE_MATCHING = "preference_matching"
    GEOGRAPHIC_ALIGNMENT = "geographic_alignment"
    DEMOGRAPHIC_TARGETING = "demographic_targeting"
    EDGE_CASE = "edge_case"
    PERFORMANCE_TEST = "performance_test"

@dataclass
class CauseTestContext:
    """Input context for cause recommendation testing"""
    test_id: str
    scenario_type: TestScenario
    user_profile: Dict[str, Any]
    expected_outcomes: Dict[str, Any]
    test_parameters: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class CauseTestResult:
    """Output result from cause recommendation testing"""
    test_id: str
    passed: bool
    primary_recommendation: Optional[Dict[str, Any]]
    alternative_recommendations: List[Dict[str, Any]]
    personalization_score: float
    geographic_relevance: float
    execution_time_ms: float
    assertions_passed: int
    assertions_failed: int
    error_message: Optional[str]
    performance_metrics: Dict[str, float]
    validation_details: Dict[str, Any]

class CauseRecommendationTestAgent:
    """
    Test agent for cause recommendation functionality
    Follows input-tool call-output pattern with comprehensive validation
    """
    
    def __init__(self):
        self.agent_id = "cause_recommendation_test_agent"
        self.test_history = []
        self.performance_stats = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "average_execution_time": 0.0,
            "recommendation_accuracy": 0.0
        }
        
    def execute_test(self, test_context: CauseTestContext) -> CauseTestResult:
        """
        Execute cause recommendation test with input-tool call-output pattern
        
        Input: CauseTestContext with user profile and expectations
        Tool Calls: Recommendation validation and analysis tools
        Output: CauseTestResult with comprehensive assessment
        """
        start_time = datetime.now()
        logger.info(f"Executing cause test: {test_context.test_id} - {test_context.scenario_type.value}")
        
        try:
            # Import the agent under test
            from src.agents.cause_recommendation_agent import CauseRecommendationAgent, UserProfile, CauseRecommendations
            
            # Initialize test metrics
            assertions_passed = 0
            assertions_failed = 0
            validation_details = {}
            
            # TOOL CALL 1: Setup test environment
            test_agent = CauseRecommendationAgent()
            user_profile = UserProfile(**test_context.user_profile)
            
            # TOOL CALL 2: Execute agent recommendation
            agent_result = test_agent.recommend_causes(user_profile)
            
            # TOOL CALL 3: Validate primary recommendation
            primary_validation = self._validate_primary_recommendation(
                agent_result.primary_recommendation,
                test_context.expected_outcomes
            )
            if primary_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["primary_validation"] = primary_validation
            
            # TOOL CALL 4: Validate alternative recommendations
            alternatives_validation = self._validate_alternative_recommendations(
                agent_result.alternative_recommendations,
                test_context.expected_outcomes
            )
            if alternatives_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["alternatives_validation"] = alternatives_validation
            
            # TOOL CALL 5: Validate personalization score
            personalization_validation = self._validate_personalization_score(
                agent_result.personalization_score,
                test_context.expected_outcomes
            )
            if personalization_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["personalization_validation"] = personalization_validation
            
            # TOOL CALL 6: Validate geographic relevance
            geographic_validation = self._validate_geographic_relevance(
                agent_result.geographic_relevance,
                user_profile.location,
                test_context.expected_outcomes
            )
            if geographic_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["geographic_validation"] = geographic_validation
            
            # TOOL CALL 7: Validate cause category matching
            category_validation = self._validate_cause_category_matching(
                agent_result,
                user_profile.preferred_causes,
                test_context.expected_outcomes
            )
            if category_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["category_validation"] = category_validation
            
            # TOOL CALL 8: Performance metrics calculation
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_metrics = self._calculate_performance_metrics(
                execution_time,
                agent_result,
                test_context
            )
            
            # TOOL CALL 9: Scenario-specific validations
            scenario_validation = self._validate_scenario_specific(
                test_context.scenario_type,
                agent_result,
                user_profile,
                test_context.expected_outcomes
            )
            if scenario_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["scenario_validation"] = scenario_validation
            
            # Determine overall test result
            test_passed = assertions_failed == 0
            
            # Create comprehensive test result
            test_result = CauseTestResult(
                test_id=test_context.test_id,
                passed=test_passed,
                primary_recommendation=agent_result.primary_recommendation.__dict__ if agent_result.primary_recommendation else None,
                alternative_recommendations=[rec.__dict__ for rec in agent_result.alternative_recommendations],
                personalization_score=agent_result.personalization_score,
                geographic_relevance=agent_result.geographic_relevance,
                execution_time_ms=execution_time,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                error_message=None,
                performance_metrics=performance_metrics,
                validation_details=validation_details
            )
            
            # Update performance statistics
            self._update_performance_stats(test_result)
            
            logger.info(f"Cause test completed: {test_context.test_id} - {'PASSED' if test_passed else 'FAILED'}")
            return test_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_result = CauseTestResult(
                test_id=test_context.test_id,
                passed=False,
                primary_recommendation=None,
                alternative_recommendations=[],
                personalization_score=0.0,
                geographic_relevance=0.0,
                execution_time_ms=execution_time,
                assertions_passed=0,
                assertions_failed=1,
                error_message=str(e),
                performance_metrics={},
                validation_details={"error": str(e)}
            )
            
            self._update_performance_stats(error_result)
            logger.error(f"Cause test failed with error: {test_context.test_id} - {str(e)}")
            return error_result
    
    def _validate_primary_recommendation(self, recommendation: Any, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate primary recommendation structure and content"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check if recommendation exists
            if recommendation is None:
                validation["passed"] = False
                validation["errors"].append("No primary recommendation provided")
                return validation
            
            validation["details"].append("Primary recommendation exists")
            
            # Check required fields
            required_fields = ["ngo_id", "name", "cause_category", "relevance_score"]
            for field in required_fields:
                if not hasattr(recommendation, field):
                    validation["passed"] = False
                    validation["errors"].append(f"Missing required field: {field}")
                else:
                    validation["details"].append(f"Field {field} present")
            
            # Check relevance score range
            if hasattr(recommendation, 'relevance_score'):
                if not (0 <= recommendation.relevance_score <= 1):
                    validation["passed"] = False
                    validation["errors"].append(f"Relevance score {recommendation.relevance_score} not in range [0-1]")
                else:
                    validation["details"].append(f"Relevance score {recommendation.relevance_score} in valid range")
            
            # Check expected cause category if provided
            if "expected_cause_category" in expected and hasattr(recommendation, 'cause_category'):
                if recommendation.cause_category != expected["expected_cause_category"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Expected cause {expected['expected_cause_category']}, got {recommendation.cause_category}")
                else:
                    validation["details"].append(f"Cause category matches expected: {recommendation.cause_category}")
            
            # Check minimum relevance score if provided
            if "min_relevance_score" in expected and hasattr(recommendation, 'relevance_score'):
                if recommendation.relevance_score < expected["min_relevance_score"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Relevance score {recommendation.relevance_score} below minimum {expected['min_relevance_score']}")
                else:
                    validation["details"].append(f"Relevance score meets minimum: {recommendation.relevance_score} >= {expected['min_relevance_score']}")
                    
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_alternative_recommendations(self, alternatives: List[Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate alternative recommendations"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check if alternatives exist
            if not alternatives:
                if expected.get("require_alternatives", True):
                    validation["passed"] = False
                    validation["errors"].append("No alternative recommendations provided")
                else:
                    validation["details"].append("No alternatives required and none provided")
            else:
                validation["details"].append(f"Found {len(alternatives)} alternative recommendations")
            
            # Check minimum number of alternatives if provided
            if "min_alternatives" in expected:
                if len(alternatives) < expected["min_alternatives"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Only {len(alternatives)} alternatives, expected at least {expected['min_alternatives']}")
                else:
                    validation["details"].append(f"Sufficient alternatives: {len(alternatives)} >= {expected['min_alternatives']}")
            
            # Check maximum number of alternatives if provided
            if "max_alternatives" in expected:
                if len(alternatives) > expected["max_alternatives"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Too many alternatives: {len(alternatives)} > {expected['max_alternatives']}")
                else:
                    validation["details"].append(f"Alternatives within limit: {len(alternatives)} <= {expected['max_alternatives']}")
            
            # Validate each alternative
            for i, alt in enumerate(alternatives):
                if not hasattr(alt, 'relevance_score'):
                    validation["passed"] = False
                    validation["errors"].append(f"Alternative {i} missing relevance_score")
                elif not (0 <= alt.relevance_score <= 1):
                    validation["passed"] = False
                    validation["errors"].append(f"Alternative {i} relevance score {alt.relevance_score} not in range [0-1]")
                else:
                    validation["details"].append(f"Alternative {i} has valid relevance score")
            
            # Check that alternatives are sorted by relevance (descending)
            if len(alternatives) > 1:
                for i in range(len(alternatives) - 1):
                    if alternatives[i].relevance_score < alternatives[i + 1].relevance_score:
                        validation["passed"] = False
                        validation["errors"].append(f"Alternatives not sorted by relevance: {alternatives[i].relevance_score} < {alternatives[i + 1].relevance_score}")
                        break
                else:
                    validation["details"].append("Alternatives properly sorted by relevance")
                    
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_personalization_score(self, score: float, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate personalization score"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check score range
            if not (0 <= score <= 1):
                validation["passed"] = False
                validation["errors"].append(f"Personalization score {score} not in valid range [0-1]")
            else:
                validation["details"].append(f"Personalization score {score} in valid range")
            
            # Check minimum score if provided
            if "min_personalization_score" in expected:
                if score < expected["min_personalization_score"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Personalization score {score} below minimum {expected['min_personalization_score']}")
                else:
                    validation["details"].append(f"Personalization score meets minimum: {score} >= {expected['min_personalization_score']}")
            
            # Check expected score range if provided
            if "personalization_score_range" in expected:
                min_score, max_score = expected["personalization_score_range"]
                if not (min_score <= score <= max_score):
                    validation["passed"] = False
                    validation["errors"].append(f"Personalization score {score} not in expected range [{min_score}-{max_score}]")
                else:
                    validation["details"].append(f"Personalization score in expected range")
                    
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_geographic_relevance(self, relevance: float, location: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate geographic relevance"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check relevance range
            if not (0 <= relevance <= 1):
                validation["passed"] = False
                validation["errors"].append(f"Geographic relevance {relevance} not in valid range [0-1]")
            else:
                validation["details"].append(f"Geographic relevance {relevance} in valid range")
            
            # Check minimum relevance if provided
            if "min_geographic_relevance" in expected:
                if relevance < expected["min_geographic_relevance"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Geographic relevance {relevance} below minimum {expected['min_geographic_relevance']}")
                else:
                    validation["details"].append(f"Geographic relevance meets minimum: {relevance} >= {expected['min_geographic_relevance']}")
            
            # Check region-specific expectations
            if "region_expectations" in expected and "region" in location:
                user_region = location["region"]
                if user_region in expected["region_expectations"]:
                    expected_relevance = expected["region_expectations"][user_region]
                    if relevance < expected_relevance:
                        validation["passed"] = False
                        validation["errors"].append(f"Geographic relevance {relevance} below expected for region {user_region}: {expected_relevance}")
                    else:
                        validation["details"].append(f"Geographic relevance meets region expectation for {user_region}")
                        
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_cause_category_matching(self, result: Any, preferred_causes: List[str], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cause category matching against user preferences"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check if primary recommendation matches preferred causes
            if preferred_causes and result.primary_recommendation:
                primary_cause = result.primary_recommendation.cause_category
                if primary_cause not in preferred_causes:
                    # Check if it's related to preferred causes
                    cause_relations = {
                        "children": ["education", "healthcare", "nutrition"],
                        "education": ["children", "youth_development"],
                        "healthcare": ["children", "elderly", "mental_health"],
                        "environment": ["climate_change", "conservation", "sustainability"]
                    }
                    
                    is_related = False
                    for pref_cause in preferred_causes:
                        if pref_cause in cause_relations and primary_cause in cause_relations[pref_cause]:
                            is_related = True
                            break
                        if primary_cause in cause_relations and pref_cause in cause_relations[primary_cause]:
                            is_related = True
                            break
                    
                    if not is_related and expected.get("require_exact_match", False):
                        validation["passed"] = False
                        validation["errors"].append(f"Primary recommendation {primary_cause} doesn't match preferred causes {preferred_causes}")
                    else:
                        validation["details"].append(f"Primary recommendation {primary_cause} is related to preferences or exact match not required")
                else:
                    validation["details"].append(f"Primary recommendation {primary_cause} matches preferred causes")
            else:
                validation["details"].append("No preferred causes or no primary recommendation to validate")
            
            # Check cause diversity in alternatives if expected
            if "require_cause_diversity" in expected and expected["require_cause_diversity"]:
                if result.alternative_recommendations:
                    causes = [alt.cause_category for alt in result.alternative_recommendations]
                    if result.primary_recommendation:
                        causes.append(result.primary_recommendation.cause_category)
                    
                    unique_causes = set(causes)
                    if len(unique_causes) < expected.get("min_cause_diversity", 2):
                        validation["passed"] = False
                        validation["errors"].append(f"Insufficient cause diversity: {len(unique_causes)} unique causes")
                    else:
                        validation["details"].append(f"Good cause diversity: {len(unique_causes)} unique causes")
                        
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_scenario_specific(self, scenario: TestScenario, result: Any, user_profile: Any, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scenario-specific requirements"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            if scenario == TestScenario.PREFERENCE_MATCHING:
                # Should strongly match user preferences
                if user_profile.preferred_causes and result.primary_recommendation:
                    primary_cause = result.primary_recommendation.cause_category
                    if primary_cause not in user_profile.preferred_causes:
                        validation["passed"] = False
                        validation["errors"].append(f"Preference matching scenario failed: {primary_cause} not in {user_profile.preferred_causes}")
                    else:
                        validation["details"].append("Preference matching successful")
                        
            elif scenario == TestScenario.GEOGRAPHIC_ALIGNMENT:
                # Should prefer local/regional NGOs
                if result.geographic_relevance < 0.7:
                    validation["passed"] = False
                    validation["errors"].append(f"Geographic alignment scenario should have high geographic relevance, got {result.geographic_relevance}")
                else:
                    validation["details"].append("Geographic alignment successful")
                    
            elif scenario == TestScenario.DEMOGRAPHIC_TARGETING:
                # Should consider demographic information
                if result.personalization_score < 0.6:
                    validation["passed"] = False
                    validation["errors"].append(f"Demographic targeting scenario should have high personalization, got {result.personalization_score}")
                else:
                    validation["details"].append("Demographic targeting successful")
                    
            elif scenario == TestScenario.EDGE_CASE:
                # Should handle edge cases gracefully
                if not result.primary_recommendation and not result.alternative_recommendations:
                    validation["passed"] = False
                    validation["errors"].append("Edge case scenario should still provide some recommendations")
                else:
                    validation["details"].append("Edge case handled gracefully")
                    
            elif scenario == TestScenario.PERFORMANCE_TEST:
                # Should complete within time limits
                if "max_execution_time" in expected:
                    if hasattr(result, 'execution_time_ms') and result.execution_time_ms > expected["max_execution_time"]:
                        validation["passed"] = False
                        validation["errors"].append(f"Performance test exceeded time limit: {result.execution_time_ms}ms > {expected['max_execution_time']}ms")
                    else:
                        validation["details"].append("Performance test completed within time limit")
            
            if validation["passed"]:
                validation["details"].append(f"Scenario {scenario.value} validation passed")
                
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Scenario validation error: {str(e)}")
        
        return validation
    
    def _calculate_performance_metrics(self, execution_time: float, result: Any, context: CauseTestContext) -> Dict[str, float]:
        """Calculate performance metrics for the test execution"""
        metrics = {
            "execution_time_ms": execution_time,
            "recommendation_completeness": 1.0 if result.primary_recommendation else 0.0,
            "alternatives_count": len(result.alternative_recommendations),
            "personalization_quality": result.personalization_score,
            "geographic_alignment": result.geographic_relevance
        }
        
        # Calculate recommendation coverage
        total_recommendations = 1 if result.primary_recommendation else 0
        total_recommendations += len(result.alternative_recommendations)
        metrics["recommendation_coverage"] = min(1.0, total_recommendations / 3.0)  # Expect at least 3 total
        
        # Calculate overall quality score
        quality_factors = [
            metrics["recommendation_completeness"],
            metrics["recommendation_coverage"],
            metrics["personalization_quality"],
            metrics["geographic_alignment"]
        ]
        metrics["overall_quality"] = sum(quality_factors) / len(quality_factors)
        
        return metrics
    
    def _update_performance_stats(self, result: CauseTestResult):
        """Update overall performance statistics"""
        self.performance_stats["total_tests"] += 1
        
        if result.passed:
            self.performance_stats["passed_tests"] += 1
        else:
            self.performance_stats["failed_tests"] += 1
        
        # Update average execution time
        total_time = (self.performance_stats["average_execution_time"] * 
                     (self.performance_stats["total_tests"] - 1) + 
                     result.execution_time_ms)
        self.performance_stats["average_execution_time"] = total_time / self.performance_stats["total_tests"]
        
        # Update recommendation accuracy
        if result.primary_recommendation:
            accuracy_score = result.personalization_score * 0.5 + result.geographic_relevance * 0.5
            total_accuracy = (self.performance_stats["recommendation_accuracy"] * 
                            (self.performance_stats["total_tests"] - 1) + accuracy_score)
            self.performance_stats["recommendation_accuracy"] = total_accuracy / self.performance_stats["total_tests"]
        
        # Add to test history
        self.test_history.append(result)
    
    def run_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite for cause recommendation agent"""
        logger.info("Starting cause recommendation test suite")
        
        test_scenarios = [
            # Preference matching scenario
            CauseTestContext(
                test_id="preference_matching_education",
                scenario_type=TestScenario.PREFERENCE_MATCHING,
                user_profile={
                    "user_id": "test_user_1",
                    "preferred_causes": ["education", "children"],
                    "location": {"region": "NCR", "city": "Quezon City"},
                    "demographic_info": {
                        "age_range": "25-34",
                        "income_bracket": "middle",
                        "education_level": "college"
                    },
                    "giving_history": {
                        "total_donated": 1500.0,
                        "donation_count": 15,
                        "average_amount": 100.0,
                        "favorite_causes": ["education"]
                    },
                    "engagement_patterns": {
                        "preferred_time": "evening",
                        "device_type": "mobile",
                        "response_rate": 0.65
                    }
                },
                expected_outcomes={
                    "expected_cause_category": "education",
                    "min_relevance_score": 0.8,
                    "min_personalization_score": 0.7,
                    "require_exact_match": True
                },
                test_parameters={},
                metadata={"description": "Should match user's preferred education causes"}
            ),
            
            # Geographic alignment scenario
            CauseTestContext(
                test_id="geographic_alignment_local",
                scenario_type=TestScenario.GEOGRAPHIC_ALIGNMENT,
                user_profile={
                    "user_id": "test_user_2",
                    "preferred_causes": [],
                    "location": {"region": "CALABARZON", "city": "Laguna"},
                    "demographic_info": {
                        "age_range": "35-44",
                        "income_bracket": "upper_middle"
                    },
                    "giving_history": {
                        "total_donated": 500.0,
                        "donation_count": 5,
                        "average_amount": 100.0
                    },
                    "engagement_patterns": {
                        "preferred_time": "morning",
                        "device_type": "web"
                    }
                },
                expected_outcomes={
                    "min_geographic_relevance": 0.7,
                    "region_expectations": {"CALABARZON": 0.8},
                    "min_alternatives": 2
                },
                test_parameters={},
                metadata={"description": "Should prioritize NGOs operating in CALABARZON region"}
            ),
            
            # Edge case scenario
            CauseTestContext(
                test_id="edge_case_minimal_data",
                scenario_type=TestScenario.EDGE_CASE,
                user_profile={
                    "user_id": "test_user_3",
                    "preferred_causes": [],
                    "location": {"region": "Unknown"},
                    "demographic_info": {},
                    "giving_history": {
                        "total_donated": 0.0,
                        "donation_count": 0,
                        "average_amount": 0.0
                    },
                    "engagement_patterns": {}
                },
                expected_outcomes={
                    "require_alternatives": False,
                    "min_personalization_score": 0.1,
                    "min_geographic_relevance": 0.1
                },
                test_parameters={},
                metadata={"description": "Should handle minimal user data gracefully"}
            )
        ]
        
        results = []
        for scenario in test_scenarios:
            result = self.execute_test(scenario)
            results.append(result)
        
        # Generate summary report
        summary = {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "pass_rate": sum(1 for r in results if r.passed) / len(results),
            "average_execution_time": sum(r.execution_time_ms for r in results) / len(results),
            "average_personalization_score": sum(r.personalization_score for r in results) / len(results),
            "average_geographic_relevance": sum(r.geographic_relevance for r in results) / len(results),
            "results": results,
            "performance_stats": self.performance_stats
        }
        
        logger.info(f"Cause recommendation test suite completed. Pass rate: {summary['pass_rate']:.2%}")
        return summary

def run_cause_recommendation_tests():
    """Entry point for running cause recommendation tests"""
    test_agent = CauseRecommendationTestAgent()
    return test_agent.run_test_suite()

if __name__ == "__main__":
    results = run_cause_recommendation_tests()
    print(json.dumps(results, indent=2, default=str))
