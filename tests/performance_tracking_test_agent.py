"""
Performance Tracking Test Agent
Comprehensive testing agent for performance tracking functionality with input-tool call-output pattern
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
    PATTERN_RECOGNITION = "pattern_recognition"
    LEARNING_INSIGHTS = "learning_insights"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    EDGE_CASE = "edge_case"
    INTEGRATION_TEST = "integration_test"

@dataclass
class PerformanceTestContext:
    """Input context for performance tracking testing"""
    test_id: str
    scenario_type: TestScenario
    agent_decisions: List[Dict[str, Any]]
    user_feedback: List[Dict[str, Any]]
    expected_outcomes: Dict[str, Any]
    test_parameters: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class PerformanceTestResult:
    """Output result from performance tracking testing"""
    test_id: str
    passed: bool
    performance_report: Dict[str, Any]
    identified_patterns: List[Dict[str, Any]]
    learning_insights: List[str]
    recommendations: List[str]
    execution_time_ms: float
    assertions_passed: int
    assertions_failed: int
    error_message: Optional[str]
    performance_metrics: Dict[str, float]
    validation_details: Dict[str, Any]

class PerformanceTrackingTestAgent:
    """
    Test agent for performance tracking functionality
    Follows input-tool call-output pattern with comprehensive validation
    """
    
    def __init__(self):
        self.agent_id = "performance_tracking_test_agent"
        self.test_history = []
        self.performance_stats = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "average_execution_time": 0.0,
            "analysis_accuracy": 0.0
        }
        
    def execute_test(self, test_context: PerformanceTestContext) -> PerformanceTestResult:
        """
        Execute performance tracking test with input-tool call-output pattern
        
        Input: PerformanceTestContext with agent decisions and user feedback
        Tool Calls: Pattern analysis and insight generation tools
        Output: PerformanceTestResult with comprehensive assessment
        """
        start_time = datetime.now()
        logger.info(f"Executing performance test: {test_context.test_id} - {test_context.scenario_type.value}")
        
        try:
            # Import the agent under test
            from src.agents.performance_tracking_agent import PerformanceTrackingAgent, AgentDecision, UserFeedback, PerformanceReport
            
            # Initialize test metrics
            assertions_passed = 0
            assertions_failed = 0
            validation_details = {}
            
            # TOOL CALL 1: Setup test environment
            test_agent = PerformanceTrackingAgent()
            
            # Convert test data to agent format
            agent_decisions = [AgentDecision(**decision) for decision in test_context.agent_decisions]
            user_feedback_list = [UserFeedback(**feedback) for feedback in test_context.user_feedback]
            
            # TOOL CALL 2: Execute agent analysis
            agent_result = test_agent.analyze_performance(agent_decisions, user_feedback_list)
            
            # TOOL CALL 3: Validate performance report structure
            report_validation = self._validate_performance_report(
                agent_result,
                test_context.expected_outcomes
            )
            if report_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["report_validation"] = report_validation
            
            # TOOL CALL 4: Validate pattern recognition
            pattern_validation = self._validate_pattern_recognition(
                agent_result.identified_patterns,
                test_context.expected_outcomes
            )
            if pattern_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["pattern_validation"] = pattern_validation
            
            # TOOL CALL 5: Validate learning insights
            insights_validation = self._validate_learning_insights(
                agent_result.learning_insights,
                test_context.expected_outcomes
            )
            if insights_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["insights_validation"] = insights_validation
            
            # TOOL CALL 6: Validate recommendations
            recommendations_validation = self._validate_recommendations(
                agent_result.recommendations,
                test_context.expected_outcomes
            )
            if recommendations_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["recommendations_validation"] = recommendations_validation
            
            # TOOL CALL 7: Validate statistical analysis
            stats_validation = self._validate_statistical_analysis(
                agent_result,
                agent_decisions,
                user_feedback_list,
                test_context.expected_outcomes
            )
            if stats_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["stats_validation"] = stats_validation
            
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
                agent_decisions,
                user_feedback_list,
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
            test_result = PerformanceTestResult(
                test_id=test_context.test_id,
                passed=test_passed,
                performance_report=agent_result.__dict__,
                identified_patterns=[pattern.__dict__ for pattern in agent_result.identified_patterns],
                learning_insights=agent_result.learning_insights,
                recommendations=agent_result.recommendations,
                execution_time_ms=execution_time,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                error_message=None,
                performance_metrics=performance_metrics,
                validation_details=validation_details
            )
            
            # Update performance statistics
            self._update_performance_stats(test_result)
            
            logger.info(f"Performance test completed: {test_context.test_id} - {'PASSED' if test_passed else 'FAILED'}")
            return test_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_result = PerformanceTestResult(
                test_id=test_context.test_id,
                passed=False,
                performance_report={},
                identified_patterns=[],
                learning_insights=[],
                recommendations=[],
                execution_time_ms=execution_time,
                assertions_passed=0,
                assertions_failed=1,
                error_message=str(e),
                performance_metrics={},
                validation_details={"error": str(e)}
            )
            
            self._update_performance_stats(error_result)
            logger.error(f"Performance test failed with error: {test_context.test_id} - {str(e)}")
            return error_result
    
    def _validate_performance_report(self, report: Any, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance report structure and content"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check required fields
            required_fields = [
                "overall_accuracy", "agent_specific_metrics", "trend_analysis",
                "identified_patterns", "learning_insights", "recommendations"
            ]
            
            for field in required_fields:
                if not hasattr(report, field):
                    validation["passed"] = False
                    validation["errors"].append(f"Missing required field: {field}")
                else:
                    validation["details"].append(f"Field {field} present")
            
            # Check overall accuracy range
            if hasattr(report, 'overall_accuracy'):
                if not (0 <= report.overall_accuracy <= 1):
                    validation["passed"] = False
                    validation["errors"].append(f"Overall accuracy {report.overall_accuracy} not in range [0-1]")
                else:
                    validation["details"].append(f"Overall accuracy {report.overall_accuracy} in valid range")
            
            # Check agent-specific metrics
            if hasattr(report, 'agent_specific_metrics'):
                for agent_name, metrics in report.agent_specific_metrics.items():
                    if not isinstance(metrics, dict):
                        validation["passed"] = False
                        validation["errors"].append(f"Agent metrics for {agent_name} should be a dictionary")
                    else:
                        validation["details"].append(f"Agent metrics for {agent_name} properly structured")
            
            # Check minimum accuracy if provided
            if "min_overall_accuracy" in expected and hasattr(report, 'overall_accuracy'):
                if report.overall_accuracy < expected["min_overall_accuracy"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Overall accuracy {report.overall_accuracy} below minimum {expected['min_overall_accuracy']}")
                else:
                    validation["details"].append(f"Overall accuracy meets minimum: {report.overall_accuracy} >= {expected['min_overall_accuracy']}")
                    
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_pattern_recognition(self, patterns: List[Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate identified patterns"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check if patterns exist
            if not patterns:
                if expected.get("require_patterns", True):
                    validation["passed"] = False
                    validation["errors"].append("No patterns identified")
                else:
                    validation["details"].append("No patterns required and none found")
            else:
                validation["details"].append(f"Found {len(patterns)} patterns")
            
            # Check minimum number of patterns if provided
            if "min_patterns" in expected:
                if len(patterns) < expected["min_patterns"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Only {len(patterns)} patterns found, expected at least {expected['min_patterns']}")
                else:
                    validation["details"].append(f"Sufficient patterns found: {len(patterns)} >= {expected['min_patterns']}")
            
            # Validate pattern structure
            for i, pattern in enumerate(patterns):
                required_pattern_fields = ["pattern_type", "description", "confidence", "frequency"]
                for field in required_pattern_fields:
                    if not hasattr(pattern, field):
                        validation["passed"] = False
                        validation["errors"].append(f"Pattern {i} missing field: {field}")
                    else:
                        validation["details"].append(f"Pattern {i} has field {field}")
                
                # Check confidence range
                if hasattr(pattern, 'confidence'):
                    if not (0 <= pattern.confidence <= 1):
                        validation["passed"] = False
                        validation["errors"].append(f"Pattern {i} confidence {pattern.confidence} not in range [0-1]")
                    else:
                        validation["details"].append(f"Pattern {i} confidence in valid range")
            
            # Check expected pattern types if provided
            if "expected_pattern_types" in expected:
                found_types = [pattern.pattern_type for pattern in patterns]
                for expected_type in expected["expected_pattern_types"]:
                    if expected_type not in found_types:
                        validation["passed"] = False
                        validation["errors"].append(f"Expected pattern type not found: {expected_type}")
                    else:
                        validation["details"].append(f"Found expected pattern type: {expected_type}")
                        
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_learning_insights(self, insights: List[str], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate learning insights"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check if insights exist
            if not insights:
                if expected.get("require_insights", True):
                    validation["passed"] = False
                    validation["errors"].append("No learning insights provided")
                else:
                    validation["details"].append("No insights required and none provided")
            else:
                validation["details"].append(f"Found {len(insights)} learning insights")
            
            # Check minimum number of insights if provided
            if "min_insights" in expected:
                if len(insights) < expected["min_insights"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Only {len(insights)} insights found, expected at least {expected['min_insights']}")
                else:
                    validation["details"].append(f"Sufficient insights found: {len(insights)} >= {expected['min_insights']}")
            
            # Check insight quality (length and content)
            for i, insight in enumerate(insights):
                if len(insight) < 20:  # Insights should be meaningful
                    validation["passed"] = False
                    validation["errors"].append(f"Insight {i} too short: '{insight[:50]}...'")
                else:
                    validation["details"].append(f"Insight {i} has sufficient detail")
            
            # Check for required insight keywords if provided
            if "required_insight_keywords" in expected:
                all_insights_text = " ".join(insights).lower()
                for keyword in expected["required_insight_keywords"]:
                    if keyword.lower() not in all_insights_text:
                        validation["passed"] = False
                        validation["errors"].append(f"Required insight keyword not found: {keyword}")
                    else:
                        validation["details"].append(f"Found required insight keyword: {keyword}")
                        
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_recommendations(self, recommendations: List[str], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate recommendations"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check if recommendations exist
            if not recommendations:
                if expected.get("require_recommendations", True):
                    validation["passed"] = False
                    validation["errors"].append("No recommendations provided")
                else:
                    validation["details"].append("No recommendations required and none provided")
            else:
                validation["details"].append(f"Found {len(recommendations)} recommendations")
            
            # Check minimum number of recommendations if provided
            if "min_recommendations" in expected:
                if len(recommendations) < expected["min_recommendations"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Only {len(recommendations)} recommendations found, expected at least {expected['min_recommendations']}")
                else:
                    validation["details"].append(f"Sufficient recommendations found: {len(recommendations)} >= {expected['min_recommendations']}")
            
            # Check recommendation quality (actionable and specific)
            actionable_keywords = ["adjust", "increase", "decrease", "implement", "modify", "optimize", "improve", "reduce", "enhance"]
            for i, recommendation in enumerate(recommendations):
                if len(recommendation) < 30:  # Recommendations should be detailed
                    validation["passed"] = False
                    validation["errors"].append(f"Recommendation {i} too short: '{recommendation[:50]}...'")
                else:
                    validation["details"].append(f"Recommendation {i} has sufficient detail")
                
                # Check if recommendation is actionable
                is_actionable = any(keyword in recommendation.lower() for keyword in actionable_keywords)
                if not is_actionable:
                    validation["details"].append(f"Recommendation {i} may not be actionable")
                else:
                    validation["details"].append(f"Recommendation {i} appears actionable")
            
            # Check for required recommendation categories if provided
            if "required_recommendation_categories" in expected:
                all_recommendations_text = " ".join(recommendations).lower()
                for category in expected["required_recommendation_categories"]:
                    if category.lower() not in all_recommendations_text:
                        validation["passed"] = False
                        validation["errors"].append(f"Required recommendation category not found: {category}")
                    else:
                        validation["details"].append(f"Found required recommendation category: {category}")
                        
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_statistical_analysis(self, report: Any, decisions: List[Any], feedback: List[Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical analysis accuracy"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check basic statistics calculation
            if decisions and hasattr(report, 'overall_accuracy'):
                # Calculate expected accuracy from test data
                correct_decisions = sum(1 for f in feedback if f.outcome == "accepted")
                expected_accuracy = correct_decisions / len(feedback) if feedback else 0
                
                # Allow for some variance in calculation
                accuracy_diff = abs(report.overall_accuracy - expected_accuracy)
                if accuracy_diff > 0.1:  # 10% tolerance
                    validation["passed"] = False
                    validation["errors"].append(f"Accuracy calculation incorrect: expected ~{expected_accuracy:.3f}, got {report.overall_accuracy:.3f}")
                else:
                    validation["details"].append(f"Accuracy calculation correct: {report.overall_accuracy:.3f}")
            
            # Check trend analysis if available
            if hasattr(report, 'trend_analysis') and report.trend_analysis:
                trend_fields = ["direction", "magnitude", "confidence"]
                for field in trend_fields:
                    if field not in report.trend_analysis:
                        validation["passed"] = False
                        validation["errors"].append(f"Trend analysis missing field: {field}")
                    else:
                        validation["details"].append(f"Trend analysis has field: {field}")
            
            # Check agent-specific metrics
            if hasattr(report, 'agent_specific_metrics'):
                agent_names = set(d.agent_type for d in decisions)
                for agent_name in agent_names:
                    if agent_name not in report.agent_specific_metrics:
                        validation["passed"] = False
                        validation["errors"].append(f"Missing metrics for agent: {agent_name}")
                    else:
                        validation["details"].append(f"Found metrics for agent: {agent_name}")
                        
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Statistical validation error: {str(e)}")
        
        return validation
    
    def _validate_scenario_specific(self, scenario: TestScenario, result: Any, decisions: List[Any], feedback: List[Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scenario-specific requirements"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            if scenario == TestScenario.PATTERN_RECOGNITION:
                # Should identify meaningful patterns
                if len(result.identified_patterns) < 2:
                    validation["passed"] = False
                    validation["errors"].append(f"Pattern recognition scenario should identify multiple patterns, got {len(result.identified_patterns)}")
                else:
                    validation["details"].append("Pattern recognition successful")
                    
            elif scenario == TestScenario.LEARNING_INSIGHTS:
                # Should generate actionable insights
                if len(result.learning_insights) < 3:
                    validation["passed"] = False
                    validation["errors"].append(f"Learning insights scenario should generate multiple insights, got {len(result.learning_insights)}")
                else:
                    validation["details"].append("Learning insights generation successful")
                    
            elif scenario == TestScenario.PERFORMANCE_ANALYSIS:
                # Should provide comprehensive performance analysis
                if result.overall_accuracy == 0 or not result.agent_specific_metrics:
                    validation["passed"] = False
                    validation["errors"].append("Performance analysis scenario should provide comprehensive metrics")
                else:
                    validation["details"].append("Performance analysis successful")
                    
            elif scenario == TestScenario.EDGE_CASE:
                # Should handle edge cases gracefully
                if not result.recommendations:
                    validation["passed"] = False
                    validation["errors"].append("Edge case scenario should still provide recommendations")
                else:
                    validation["details"].append("Edge case handled gracefully")
                    
            elif scenario == TestScenario.INTEGRATION_TEST:
                # Should integrate all components effectively
                components_working = (
                    len(result.identified_patterns) > 0 and
                    len(result.learning_insights) > 0 and
                    len(result.recommendations) > 0 and
                    result.overall_accuracy > 0
                )
                if not components_working:
                    validation["passed"] = False
                    validation["errors"].append("Integration test should have all components working")
                else:
                    validation["details"].append("Integration test successful")
            
            if validation["passed"]:
                validation["details"].append(f"Scenario {scenario.value} validation passed")
                
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Scenario validation error: {str(e)}")
        
        return validation
    
    def _calculate_performance_metrics(self, execution_time: float, result: Any, context: PerformanceTestContext) -> Dict[str, float]:
        """Calculate performance metrics for the test execution"""
        metrics = {
            "execution_time_ms": execution_time,
            "analysis_completeness": 1.0 if hasattr(result, 'overall_accuracy') and result.overall_accuracy > 0 else 0.0,
            "pattern_recognition_quality": min(1.0, len(result.identified_patterns) / 3.0),  # Expect at least 3 patterns
            "insight_generation_quality": min(1.0, len(result.learning_insights) / 3.0),  # Expect at least 3 insights
            "recommendation_quality": min(1.0, len(result.recommendations) / 3.0)  # Expect at least 3 recommendations
        }
        
        # Calculate pattern confidence average
        if result.identified_patterns:
            avg_pattern_confidence = sum(pattern.confidence for pattern in result.identified_patterns) / len(result.identified_patterns)
            metrics["pattern_confidence"] = avg_pattern_confidence
        else:
            metrics["pattern_confidence"] = 0.0
        
        # Calculate overall analysis quality
        quality_factors = [
            metrics["analysis_completeness"],
            metrics["pattern_recognition_quality"],
            metrics["insight_generation_quality"],
            metrics["recommendation_quality"],
            metrics["pattern_confidence"]
        ]
        metrics["overall_quality"] = sum(quality_factors) / len(quality_factors)
        
        return metrics
    
    def _update_performance_stats(self, result: PerformanceTestResult):
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
        
        # Update analysis accuracy
        if result.performance_report and "overall_accuracy" in result.performance_report:
            accuracy = result.performance_report["overall_accuracy"]
            total_accuracy = (self.performance_stats["analysis_accuracy"] * 
                            (self.performance_stats["total_tests"] - 1) + accuracy)
            self.performance_stats["analysis_accuracy"] = total_accuracy / self.performance_stats["total_tests"]
        
        # Add to test history
        self.test_history.append(result)
    
    def run_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite for performance tracking agent"""
        logger.info("Starting performance tracking test suite")
        
        # Sample agent decisions and feedback for testing
        sample_decisions = [
            {
                "decision_id": "d1",
                "agent_type": "donation_likelihood",
                "user_id": "user1",
                "context": {"transaction_amount": 500.0, "wallet_balance": 2000.0},
                "decision": {"likelihood_score": 85, "recommendation": "prompt_now"},
                "timestamp": datetime.now() - timedelta(days=1),
                "confidence": 0.8
            },
            {
                "decision_id": "d2",
                "agent_type": "cause_recommendation",
                "user_id": "user1",
                "context": {"preferred_causes": ["education"]},
                "decision": {"primary_recommendation": "education_ngo_1"},
                "timestamp": datetime.now() - timedelta(days=1),
                "confidence": 0.9
            },
            {
                "decision_id": "d3",
                "agent_type": "amount_optimization",
                "user_id": "user1",
                "context": {"financial_capacity": "high"},
                "decision": {"recommended_amounts": {"suggested": 75.0}},
                "timestamp": datetime.now() - timedelta(days=1),
                "confidence": 0.7
            }
        ]
        
        sample_feedback = [
            {
                "feedback_id": "f1",
                "decision_id": "d1",
                "user_id": "user1",
                "outcome": "accepted",
                "donation_amount": 50.0,
                "timestamp": datetime.now() - timedelta(hours=12),
                "user_satisfaction": 4.5,
                "completion_time": 120.0
            },
            {
                "feedback_id": "f2",
                "decision_id": "d2",
                "user_id": "user1",
                "outcome": "accepted",
                "selected_ngo": "education_ngo_1",
                "timestamp": datetime.now() - timedelta(hours=12),
                "user_satisfaction": 4.8,
                "completion_time": 90.0
            },
            {
                "feedback_id": "f3",
                "decision_id": "d3",
                "user_id": "user1",
                "outcome": "accepted",
                "donation_amount": 75.0,
                "timestamp": datetime.now() - timedelta(hours=12),
                "user_satisfaction": 4.2,
                "completion_time": 150.0
            }
        ]
        
        test_scenarios = [
            # Pattern recognition scenario
            PerformanceTestContext(
                test_id="pattern_recognition_success",
                scenario_type=TestScenario.PATTERN_RECOGNITION,
                agent_decisions=sample_decisions,
                user_feedback=sample_feedback,
                expected_outcomes={
                    "min_patterns": 2,
                    "expected_pattern_types": ["success_correlation", "timing_pattern"],
                    "min_overall_accuracy": 0.8
                },
                test_parameters={},
                metadata={"description": "Should identify patterns in successful decisions"}
            ),
            
            # Learning insights scenario
            PerformanceTestContext(
                test_id="learning_insights_generation",
                scenario_type=TestScenario.LEARNING_INSIGHTS,
                agent_decisions=sample_decisions * 3,  # More data for insights
                user_feedback=sample_feedback * 3,
                expected_outcomes={
                    "min_insights": 3,
                    "required_insight_keywords": ["user", "accuracy", "improvement"],
                    "min_recommendations": 2
                },
                test_parameters={},
                metadata={"description": "Should generate actionable learning insights"}
            ),
            
            # Edge case scenario
            PerformanceTestContext(
                test_id="edge_case_minimal_data",
                scenario_type=TestScenario.EDGE_CASE,
                agent_decisions=[sample_decisions[0]],  # Minimal data
                user_feedback=[sample_feedback[0]],
                expected_outcomes={
                    "require_patterns": False,
                    "require_insights": False,
                    "require_recommendations": True
                },
                test_parameters={},
                metadata={"description": "Should handle minimal data gracefully"}
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
            "average_patterns_found": sum(len(r.identified_patterns) for r in results) / len(results),
            "average_insights_generated": sum(len(r.learning_insights) for r in results) / len(results),
            "average_recommendations_made": sum(len(r.recommendations) for r in results) / len(results),
            "results": results,
            "performance_stats": self.performance_stats
        }
        
        logger.info(f"Performance tracking test suite completed. Pass rate: {summary['pass_rate']:.2%}")
        return summary

def run_performance_tracking_tests():
    """Entry point for running performance tracking tests"""
    test_agent = PerformanceTrackingTestAgent()
    return test_agent.run_test_suite()

if __name__ == "__main__":
    results = run_performance_tracking_tests()
    print(json.dumps(results, indent=2, default=str))
