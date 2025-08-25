"""
Donation Likelihood Test Agent
Comprehensive testing agent for donation likelihood assessment with input-tool call-output pattern
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
    HIGH_LIKELIHOOD = "high_likelihood"
    LOW_LIKELIHOOD = "low_likelihood"
    EDGE_CASE = "edge_case"
    STRESS_TEST = "stress_test"
    REGRESSION = "regression"

@dataclass
class TestContext:
    """Input context for test execution"""
    test_id: str
    scenario_type: TestScenario
    user_context: Dict[str, Any]
    expected_outcomes: Dict[str, Any]
    test_parameters: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class TestResult:
    """Output result from test execution"""
    test_id: str
    passed: bool
    likelihood_score: float
    recommendation: str
    reasoning_factors: List[str]
    confidence_level: str
    execution_time_ms: float
    assertions_passed: int
    assertions_failed: int
    error_message: Optional[str]
    performance_metrics: Dict[str, float]
    validation_details: Dict[str, Any]

class DonationLikelihoodTestAgent:
    """
    Test agent for donation likelihood functionality
    Follows input-tool call-output pattern with comprehensive test scenarios
    """
    
    def __init__(self):
        self.agent_id = "donation_likelihood_test_agent"
        self.test_history = []
        self.performance_stats = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "average_execution_time": 0.0,
            "test_coverage": {}
        }
        
    def execute_test(self, test_context: TestContext) -> TestResult:
        """
        Execute test scenario with input-tool call-output pattern
        
        Input: TestContext with scenario details
        Tool Calls: Various validation and testing tools
        Output: TestResult with comprehensive results
        """
        start_time = datetime.now()
        logger.info(f"Executing test: {test_context.test_id} - {test_context.scenario_type.value}")
        
        try:
            # Import the agent under test
            from src.agents.donation_likelihood_agent import DonationLikelihoodAgent, UserContext, LikelihoodScore
            
            # Initialize test metrics
            assertions_passed = 0
            assertions_failed = 0
            validation_details = {}
            
            # TOOL CALL 1: Setup test environment
            test_agent = DonationLikelihoodAgent()
            user_context = UserContext(**test_context.user_context)
            
            # TOOL CALL 2: Execute agent decision
            agent_result = test_agent.analyze_likelihood(user_context)
            
            # TOOL CALL 3: Validate likelihood score
            score_validation = self._validate_likelihood_score(
                agent_result.likelihood_score,
                test_context.expected_outcomes
            )
            if score_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["score_validation"] = score_validation
            
            # TOOL CALL 4: Validate recommendation
            recommendation_validation = self._validate_recommendation(
                agent_result.recommendation,
                test_context.expected_outcomes
            )
            if recommendation_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["recommendation_validation"] = recommendation_validation
            
            # TOOL CALL 5: Validate reasoning factors
            reasoning_validation = self._validate_reasoning_factors(
                agent_result.reasoning_factors,
                test_context.expected_outcomes
            )
            if reasoning_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["reasoning_validation"] = reasoning_validation
            
            # TOOL CALL 6: Validate confidence level
            confidence_validation = self._validate_confidence_level(
                agent_result.confidence_level,
                test_context.expected_outcomes
            )
            if confidence_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["confidence_validation"] = confidence_validation
            
            # TOOL CALL 7: Performance metrics calculation
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_metrics = self._calculate_performance_metrics(
                execution_time,
                agent_result,
                test_context
            )
            
            # TOOL CALL 8: Scenario-specific validations
            scenario_validation = self._validate_scenario_specific(
                test_context.scenario_type,
                agent_result,
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
            test_result = TestResult(
                test_id=test_context.test_id,
                passed=test_passed,
                likelihood_score=agent_result.likelihood_score,
                recommendation=agent_result.recommendation,
                reasoning_factors=agent_result.reasoning_factors,
                confidence_level=agent_result.confidence_level,
                execution_time_ms=execution_time,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                error_message=None,
                performance_metrics=performance_metrics,
                validation_details=validation_details
            )
            
            # Update performance statistics
            self._update_performance_stats(test_result)
            
            logger.info(f"Test completed: {test_context.test_id} - {'PASSED' if test_passed else 'FAILED'}")
            return test_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_result = TestResult(
                test_id=test_context.test_id,
                passed=False,
                likelihood_score=0.0,
                recommendation="error",
                reasoning_factors=[],
                confidence_level="none",
                execution_time_ms=execution_time,
                assertions_passed=0,
                assertions_failed=1,
                error_message=str(e),
                performance_metrics={},
                validation_details={"error": str(e)}
            )
            
            self._update_performance_stats(error_result)
            logger.error(f"Test failed with error: {test_context.test_id} - {str(e)}")
            return error_result
    
    def _validate_likelihood_score(self, score: float, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate likelihood score against expectations"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check score range
            if not (0 <= score <= 100):
                validation["passed"] = False
                validation["errors"].append(f"Score {score} not in valid range [0-100]")
            else:
                validation["details"].append(f"Score {score} in valid range")
            
            # Check expected range if provided
            if "score_range" in expected:
                min_score, max_score = expected["score_range"]
                if not (min_score <= score <= max_score):
                    validation["passed"] = False
                    validation["errors"].append(f"Score {score} not in expected range [{min_score}-{max_score}]")
                else:
                    validation["details"].append(f"Score {score} in expected range [{min_score}-{max_score}]")
            
            # Check minimum score if provided
            if "min_score" in expected:
                if score < expected["min_score"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Score {score} below minimum {expected['min_score']}")
                else:
                    validation["details"].append(f"Score {score} meets minimum {expected['min_score']}")
            
            # Check maximum score if provided
            if "max_score" in expected:
                if score > expected["max_score"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Score {score} above maximum {expected['max_score']}")
                else:
                    validation["details"].append(f"Score {score} below maximum {expected['max_score']}")
                    
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_recommendation(self, recommendation: str, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate recommendation against expectations"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            valid_recommendations = ["prompt_now", "prompt_later", "skip_prompt", "wait_longer"]
            
            # Check if recommendation is valid
            if recommendation not in valid_recommendations:
                validation["passed"] = False
                validation["errors"].append(f"Invalid recommendation: {recommendation}")
            else:
                validation["details"].append(f"Valid recommendation: {recommendation}")
            
            # Check expected recommendation if provided
            if "expected_recommendation" in expected:
                if recommendation != expected["expected_recommendation"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Expected {expected['expected_recommendation']}, got {recommendation}")
                else:
                    validation["details"].append(f"Recommendation matches expected: {recommendation}")
            
            # Check recommendation category if provided
            if "recommendation_category" in expected:
                expected_category = expected["recommendation_category"]
                if expected_category == "positive" and recommendation not in ["prompt_now", "prompt_later"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Expected positive recommendation, got {recommendation}")
                elif expected_category == "negative" and recommendation not in ["skip_prompt", "wait_longer"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Expected negative recommendation, got {recommendation}")
                else:
                    validation["details"].append(f"Recommendation category matches expected: {expected_category}")
                    
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_reasoning_factors(self, factors: List[str], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate reasoning factors against expectations"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check if factors exist
            if not factors:
                validation["passed"] = False
                validation["errors"].append("No reasoning factors provided")
            else:
                validation["details"].append(f"Found {len(factors)} reasoning factors")
            
            # Check required factors if provided
            if "required_factors" in expected:
                for required_factor in expected["required_factors"]:
                    if required_factor not in factors:
                        validation["passed"] = False
                        validation["errors"].append(f"Missing required factor: {required_factor}")
                    else:
                        validation["details"].append(f"Found required factor: {required_factor}")
            
            # Check forbidden factors if provided
            if "forbidden_factors" in expected:
                for forbidden_factor in expected["forbidden_factors"]:
                    if forbidden_factor in factors:
                        validation["passed"] = False
                        validation["errors"].append(f"Found forbidden factor: {forbidden_factor}")
                    else:
                        validation["details"].append(f"Forbidden factor not found: {forbidden_factor}")
            
            # Check minimum number of factors if provided
            if "min_factors" in expected:
                if len(factors) < expected["min_factors"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Only {len(factors)} factors, expected at least {expected['min_factors']}")
                else:
                    validation["details"].append(f"Sufficient factors: {len(factors)} >= {expected['min_factors']}")
                    
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_confidence_level(self, confidence: str, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate confidence level against expectations"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            valid_levels = ["low", "medium", "high"]
            
            # Check if confidence level is valid
            if confidence not in valid_levels:
                validation["passed"] = False
                validation["errors"].append(f"Invalid confidence level: {confidence}")
            else:
                validation["details"].append(f"Valid confidence level: {confidence}")
            
            # Check expected confidence if provided
            if "expected_confidence" in expected:
                if confidence != expected["expected_confidence"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Expected {expected['expected_confidence']}, got {confidence}")
                else:
                    validation["details"].append(f"Confidence matches expected: {confidence}")
            
            # Check minimum confidence if provided
            if "min_confidence" in expected:
                confidence_order = {"low": 1, "medium": 2, "high": 3}
                if confidence_order[confidence] < confidence_order[expected["min_confidence"]]:
                    validation["passed"] = False
                    validation["errors"].append(f"Confidence {confidence} below minimum {expected['min_confidence']}")
                else:
                    validation["details"].append(f"Confidence {confidence} meets minimum {expected['min_confidence']}")
                    
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_scenario_specific(self, scenario: TestScenario, result: Any, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scenario-specific requirements"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            if scenario == TestScenario.HIGH_LIKELIHOOD:
                # High likelihood scenarios should have high scores and positive recommendations
                if result.likelihood_score < 70:
                    validation["passed"] = False
                    validation["errors"].append(f"High likelihood scenario should have score >= 70, got {result.likelihood_score}")
                
                if result.recommendation not in ["prompt_now", "prompt_later"]:
                    validation["passed"] = False
                    validation["errors"].append(f"High likelihood scenario should recommend prompting, got {result.recommendation}")
                    
            elif scenario == TestScenario.LOW_LIKELIHOOD:
                # Low likelihood scenarios should have low scores and negative recommendations
                if result.likelihood_score > 40:
                    validation["passed"] = False
                    validation["errors"].append(f"Low likelihood scenario should have score <= 40, got {result.likelihood_score}")
                
                if result.recommendation not in ["skip_prompt", "wait_longer"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Low likelihood scenario should recommend skipping, got {result.recommendation}")
                    
            elif scenario == TestScenario.EDGE_CASE:
                # Edge cases should still produce valid outputs
                if not (0 <= result.likelihood_score <= 100):
                    validation["passed"] = False
                    validation["errors"].append(f"Edge case produced invalid score: {result.likelihood_score}")
                    
            elif scenario == TestScenario.STRESS_TEST:
                # Stress tests should complete within time limits
                if "max_execution_time" in expected:
                    if hasattr(result, 'execution_time_ms') and result.execution_time_ms > expected["max_execution_time"]:
                        validation["passed"] = False
                        validation["errors"].append(f"Stress test exceeded time limit: {result.execution_time_ms}ms > {expected['max_execution_time']}ms")
            
            if validation["passed"]:
                validation["details"].append(f"Scenario {scenario.value} validation passed")
                
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Scenario validation error: {str(e)}")
        
        return validation
    
    def _calculate_performance_metrics(self, execution_time: float, result: Any, context: TestContext) -> Dict[str, float]:
        """Calculate performance metrics for the test execution"""
        metrics = {
            "execution_time_ms": execution_time,
            "score_validity": 1.0 if 0 <= result.likelihood_score <= 100 else 0.0,
            "recommendation_validity": 1.0 if result.recommendation in ["prompt_now", "prompt_later", "skip_prompt", "wait_longer"] else 0.0,
            "reasoning_completeness": min(1.0, len(result.reasoning_factors) / 3.0),  # Expect at least 3 factors
            "confidence_appropriateness": 1.0 if result.confidence_level in ["low", "medium", "high"] else 0.0
        }
        
        # Calculate overall quality score
        quality_factors = [
            metrics["score_validity"],
            metrics["recommendation_validity"], 
            metrics["reasoning_completeness"],
            metrics["confidence_appropriateness"]
        ]
        metrics["overall_quality"] = sum(quality_factors) / len(quality_factors)
        
        return metrics
    
    def _update_performance_stats(self, result: TestResult):
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
        
        # Add to test history
        self.test_history.append(result)
    
    def run_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite for donation likelihood agent"""
        logger.info("Starting donation likelihood test suite")
        
        test_scenarios = [
            # High likelihood scenarios
            TestContext(
                test_id="high_likelihood_large_transaction",
                scenario_type=TestScenario.HIGH_LIKELIHOOD,
                user_context={
                    "user_id": "test_user_1",
                    "transaction_amount": 1000.0,
                    "wallet_balance": 5000.0,
                    "location": {"region": "NCR"},
                    "transaction_category": "groceries",
                    "days_since_last_prompt": 10,
                    "days_since_last_donation": 30,
                    "average_donation_amount": 5.0,
                    "total_lifetime_donations": 50.0,
                    "preferred_causes": ["Health"],
                    "notification_preferences": {"impact_updates": True},
                    "demographic_hints": {"age_range": "25-34"}
                },
                expected_outcomes={
                    "min_score": 70,
                    "recommendation_category": "positive",
                    "required_factors": ["large_transaction_generosity", "healthy_wallet_balance"],
                    "min_confidence": "medium"
                },
                test_parameters={},
                metadata={"description": "Large transaction with healthy balance should trigger high likelihood"}
            ),
            
            # Low likelihood scenarios
            TestContext(
                test_id="low_likelihood_poor_conditions",
                scenario_type=TestScenario.LOW_LIKELIHOOD,
                user_context={
                    "user_id": "test_user_2",
                    "transaction_amount": 25.0,
                    "wallet_balance": 50.0,
                    "location": {"region": "NCR"},
                    "transaction_category": "entertainment",
                    "days_since_last_prompt": 1,
                    "days_since_last_donation": 5,
                    "average_donation_amount": 0.0,
                    "total_lifetime_donations": 0.0,
                    "preferred_causes": [],
                    "notification_preferences": {},
                    "demographic_hints": {}
                },
                expected_outcomes={
                    "max_score": 40,
                    "recommendation_category": "negative",
                    "required_factors": ["low_wallet_balance", "too_recent_prompt"],
                    "min_factors": 2
                },
                test_parameters={},
                metadata={"description": "Poor financial conditions should result in low likelihood"}
            ),
            
            # Edge case scenarios
            TestContext(
                test_id="edge_case_zero_balance",
                scenario_type=TestScenario.EDGE_CASE,
                user_context={
                    "user_id": "test_user_3",
                    "transaction_amount": 0.0,
                    "wallet_balance": 0.0,
                    "location": {"region": "Unknown"},
                    "transaction_category": "",
                    "days_since_last_prompt": 0,
                    "days_since_last_donation": 0,
                    "average_donation_amount": 0.0,
                    "total_lifetime_donations": 0.0,
                    "preferred_causes": [],
                    "notification_preferences": {},
                    "demographic_hints": {}
                },
                expected_outcomes={
                    "score_range": [0, 100],
                    "expected_recommendation": "skip_prompt"
                },
                test_parameters={},
                metadata={"description": "Zero values should be handled gracefully"}
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
            "results": results,
            "performance_stats": self.performance_stats
        }
        
        logger.info(f"Test suite completed. Pass rate: {summary['pass_rate']:.2%}")
        return summary

def run_donation_likelihood_tests():
    """Entry point for running donation likelihood tests"""
    test_agent = DonationLikelihoodTestAgent()
    return test_agent.run_test_suite()

if __name__ == "__main__":
    results = run_donation_likelihood_tests()
    print(json.dumps(results, indent=2, default=str))
