"""
Amount Optimization Test Agent
Comprehensive testing agent for donation amount optimization with input-tool call-output pattern
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
    AFFORDABILITY_ANALYSIS = "affordability_analysis"
    PSYCHOLOGICAL_OPTIMIZATION = "psychological_optimization"
    BEHAVIORAL_TARGETING = "behavioral_targeting"
    EDGE_CASE = "edge_case"
    PERFORMANCE_TEST = "performance_test"

@dataclass
class AmountTestContext:
    """Input context for amount optimization testing"""
    test_id: str
    scenario_type: TestScenario
    financial_context: Dict[str, Any]
    behavioral_profile: Dict[str, Any]
    expected_outcomes: Dict[str, Any]
    test_parameters: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class AmountTestResult:
    """Output result from amount optimization testing"""
    test_id: str
    passed: bool
    recommended_amounts: Dict[str, float]
    affordability_score: float
    psychological_factors: List[str]
    conversion_probability: float
    execution_time_ms: float
    assertions_passed: int
    assertions_failed: int
    error_message: Optional[str]
    performance_metrics: Dict[str, float]
    validation_details: Dict[str, Any]

class AmountOptimizationTestAgent:
    """
    Test agent for donation amount optimization functionality
    Follows input-tool call-output pattern with comprehensive validation
    """
    
    def __init__(self):
        self.agent_id = "amount_optimization_test_agent"
        self.test_history = []
        self.performance_stats = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "average_execution_time": 0.0,
            "optimization_accuracy": 0.0
        }
        
    def execute_test(self, test_context: AmountTestContext) -> AmountTestResult:
        """
        Execute amount optimization test with input-tool call-output pattern
        
        Input: AmountTestContext with financial and behavioral data
        Tool Calls: Optimization validation and analysis tools
        Output: AmountTestResult with comprehensive assessment
        """
        start_time = datetime.now()
        logger.info(f"Executing amount test: {test_context.test_id} - {test_context.scenario_type.value}")
        
        try:
            # Import the agent under test
            from src.agents.amount_optimization_agent import AmountOptimizationAgent, FinancialContext, BehavioralProfile, OptimizedAmounts
            
            # Initialize test metrics
            assertions_passed = 0
            assertions_failed = 0
            validation_details = {}
            
            # TOOL CALL 1: Setup test environment
            test_agent = AmountOptimizationAgent()
            financial_context = FinancialContext(**test_context.financial_context)
            behavioral_profile = BehavioralProfile(**test_context.behavioral_profile)
            
            # TOOL CALL 2: Execute agent optimization
            agent_result = test_agent.optimize_amounts(financial_context, behavioral_profile)
            
            # TOOL CALL 3: Validate recommended amounts
            amounts_validation = self._validate_recommended_amounts(
                agent_result.recommended_amounts,
                financial_context,
                test_context.expected_outcomes
            )
            if amounts_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["amounts_validation"] = amounts_validation
            
            # TOOL CALL 4: Validate affordability score
            affordability_validation = self._validate_affordability_score(
                agent_result.affordability_score,
                financial_context,
                test_context.expected_outcomes
            )
            if affordability_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["affordability_validation"] = affordability_validation
            
            # TOOL CALL 5: Validate psychological factors
            psychology_validation = self._validate_psychological_factors(
                agent_result.psychological_factors,
                behavioral_profile,
                test_context.expected_outcomes
            )
            if psychology_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["psychology_validation"] = psychology_validation
            
            # TOOL CALL 6: Validate conversion probability
            conversion_validation = self._validate_conversion_probability(
                agent_result.conversion_probability,
                test_context.expected_outcomes
            )
            if conversion_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["conversion_validation"] = conversion_validation
            
            # TOOL CALL 7: Validate amount progression logic
            progression_validation = self._validate_amount_progression(
                agent_result.recommended_amounts,
                test_context.expected_outcomes
            )
            if progression_validation["passed"]:
                assertions_passed += 1
            else:
                assertions_failed += 1
            validation_details["progression_validation"] = progression_validation
            
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
                financial_context,
                behavioral_profile,
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
            test_result = AmountTestResult(
                test_id=test_context.test_id,
                passed=test_passed,
                recommended_amounts=agent_result.recommended_amounts,
                affordability_score=agent_result.affordability_score,
                psychological_factors=agent_result.psychological_factors,
                conversion_probability=agent_result.conversion_probability,
                execution_time_ms=execution_time,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
                error_message=None,
                performance_metrics=performance_metrics,
                validation_details=validation_details
            )
            
            # Update performance statistics
            self._update_performance_stats(test_result)
            
            logger.info(f"Amount test completed: {test_context.test_id} - {'PASSED' if test_passed else 'FAILED'}")
            return test_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_result = AmountTestResult(
                test_id=test_context.test_id,
                passed=False,
                recommended_amounts={},
                affordability_score=0.0,
                psychological_factors=[],
                conversion_probability=0.0,
                execution_time_ms=execution_time,
                assertions_passed=0,
                assertions_failed=1,
                error_message=str(e),
                performance_metrics={},
                validation_details={"error": str(e)}
            )
            
            self._update_performance_stats(error_result)
            logger.error(f"Amount test failed with error: {test_context.test_id} - {str(e)}")
            return error_result
    
    def _validate_recommended_amounts(self, amounts: Dict[str, float], financial_context: Any, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate recommended amounts structure and values"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check if amounts exist
            if not amounts:
                validation["passed"] = False
                validation["errors"].append("No recommended amounts provided")
                return validation
            
            validation["details"].append(f"Found {len(amounts)} recommended amounts")
            
            # Check required amount types
            required_types = ["suggested", "minimum", "maximum"]
            for amount_type in required_types:
                if amount_type not in amounts:
                    validation["passed"] = False
                    validation["errors"].append(f"Missing required amount type: {amount_type}")
                else:
                    validation["details"].append(f"Amount type {amount_type} present")
            
            # Check amount values are positive
            for amount_type, amount in amounts.items():
                if amount <= 0:
                    validation["passed"] = False
                    validation["errors"].append(f"Amount {amount_type} should be positive, got {amount}")
                else:
                    validation["details"].append(f"Amount {amount_type} is positive: {amount}")
            
            # Check amount hierarchy (minimum <= suggested <= maximum)
            if "minimum" in amounts and "suggested" in amounts and "maximum" in amounts:
                if not (amounts["minimum"] <= amounts["suggested"] <= amounts["maximum"]):
                    validation["passed"] = False
                    validation["errors"].append(f"Amount hierarchy violated: min={amounts['minimum']}, suggested={amounts['suggested']}, max={amounts['maximum']}")
                else:
                    validation["details"].append("Amount hierarchy is correct")
            
            # Check affordability constraints
            if hasattr(financial_context, 'disposable_income'):
                max_affordable = financial_context.disposable_income * 0.1  # 10% rule
                if amounts.get("maximum", 0) > max_affordable:
                    validation["passed"] = False
                    validation["errors"].append(f"Maximum amount {amounts['maximum']} exceeds 10% of disposable income {max_affordable}")
                else:
                    validation["details"].append("Amounts respect affordability constraints")
            
            # Check expected amount ranges if provided
            if "expected_amount_ranges" in expected:
                for amount_type, (min_val, max_val) in expected["expected_amount_ranges"].items():
                    if amount_type in amounts:
                        if not (min_val <= amounts[amount_type] <= max_val):
                            validation["passed"] = False
                            validation["errors"].append(f"Amount {amount_type} {amounts[amount_type]} not in expected range [{min_val}-{max_val}]")
                        else:
                            validation["details"].append(f"Amount {amount_type} in expected range")
                            
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_affordability_score(self, score: float, financial_context: Any, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate affordability score"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check score range
            if not (0 <= score <= 1):
                validation["passed"] = False
                validation["errors"].append(f"Affordability score {score} not in valid range [0-1]")
            else:
                validation["details"].append(f"Affordability score {score} in valid range")
            
            # Check score logic based on financial situation
            if hasattr(financial_context, 'disposable_income') and hasattr(financial_context, 'monthly_income'):
                income_ratio = financial_context.disposable_income / financial_context.monthly_income
                expected_score_range = (income_ratio * 0.5, income_ratio * 1.5)  # Rough expectation
                
                if not (expected_score_range[0] <= score <= expected_score_range[1]):
                    # This is a warning, not a failure, as affordability calculation can be complex
                    validation["details"].append(f"Affordability score {score} may not align with income ratio {income_ratio}")
                else:
                    validation["details"].append("Affordability score aligns with financial situation")
            
            # Check minimum score if provided
            if "min_affordability_score" in expected:
                if score < expected["min_affordability_score"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Affordability score {score} below minimum {expected['min_affordability_score']}")
                else:
                    validation["details"].append(f"Affordability score meets minimum: {score} >= {expected['min_affordability_score']}")
                    
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_psychological_factors(self, factors: List[str], behavioral_profile: Any, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate psychological factors"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check if factors exist
            if not factors:
                validation["passed"] = False
                validation["errors"].append("No psychological factors provided")
            else:
                validation["details"].append(f"Found {len(factors)} psychological factors")
            
            # Check for valid psychological factors
            valid_factors = [
                "anchoring_effect", "loss_aversion", "social_proof", "reciprocity",
                "commitment_consistency", "scarcity", "authority", "round_numbers",
                "mental_accounting", "framing_effect", "endowment_effect"
            ]
            
            for factor in factors:
                if factor not in valid_factors:
                    validation["passed"] = False
                    validation["errors"].append(f"Invalid psychological factor: {factor}")
                else:
                    validation["details"].append(f"Valid psychological factor: {factor}")
            
            # Check required factors if provided
            if "required_psychological_factors" in expected:
                for required_factor in expected["required_psychological_factors"]:
                    if required_factor not in factors:
                        validation["passed"] = False
                        validation["errors"].append(f"Missing required psychological factor: {required_factor}")
                    else:
                        validation["details"].append(f"Found required psychological factor: {required_factor}")
            
            # Check behavioral profile alignment
            if hasattr(behavioral_profile, 'giving_personality'):
                personality = behavioral_profile.giving_personality
                expected_factors_by_personality = {
                    "analytical": ["anchoring_effect", "loss_aversion"],
                    "emotional": ["social_proof", "reciprocity"],
                    "practical": ["round_numbers", "mental_accounting"],
                    "altruistic": ["commitment_consistency", "authority"]
                }
                
                if personality in expected_factors_by_personality:
                    expected_factors = expected_factors_by_personality[personality]
                    found_expected = any(factor in factors for factor in expected_factors)
                    if not found_expected:
                        validation["details"].append(f"No personality-aligned factors found for {personality}")
                    else:
                        validation["details"].append(f"Found personality-aligned factors for {personality}")
                        
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_conversion_probability(self, probability: float, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate conversion probability"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check probability range
            if not (0 <= probability <= 1):
                validation["passed"] = False
                validation["errors"].append(f"Conversion probability {probability} not in valid range [0-1]")
            else:
                validation["details"].append(f"Conversion probability {probability} in valid range")
            
            # Check minimum probability if provided
            if "min_conversion_probability" in expected:
                if probability < expected["min_conversion_probability"]:
                    validation["passed"] = False
                    validation["errors"].append(f"Conversion probability {probability} below minimum {expected['min_conversion_probability']}")
                else:
                    validation["details"].append(f"Conversion probability meets minimum: {probability} >= {expected['min_conversion_probability']}")
            
            # Check probability range if provided
            if "conversion_probability_range" in expected:
                min_prob, max_prob = expected["conversion_probability_range"]
                if not (min_prob <= probability <= max_prob):
                    validation["passed"] = False
                    validation["errors"].append(f"Conversion probability {probability} not in expected range [{min_prob}-{max_prob}]")
                else:
                    validation["details"].append(f"Conversion probability in expected range")
                    
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_amount_progression(self, amounts: Dict[str, float], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate amount progression logic"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            # Check if multiple amounts create a logical progression
            if "suggested" in amounts and "minimum" in amounts and "maximum" in amounts:
                min_amt, sug_amt, max_amt = amounts["minimum"], amounts["suggested"], amounts["maximum"]
                
                # Check reasonable ratios
                if sug_amt / min_amt > 5:  # Suggested shouldn't be more than 5x minimum
                    validation["passed"] = False
                    validation["errors"].append(f"Suggested amount {sug_amt} too high relative to minimum {min_amt}")
                
                if max_amt / sug_amt > 3:  # Maximum shouldn't be more than 3x suggested
                    validation["passed"] = False
                    validation["errors"].append(f"Maximum amount {max_amt} too high relative to suggested {sug_amt}")
                
                if validation["passed"]:
                    validation["details"].append("Amount progression ratios are reasonable")
            
            # Check round number preferences if expected
            if "expect_round_numbers" in expected and expected["expect_round_numbers"]:
                for amount_type, amount in amounts.items():
                    # Check if amount is a "round" number (ending in 0 or 5, or common donation amounts)
                    is_round = (amount % 10 == 0 or amount % 5 == 0 or 
                              amount in [1, 2, 3, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 500])
                    if not is_round:
                        validation["details"].append(f"Amount {amount_type} {amount} is not a round number")
                    else:
                        validation["details"].append(f"Amount {amount_type} {amount} is a round number")
                        
        except Exception as e:
            validation["passed"] = False
            validation["errors"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _validate_scenario_specific(self, scenario: TestScenario, result: Any, financial_context: Any, behavioral_profile: Any, expected: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scenario-specific requirements"""
        validation = {
            "passed": True,
            "details": [],
            "errors": []
        }
        
        try:
            if scenario == TestScenario.AFFORDABILITY_ANALYSIS:
                # Should respect affordability constraints strictly
                if result.affordability_score < 0.5:
                    validation["passed"] = False
                    validation["errors"].append(f"Affordability analysis should maintain reasonable affordability score, got {result.affordability_score}")
                else:
                    validation["details"].append("Affordability analysis successful")
                    
            elif scenario == TestScenario.PSYCHOLOGICAL_OPTIMIZATION:
                # Should apply psychological principles
                if len(result.psychological_factors) < 2:
                    validation["passed"] = False
                    validation["errors"].append(f"Psychological optimization should apply multiple factors, got {len(result.psychological_factors)}")
                else:
                    validation["details"].append("Psychological optimization successful")
                    
            elif scenario == TestScenario.BEHAVIORAL_TARGETING:
                # Should align with behavioral profile
                if result.conversion_probability < 0.5:
                    validation["passed"] = False
                    validation["errors"].append(f"Behavioral targeting should achieve reasonable conversion probability, got {result.conversion_probability}")
                else:
                    validation["details"].append("Behavioral targeting successful")
                    
            elif scenario == TestScenario.EDGE_CASE:
                # Should handle edge cases gracefully
                if not result.recommended_amounts or result.conversion_probability == 0:
                    validation["passed"] = False
                    validation["errors"].append("Edge case scenario should still provide reasonable recommendations")
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
    
    def _calculate_performance_metrics(self, execution_time: float, result: Any, context: AmountTestContext) -> Dict[str, float]:
        """Calculate performance metrics for the test execution"""
        metrics = {
            "execution_time_ms": execution_time,
            "amount_completeness": len(result.recommended_amounts) / 3.0,  # Expect 3 amounts
            "affordability_quality": result.affordability_score,
            "psychological_richness": min(1.0, len(result.psychological_factors) / 3.0),  # Expect at least 3 factors
            "conversion_potential": result.conversion_probability
        }
        
        # Calculate amount reasonableness
        if "suggested" in result.recommended_amounts:
            suggested = result.recommended_amounts["suggested"]
            # Reasonable donation range: $1 to $1000
            if 1 <= suggested <= 1000:
                metrics["amount_reasonableness"] = 1.0
            elif suggested < 1:
                metrics["amount_reasonableness"] = max(0.0, suggested)
            else:
                metrics["amount_reasonableness"] = max(0.0, 1 - (suggested - 1000) / 1000)
        else:
            metrics["amount_reasonableness"] = 0.0
        
        # Calculate overall quality score
        quality_factors = [
            metrics["amount_completeness"],
            metrics["affordability_quality"],
            metrics["psychological_richness"],
            metrics["conversion_potential"],
            metrics["amount_reasonableness"]
        ]
        metrics["overall_quality"] = sum(quality_factors) / len(quality_factors)
        
        return metrics
    
    def _update_performance_stats(self, result: AmountTestResult):
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
        
        # Update optimization accuracy
        optimization_score = result.affordability_score * 0.4 + result.conversion_probability * 0.6
        total_accuracy = (self.performance_stats["optimization_accuracy"] * 
                        (self.performance_stats["total_tests"] - 1) + optimization_score)
        self.performance_stats["optimization_accuracy"] = total_accuracy / self.performance_stats["total_tests"]
        
        # Add to test history
        self.test_history.append(result)
    
    def run_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite for amount optimization agent"""
        logger.info("Starting amount optimization test suite")
        
        test_scenarios = [
            # Affordability analysis scenario
            AmountTestContext(
                test_id="affordability_high_income",
                scenario_type=TestScenario.AFFORDABILITY_ANALYSIS,
                financial_context={
                    "user_id": "test_user_1",
                    "monthly_income": 5000.0,
                    "disposable_income": 1500.0,
                    "current_balance": 8000.0,
                    "recent_transactions": [200.0, 150.0, 300.0],
                    "savings_rate": 0.3,
                    "debt_obligations": 800.0,
                    "financial_stability_score": 0.85
                },
                behavioral_profile={
                    "user_id": "test_user_1",
                    "giving_personality": "analytical",
                    "risk_tolerance": "medium",
                    "decision_speed": "deliberate",
                    "price_sensitivity": "medium",
                    "past_donation_amounts": [50.0, 75.0, 100.0, 25.0],
                    "preferred_amount_types": ["round_numbers", "planned_amounts"],
                    "response_to_anchoring": 0.7,
                    "social_influence_factor": 0.5
                },
                expected_outcomes={
                    "min_affordability_score": 0.6,
                    "expected_amount_ranges": {
                        "minimum": (10, 50),
                        "suggested": (50, 150),
                        "maximum": (100, 300)
                    },
                    "min_conversion_probability": 0.5
                },
                test_parameters={},
                metadata={"description": "High income user should have good affordability and reasonable amounts"}
            ),
            
            # Psychological optimization scenario
            AmountTestContext(
                test_id="psychological_optimization_emotional",
                scenario_type=TestScenario.PSYCHOLOGICAL_OPTIMIZATION,
                financial_context={
                    "user_id": "test_user_2",
                    "monthly_income": 3000.0,
                    "disposable_income": 600.0,
                    "current_balance": 2500.0,
                    "recent_transactions": [50.0, 75.0, 100.0],
                    "savings_rate": 0.2,
                    "debt_obligations": 400.0,
                    "financial_stability_score": 0.65
                },
                behavioral_profile={
                    "user_id": "test_user_2",
                    "giving_personality": "emotional",
                    "risk_tolerance": "low",
                    "decision_speed": "impulsive",
                    "price_sensitivity": "high",
                    "past_donation_amounts": [10.0, 15.0, 20.0, 25.0],
                    "preferred_amount_types": ["suggested_amounts", "social_proof"],
                    "response_to_anchoring": 0.8,
                    "social_influence_factor": 0.9
                },
                expected_outcomes={
                    "required_psychological_factors": ["social_proof", "anchoring_effect"],
                    "min_conversion_probability": 0.4,
                    "expect_round_numbers": True
                },
                test_parameters={},
                metadata={"description": "Emotional giver should be influenced by psychological factors"}
            ),
            
            # Edge case scenario
            AmountTestContext(
                test_id="edge_case_low_income",
                scenario_type=TestScenario.EDGE_CASE,
                financial_context={
                    "user_id": "test_user_3",
                    "monthly_income": 800.0,
                    "disposable_income": 50.0,
                    "current_balance": 200.0,
                    "recent_transactions": [20.0, 15.0, 10.0],
                    "savings_rate": 0.05,
                    "debt_obligations": 600.0,
                    "financial_stability_score": 0.25
                },
                behavioral_profile={
                    "user_id": "test_user_3",
                    "giving_personality": "practical",
                    "risk_tolerance": "very_low",
                    "decision_speed": "careful",
                    "price_sensitivity": "very_high",
                    "past_donation_amounts": [1.0, 2.0, 5.0],
                    "preferred_amount_types": ["micro_donations"],
                    "response_to_anchoring": 0.3,
                    "social_influence_factor": 0.2
                },
                expected_outcomes={
                    "expected_amount_ranges": {
                        "minimum": (1, 5),
                        "suggested": (2, 10),
                        "maximum": (5, 20)
                    }
                },
                test_parameters={},
                metadata={"description": "Low income user should get micro-donation suggestions"}
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
            "average_affordability_score": sum(r.affordability_score for r in results) / len(results),
            "average_conversion_probability": sum(r.conversion_probability for r in results) / len(results),
            "results": results,
            "performance_stats": self.performance_stats
        }
        
        logger.info(f"Amount optimization test suite completed. Pass rate: {summary['pass_rate']:.2%}")
        return summary

def run_amount_optimization_tests():
    """Entry point for running amount optimization tests"""
    test_agent = AmountOptimizationTestAgent()
    return test_agent.run_test_suite()

if __name__ == "__main__":
    results = run_amount_optimization_tests()
    print(json.dumps(results, indent=2, default=str))
