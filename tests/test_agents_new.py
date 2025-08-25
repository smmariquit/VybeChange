"""
Legacy Testing Framework for ImpactSense AI Agents
This file now serves as a bridge to the new individual test agent architecture
All agents now follow the input-tool call-output pattern with dedicated test files

New Architecture:
- tests/donation_likelihood_test_agent.py: Individual agent testing with input-tool call-output pattern
- tests/cause_recommendation_test_agent.py: Individual agent testing with input-tool call-output pattern  
- tests/amount_optimization_test_agent.py: Individual agent testing with input-tool call-output pattern
- tests/performance_tracking_test_agent.py: Individual agent testing with input-tool call-output pattern
- tests/test_orchestrator.py: Coordinates all test agents and provides comprehensive reporting
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_legacy_tests() -> Dict[str, Any]:
    """
    Run tests using the new individual test agent architecture
    This function maintains backward compatibility while using the new structure
    """
    logger.info("Running tests using new agent architecture...")
    
    try:
        # Import the new test orchestrator
        from tests.test_orchestrator import run_all_tests
        
        # Run comprehensive test suite
        results = run_all_tests(parallel=True, detailed=True)
        
        # Convert to legacy format for compatibility
        legacy_results = {
            "test_summary": {
                "total_tests": results.total_tests_run,
                "passed_tests": results.total_tests_passed,
                "failed_tests": results.total_tests_failed,
                "pass_rate": results.overall_pass_rate,
                "execution_time_ms": results.execution_time_ms
            },
            "agent_results": results.suite_results,
            "performance_metrics": results.performance_summary,
            "recommendations": results.recommendations,
            "execution_successful": results.execution_successful,
            "timestamp": datetime.now().isoformat(),
            "architecture": "individual_agents_with_input_tool_output_pattern"
        }
        
        logger.info(f"Tests completed successfully. Pass rate: {results.overall_pass_rate:.2%}")
        return legacy_results
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return {
            "test_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 1,
                "pass_rate": 0.0,
                "execution_time_ms": 0
            },
            "error": str(e),
            "execution_successful": False,
            "timestamp": datetime.now().isoformat()
        }

def run_donation_likelihood_tests():
    """Run donation likelihood tests using new architecture"""
    try:
        from tests.donation_likelihood_test_agent import run_donation_likelihood_tests
        return run_donation_likelihood_tests()
    except Exception as e:
        logger.error(f"Error running donation likelihood tests: {str(e)}")
        return {"error": str(e)}

def run_cause_recommendation_tests():
    """Run cause recommendation tests using new architecture"""
    try:
        from tests.cause_recommendation_test_agent import run_cause_recommendation_tests
        return run_cause_recommendation_tests()
    except Exception as e:
        logger.error(f"Error running cause recommendation tests: {str(e)}")
        return {"error": str(e)}

def run_amount_optimization_tests():
    """Run amount optimization tests using new architecture"""
    try:
        from tests.amount_optimization_test_agent import run_amount_optimization_tests
        return run_amount_optimization_tests()
    except Exception as e:
        logger.error(f"Error running amount optimization tests: {str(e)}")
        return {"error": str(e)}

def run_performance_tracking_tests():
    """Run performance tracking tests using new architecture"""
    try:
        from tests.performance_tracking_test_agent import run_performance_tracking_tests
        return run_performance_tracking_tests()
    except Exception as e:
        logger.error(f"Error running performance tracking tests: {str(e)}")
        return {"error": str(e)}

def run_quick_tests():
    """Run quick test suite using new architecture"""
    try:
        from tests.test_orchestrator import run_quick_tests
        return run_quick_tests()
    except Exception as e:
        logger.error(f"Error running quick tests: {str(e)}")
        return {"error": str(e)}

# Legacy compatibility functions
class TestDonationLikelyScoreAgent:
    """Legacy test class - redirects to new architecture"""
    
    def test_high_likelihood_scenario(self):
        """Legacy test - use new architecture instead"""
        logger.warning("Using legacy test interface - consider migrating to new architecture")
        results = run_donation_likelihood_tests()
        assert results.get("pass_rate", 0) > 0.8, "Donation likelihood tests should have high pass rate"
    
    def test_low_likelihood_scenario(self):
        """Legacy test - use new architecture instead"""
        logger.warning("Using legacy test interface - consider migrating to new architecture")
        results = run_donation_likelihood_tests()
        assert "error" not in results, "Donation likelihood tests should not error"

class TestLocalCauseRecommenderAgent:
    """Legacy test class - redirects to new architecture"""
    
    def test_exact_region_match_priority(self):
        """Legacy test - use new architecture instead"""
        logger.warning("Using legacy test interface - consider migrating to new architecture")
        results = run_cause_recommendation_tests()
        assert results.get("pass_rate", 0) > 0.8, "Cause recommendation tests should have high pass rate"
    
    def test_category_preference_matching(self):
        """Legacy test - use new architecture instead"""
        logger.warning("Using legacy test interface - consider migrating to new architecture")
        results = run_cause_recommendation_tests()
        assert "error" not in results, "Cause recommendation tests should not error"

class TestDonationAmountOptimizerAgent:
    """Legacy test class - redirects to new architecture"""
    
    def test_affordability_constraints(self):
        """Legacy test - use new architecture instead"""
        logger.warning("Using legacy test interface - consider migrating to new architecture")
        results = run_amount_optimization_tests()
        assert results.get("pass_rate", 0) > 0.8, "Amount optimization tests should have high pass rate"
    
    def test_psychology_based_optimization(self):
        """Legacy test - use new architecture instead"""
        logger.warning("Using legacy test interface - consider migrating to new architecture")
        results = run_amount_optimization_tests()
        assert "error" not in results, "Amount optimization tests should not error"

class TestImpactSenseOrchestrator:
    """Legacy test class - redirects to new architecture"""
    
    def test_full_pipeline_integration(self):
        """Legacy test - use new architecture instead"""
        logger.warning("Using legacy test interface - consider migrating to new architecture")
        results = run_legacy_tests()
        assert results.get("execution_successful", False), "Full pipeline should execute successfully"
    
    def test_agent_coordination(self):
        """Legacy test - use new architecture instead"""
        logger.warning("Using legacy test interface - consider migrating to new architecture")
        results = run_legacy_tests()
        assert results["test_summary"]["total_tests"] > 0, "Should execute multiple tests"

if __name__ == "__main__":
    print("=== Running Legacy Test Interface ===")
    print("Note: This interface redirects to the new individual agent architecture")
    print("For better testing, use the new test agents directly:")
    print("- tests/donation_likelihood_test_agent.py")
    print("- tests/cause_recommendation_test_agent.py") 
    print("- tests/amount_optimization_test_agent.py")
    print("- tests/performance_tracking_test_agent.py")
    print("- tests/test_orchestrator.py")
    print()
    
    # Run tests using new architecture
    results = run_legacy_tests()
    
    print("=== Test Results ===")
    print(json.dumps(results, indent=2, default=str))
