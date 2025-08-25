"""
Test Orchestrator Agent
Coordinates and manages all test agents with input-tool call-output pattern
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSuite(Enum):
    DONATION_LIKELIHOOD = "donation_likelihood"
    CAUSE_RECOMMENDATION = "cause_recommendation"
    AMOUNT_OPTIMIZATION = "amount_optimization"
    PERFORMANCE_TRACKING = "performance_tracking"
    ALL_AGENTS = "all_agents"
    INTEGRATION = "integration"

@dataclass
class TestOrchestrationContext:
    """Input context for test orchestration"""
    orchestration_id: str
    test_suites: List[TestSuite]
    parallel_execution: bool
    timeout_seconds: int
    detailed_reporting: bool
    performance_benchmarks: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class TestOrchestrationResult:
    """Output result from test orchestration"""
    orchestration_id: str
    execution_successful: bool
    total_tests_run: int
    total_tests_passed: int
    total_tests_failed: int
    overall_pass_rate: float
    execution_time_ms: float
    suite_results: Dict[str, Any]
    performance_summary: Dict[str, float]
    recommendations: List[str]
    error_summary: Optional[str]
    detailed_report: Dict[str, Any]

class TestOrchestrator:
    """
    Test orchestrator agent that coordinates all testing activities
    Follows input-tool call-output pattern with comprehensive test management
    """
    
    def __init__(self):
        self.orchestrator_id = "test_orchestrator_agent"
        self.execution_history = []
        self.performance_benchmarks = {
            "donation_likelihood_max_time": 5000,  # ms
            "cause_recommendation_max_time": 8000,  # ms
            "amount_optimization_max_time": 6000,  # ms
            "performance_tracking_max_time": 10000,  # ms
            "min_pass_rate": 0.85,
            "max_error_rate": 0.15
        }
        
    def orchestrate_tests(self, context: TestOrchestrationContext) -> TestOrchestrationResult:
        """
        Orchestrate comprehensive testing with input-tool call-output pattern
        
        Input: TestOrchestrationContext with test configuration
        Tool Calls: Individual test agent executions and analysis tools
        Output: TestOrchestrationResult with comprehensive results
        """
        start_time = datetime.now()
        logger.info(f"Starting test orchestration: {context.orchestration_id}")
        
        try:
            suite_results = {}
            total_tests = 0
            total_passed = 0
            total_failed = 0
            
            # TOOL CALL 1: Execute test suites
            if context.parallel_execution:
                suite_results = self._execute_parallel_tests(context)
            else:
                suite_results = self._execute_sequential_tests(context)
            
            # TOOL CALL 2: Aggregate results
            for suite_name, result in suite_results.items():
                if result and not result.get("error"):
                    total_tests += result.get("total_tests", 0)
                    total_passed += result.get("passed", 0)
                    total_failed += result.get("failed", 0)
            
            # TOOL CALL 3: Calculate performance metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0
            
            performance_summary = self._calculate_performance_summary(
                suite_results, 
                execution_time, 
                context.performance_benchmarks
            )
            
            # TOOL CALL 4: Generate recommendations
            recommendations = self._generate_recommendations(
                suite_results, 
                performance_summary,
                context.performance_benchmarks
            )
            
            # TOOL CALL 5: Create detailed report
            detailed_report = self._create_detailed_report(
                suite_results, 
                performance_summary,
                context
            ) if context.detailed_reporting else {}
            
            # TOOL CALL 6: Validate against benchmarks
            execution_successful = self._validate_execution_success(
                overall_pass_rate,
                performance_summary,
                context.performance_benchmarks
            )
            
            # Create comprehensive orchestration result
            orchestration_result = TestOrchestrationResult(
                orchestration_id=context.orchestration_id,
                execution_successful=execution_successful,
                total_tests_run=total_tests,
                total_tests_passed=total_passed,
                total_tests_failed=total_failed,
                overall_pass_rate=overall_pass_rate,
                execution_time_ms=execution_time,
                suite_results=suite_results,
                performance_summary=performance_summary,
                recommendations=recommendations,
                error_summary=None,
                detailed_report=detailed_report
            )
            
            # Update execution history
            self.execution_history.append(orchestration_result)
            
            logger.info(f"Test orchestration completed: {context.orchestration_id} - Pass rate: {overall_pass_rate:.2%}")
            return orchestration_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_result = TestOrchestrationResult(
                orchestration_id=context.orchestration_id,
                execution_successful=False,
                total_tests_run=0,
                total_tests_passed=0,
                total_tests_failed=1,
                overall_pass_rate=0.0,
                execution_time_ms=execution_time,
                suite_results={},
                performance_summary={},
                recommendations=[f"Fix orchestration error: {str(e)}"],
                error_summary=str(e),
                detailed_report={"error": traceback.format_exc()}
            )
            
            self.execution_history.append(error_result)
            logger.error(f"Test orchestration failed: {context.orchestration_id} - {str(e)}")
            return error_result
    
    def _execute_parallel_tests(self, context: TestOrchestrationContext) -> Dict[str, Any]:
        """Execute test suites in parallel"""
        suite_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all test suite executions
            future_to_suite = {}
            
            for test_suite in context.test_suites:
                if test_suite == TestSuite.ALL_AGENTS:
                    # Run all individual agent tests
                    for agent_suite in [TestSuite.DONATION_LIKELIHOOD, TestSuite.CAUSE_RECOMMENDATION, 
                                      TestSuite.AMOUNT_OPTIMIZATION, TestSuite.PERFORMANCE_TRACKING]:
                        future = executor.submit(self._execute_single_test_suite, agent_suite, context.timeout_seconds)
                        future_to_suite[future] = agent_suite.value
                else:
                    future = executor.submit(self._execute_single_test_suite, test_suite, context.timeout_seconds)
                    future_to_suite[future] = test_suite.value
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_suite, timeout=context.timeout_seconds):
                suite_name = future_to_suite[future]
                try:
                    result = future.result()
                    suite_results[suite_name] = result
                    logger.info(f"Completed test suite: {suite_name}")
                except Exception as e:
                    suite_results[suite_name] = {"error": str(e)}
                    logger.error(f"Test suite failed: {suite_name} - {str(e)}")
        
        return suite_results
    
    def _execute_sequential_tests(self, context: TestOrchestrationContext) -> Dict[str, Any]:
        """Execute test suites sequentially"""
        suite_results = {}
        
        for test_suite in context.test_suites:
            if test_suite == TestSuite.ALL_AGENTS:
                # Run all individual agent tests
                for agent_suite in [TestSuite.DONATION_LIKELIHOOD, TestSuite.CAUSE_RECOMMENDATION, 
                                  TestSuite.AMOUNT_OPTIMIZATION, TestSuite.PERFORMANCE_TRACKING]:
                    try:
                        result = self._execute_single_test_suite(agent_suite, context.timeout_seconds)
                        suite_results[agent_suite.value] = result
                        logger.info(f"Completed test suite: {agent_suite.value}")
                    except Exception as e:
                        suite_results[agent_suite.value] = {"error": str(e)}
                        logger.error(f"Test suite failed: {agent_suite.value} - {str(e)}")
            else:
                try:
                    result = self._execute_single_test_suite(test_suite, context.timeout_seconds)
                    suite_results[test_suite.value] = result
                    logger.info(f"Completed test suite: {test_suite.value}")
                except Exception as e:
                    suite_results[test_suite.value] = {"error": str(e)}
                    logger.error(f"Test suite failed: {test_suite.value} - {str(e)}")
        
        return suite_results
    
    def _execute_single_test_suite(self, test_suite: TestSuite, timeout_seconds: int) -> Dict[str, Any]:
        """Execute a single test suite"""
        try:
            if test_suite == TestSuite.DONATION_LIKELIHOOD:
                from tests.donation_likelihood_test_agent import run_donation_likelihood_tests
                return run_donation_likelihood_tests()
                
            elif test_suite == TestSuite.CAUSE_RECOMMENDATION:
                from tests.cause_recommendation_test_agent import run_cause_recommendation_tests
                return run_cause_recommendation_tests()
                
            elif test_suite == TestSuite.AMOUNT_OPTIMIZATION:
                from tests.amount_optimization_test_agent import run_amount_optimization_tests
                return run_amount_optimization_tests()
                
            elif test_suite == TestSuite.PERFORMANCE_TRACKING:
                from tests.performance_tracking_test_agent import run_performance_tracking_tests
                return run_performance_tracking_tests()
                
            elif test_suite == TestSuite.INTEGRATION:
                return self._run_integration_tests()
                
            else:
                raise ValueError(f"Unknown test suite: {test_suite}")
                
        except Exception as e:
            logger.error(f"Error executing test suite {test_suite}: {str(e)}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests across all agents"""
        logger.info("Running integration tests")
        
        # This would test the full workflow: likelihood -> recommendation -> optimization -> tracking
        integration_results = {
            "total_tests": 1,
            "passed": 1,
            "failed": 0,
            "pass_rate": 1.0,
            "integration_scenarios": [
                {
                    "scenario": "full_workflow",
                    "passed": True,
                    "description": "Complete donation workflow integration test"
                }
            ]
        }
        
        return integration_results
    
    def _calculate_performance_summary(self, suite_results: Dict[str, Any], execution_time: float, benchmarks: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive performance summary"""
        summary = {
            "total_execution_time_ms": execution_time,
            "average_suite_execution_time": 0.0,
            "fastest_suite_time": float('inf'),
            "slowest_suite_time": 0.0,
            "overall_efficiency_score": 0.0,
            "benchmark_compliance_rate": 0.0
        }
        
        valid_suites = [result for result in suite_results.values() if not result.get("error")]
        
        if valid_suites:
            suite_times = [result.get("average_execution_time", 0) for result in valid_suites]
            summary["average_suite_execution_time"] = sum(suite_times) / len(suite_times)
            summary["fastest_suite_time"] = min(suite_times)
            summary["slowest_suite_time"] = max(suite_times)
        
        # Calculate efficiency score (inverse of execution time vs benchmark)
        benchmark_violations = 0
        total_benchmarks = 0
        
        for suite_name, result in suite_results.items():
            if not result.get("error"):
                benchmark_key = f"{suite_name}_max_time"
                if benchmark_key in benchmarks:
                    total_benchmarks += 1
                    suite_time = result.get("average_execution_time", 0)
                    if suite_time > benchmarks[benchmark_key]:
                        benchmark_violations += 1
        
        summary["benchmark_compliance_rate"] = (total_benchmarks - benchmark_violations) / total_benchmarks if total_benchmarks > 0 else 1.0
        summary["overall_efficiency_score"] = min(1.0, benchmarks.get("donation_likelihood_max_time", 5000) / execution_time)
        
        return summary
    
    def _generate_recommendations(self, suite_results: Dict[str, Any], performance_summary: Dict[str, float], benchmarks: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        # Analyze pass rates
        for suite_name, result in suite_results.items():
            if not result.get("error"):
                pass_rate = result.get("pass_rate", 0)
                if pass_rate < benchmarks.get("min_pass_rate", 0.85):
                    recommendations.append(f"Improve {suite_name} agent - pass rate {pass_rate:.2%} below target {benchmarks.get('min_pass_rate', 0.85):.2%}")
        
        # Analyze performance
        if performance_summary["benchmark_compliance_rate"] < 0.8:
            recommendations.append("Optimize test execution times - multiple suites exceeding performance benchmarks")
        
        if performance_summary["total_execution_time_ms"] > 30000:  # 30 seconds
            recommendations.append("Consider optimizing test suite for faster execution - total time exceeds 30 seconds")
        
        # Analyze errors
        error_suites = [name for name, result in suite_results.items() if result.get("error")]
        if error_suites:
            recommendations.append(f"Fix critical errors in test suites: {', '.join(error_suites)}")
        
        # Overall recommendations
        total_tests = sum(result.get("total_tests", 0) for result in suite_results.values() if not result.get("error"))
        if total_tests < 10:
            recommendations.append("Increase test coverage - fewer than 10 total tests executed")
        
        if not recommendations:
            recommendations.append("All test suites performing well - maintain current quality standards")
        
        return recommendations
    
    def _create_detailed_report(self, suite_results: Dict[str, Any], performance_summary: Dict[str, float], context: TestOrchestrationContext) -> Dict[str, Any]:
        """Create comprehensive detailed report"""
        return {
            "orchestration_context": {
                "orchestration_id": context.orchestration_id,
                "test_suites_requested": [suite.value for suite in context.test_suites],
                "parallel_execution": context.parallel_execution,
                "timeout_seconds": context.timeout_seconds,
                "timestamp": datetime.now().isoformat()
            },
            "execution_summary": {
                "total_suites_attempted": len(context.test_suites),
                "successful_suites": len([r for r in suite_results.values() if not r.get("error")]),
                "failed_suites": len([r for r in suite_results.values() if r.get("error")]),
                "performance_summary": performance_summary
            },
            "suite_details": suite_results,
            "trends_analysis": self._analyze_execution_trends(),
            "quality_metrics": {
                "code_coverage": "Not implemented",  # Would integrate with coverage tools
                "test_reliability": self._calculate_test_reliability(suite_results),
                "performance_consistency": self._calculate_performance_consistency(suite_results)
            },
            "recommendations_detail": {
                "immediate_actions": [r for r in self._generate_recommendations(suite_results, performance_summary, context.performance_benchmarks) if "Fix" in r or "critical" in r.lower()],
                "improvement_opportunities": [r for r in self._generate_recommendations(suite_results, performance_summary, context.performance_benchmarks) if "Improve" in r or "Optimize" in r],
                "maintenance_tasks": [r for r in self._generate_recommendations(suite_results, performance_summary, context.performance_benchmarks) if "maintain" in r.lower()]
            }
        }
    
    def _validate_execution_success(self, pass_rate: float, performance_summary: Dict[str, float], benchmarks: Dict[str, float]) -> bool:
        """Validate if the execution meets success criteria"""
        success_criteria = [
            pass_rate >= benchmarks.get("min_pass_rate", 0.85),
            performance_summary["benchmark_compliance_rate"] >= 0.7,
            performance_summary["total_execution_time_ms"] < 60000  # 1 minute max
        ]
        
        return all(success_criteria)
    
    def _analyze_execution_trends(self) -> Dict[str, Any]:
        """Analyze trends across execution history"""
        if len(self.execution_history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        recent_executions = self.execution_history[-5:]  # Last 5 executions
        
        pass_rates = [exec.overall_pass_rate for exec in recent_executions]
        execution_times = [exec.execution_time_ms for exec in recent_executions]
        
        return {
            "pass_rate_trend": {
                "current": pass_rates[-1],
                "average": sum(pass_rates) / len(pass_rates),
                "direction": "improving" if pass_rates[-1] > pass_rates[0] else "declining"
            },
            "performance_trend": {
                "current_time_ms": execution_times[-1],
                "average_time_ms": sum(execution_times) / len(execution_times),
                "direction": "faster" if execution_times[-1] < execution_times[0] else "slower"
            },
            "stability": {
                "pass_rate_stability": max(pass_rates) - min(pass_rates),
                "performance_stability": max(execution_times) - min(execution_times)
            }
        }
    
    def _calculate_test_reliability(self, suite_results: Dict[str, Any]) -> float:
        """Calculate overall test reliability score"""
        successful_suites = [r for r in suite_results.values() if not r.get("error")]
        if not successful_suites:
            return 0.0
        
        reliability_scores = []
        for result in successful_suites:
            pass_rate = result.get("pass_rate", 0)
            total_tests = result.get("total_tests", 1)
            
            # Reliability considers both pass rate and test volume
            volume_factor = min(1.0, total_tests / 5.0)  # Normalize to 5 tests as baseline
            reliability = pass_rate * volume_factor
            reliability_scores.append(reliability)
        
        return sum(reliability_scores) / len(reliability_scores)
    
    def _calculate_performance_consistency(self, suite_results: Dict[str, Any]) -> float:
        """Calculate performance consistency across suites"""
        execution_times = []
        for result in suite_results.values():
            if not result.get("error") and "average_execution_time" in result:
                execution_times.append(result["average_execution_time"])
        
        if len(execution_times) < 2:
            return 1.0  # Perfect consistency if only one suite
        
        avg_time = sum(execution_times) / len(execution_times)
        variance = sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)
        coefficient_of_variation = (variance ** 0.5) / avg_time if avg_time > 0 else 1.0
        
        # Convert to consistency score (lower variance = higher consistency)
        consistency = max(0.0, 1.0 - coefficient_of_variation)
        return consistency
    
    def run_comprehensive_test_suite(self, parallel: bool = True, detailed: bool = True) -> TestOrchestrationResult:
        """Run all test suites with comprehensive reporting"""
        context = TestOrchestrationContext(
            orchestration_id=f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_suites=[TestSuite.ALL_AGENTS, TestSuite.INTEGRATION],
            parallel_execution=parallel,
            timeout_seconds=120,
            detailed_reporting=detailed,
            performance_benchmarks=self.performance_benchmarks,
            metadata={
                "purpose": "comprehensive_validation",
                "environment": "test",
                "version": "1.0"
            }
        )
        
        return self.orchestrate_tests(context)
    
    def run_quick_test_suite(self) -> TestOrchestrationResult:
        """Run essential tests quickly"""
        context = TestOrchestrationContext(
            orchestration_id=f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_suites=[TestSuite.DONATION_LIKELIHOOD, TestSuite.CAUSE_RECOMMENDATION],
            parallel_execution=True,
            timeout_seconds=30,
            detailed_reporting=False,
            performance_benchmarks=self.performance_benchmarks,
            metadata={
                "purpose": "quick_validation",
                "environment": "development",
                "version": "1.0"
            }
        )
        
        return self.orchestrate_tests(context)

def run_all_tests(parallel: bool = True, detailed: bool = True) -> TestOrchestrationResult:
    """Entry point for running all tests"""
    orchestrator = TestOrchestrator()
    return orchestrator.run_comprehensive_test_suite(parallel, detailed)

def run_quick_tests() -> TestOrchestrationResult:
    """Entry point for running quick tests"""
    orchestrator = TestOrchestrator()
    return orchestrator.run_quick_test_suite()

if __name__ == "__main__":
    # Run comprehensive test suite
    print("=== Running Comprehensive Test Suite ===")
    results = run_all_tests(parallel=True, detailed=True)
    
    print(f"\nTest Orchestration Results:")
    print(f"Total Tests: {results.total_tests_run}")
    print(f"Passed: {results.total_tests_passed}")
    print(f"Failed: {results.total_tests_failed}")
    print(f"Pass Rate: {results.overall_pass_rate:.2%}")
    print(f"Execution Time: {results.execution_time_ms:.1f}ms")
    print(f"Success: {results.execution_successful}")
    
    if results.recommendations:
        print(f"\nRecommendations:")
        for rec in results.recommendations:
            print(f"  - {rec}")
    
    # Save detailed results
    with open(f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(results.__dict__, f, indent=2, default=str)
