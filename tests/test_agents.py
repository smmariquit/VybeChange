"""
Legacy Testing Framework for ImpactSense AI Agents
This file now serves as a bridge to the new individual test agent architecture
All agents now follow the input-tool call-output pattern with dedicated test files
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


class TestDonationLikelyScoreAgent:
    """Test cases for donation likelihood scoring"""
    
    def setup_method(self):
        self.agent = DonationLikelyScoreAgent()
    
    def test_high_likelihood_scenario(self):
        """Test scenario with high donation likelihood"""
        context = UserContext(
            user_id="test_user_1",
            transaction_amount=1000.0,  # Large transaction
            wallet_balance=5000.0,      # Healthy balance
            location={"region": "NCR"},
            transaction_category="groceries",  # Essential purchase
            days_since_last_prompt=10,  # Good cooldown
            days_since_last_donation=30,
            average_donation_amount=5.0,  # Established donor
            total_lifetime_donations=50.0,
            preferred_causes=["Health"],
            notification_preferences={"impact_updates": True},
            demographic_hints={"age_range": "25-34"}
        )
        
        result = self.agent.process(context)
        
        assert result["likelihood_score"] >= 70
        assert result["recommendation"] == "prompt_now"
        assert "large_transaction_generosity" in result["reasoning_factors"]
        assert "healthy_wallet_balance" in result["reasoning_factors"]
        assert result["confidence_level"] in ["medium", "high"]
    
    def test_low_likelihood_scenario(self):
        """Test scenario with low donation likelihood"""
        context = UserContext(
            user_id="test_user_2",
            transaction_amount=25.0,    # Small transaction
            wallet_balance=50.0,        # Low balance
            location={"region": "NCR"},
            transaction_category="entertainment",
            days_since_last_prompt=1,   # Too recent
            days_since_last_donation=5,
            average_donation_amount=0.0,  # No donation history
            total_lifetime_donations=0.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        result = self.agent.process(context)
        
        assert result["likelihood_score"] < 40
        assert result["recommendation"] == "skip_prompt"
        assert "low_wallet_balance" in result["reasoning_factors"]
        assert "too_recent_prompt" in result["reasoning_factors"]
    
    def test_payday_timing_bonus(self):
        """Test that payday timing increases likelihood"""
        context = UserContext(
            user_id="test_user_3",
            transaction_amount=500.0,
            wallet_balance=2000.0,
            location={"region": "NCR"},
            transaction_category="groceries",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=3.0,
            total_lifetime_donations=30.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        # Mock datetime to simulate Friday (payday)
        with patch('src.agents.core_agents.datetime') as mock_datetime:
            mock_friday = Mock()
            mock_friday.weekday.return_value = 4  # Friday
            mock_datetime.now.return_value = mock_friday
            
            result = self.agent.process(context)
            
            assert "likely_payday_timing" in result["reasoning_factors"]
    
    def test_cooldown_calculation(self):
        """Test next check timing calculation"""
        context = UserContext(
            user_id="test_user_4",
            transaction_amount=100.0,
            wallet_balance=1000.0,
            location={"region": "NCR"},
            transaction_category="food",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=0.0,
            total_lifetime_donations=0.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        result = self.agent.process(context)
        
        # Should have next_check_hours defined
        assert "next_check_hours" in result
        assert isinstance(result["next_check_hours"], int)
        assert result["next_check_hours"] > 0


class TestLocalCauseRecommenderAgent:
    """Test cases for cause recommendation logic"""
    
    def setup_method(self):
        self.agent = LocalCauseRecommenderAgent(NGOS)
    
    def test_exact_region_match_priority(self):
        """Test that exact region matches get highest priority"""
        context = UserContext(
            user_id="test_user_5",
            transaction_amount=500.0,
            wallet_balance=2000.0,
            location={"region": "NCR"},
            transaction_category="groceries",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=5.0,
            total_lifetime_donations=50.0,
            preferred_causes=["Health"],
            notification_preferences={},
            demographic_hints={}
        )
        
        result = self.agent.process(context)
        
        assert result["primary_recommendation"] is not None
        recommendation = result["primary_recommendation"]
        
        # Should prefer NCR-based NGOs
        assert "NCR" in recommendation.region_focus or recommendation.region_focus == "Nationwide"
        assert recommendation.relevance_score > 0
    
    def test_category_preference_matching(self):
        """Test that user's preferred causes get priority"""
        context = UserContext(
            user_id="test_user_6",
            transaction_amount=500.0,
            wallet_balance=2000.0,
            location={"region": "NCR"},
            transaction_category="groceries",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=5.0,
            total_lifetime_donations=50.0,
            preferred_causes=["Education"],  # Specific preference
            notification_preferences={},
            demographic_hints={}
        )
        
        result = self.agent.process(context)
        
        assert result["primary_recommendation"] is not None
        recommendation = result["primary_recommendation"]
        
        # Should match or relate to Education
        assert "Education" in recommendation.cause_category or any(
            "Education" in cause for cause in [recommendation.cause_category]
        )
    
    def test_transaction_context_relevance(self):
        """Test transaction category influences cause recommendation"""
        context = UserContext(
            user_id="test_user_7",
            transaction_amount=500.0,
            wallet_balance=2000.0,
            location={"region": "NCR"},
            transaction_category="food",  # Food purchase
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=5.0,
            total_lifetime_donations=50.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        result = self.agent.process(context)
        
        assert result["primary_recommendation"] is not None
        recommendation = result["primary_recommendation"]
        
        # Food purchases should suggest nutrition-related causes
        nutrition_related = any(word in recommendation.cause_category.lower() 
                              for word in ["nutrition", "poverty", "food"])
        assert nutrition_related or recommendation.cause_category in ["Nutrition", "Poverty Alleviation"]
    
    def test_no_suitable_causes(self):
        """Test handling when no suitable causes are found"""
        # Mock empty NGO database
        empty_agent = LocalCauseRecommenderAgent([])
        
        context = UserContext(
            user_id="test_user_8",
            transaction_amount=500.0,
            wallet_balance=2000.0,
            location={"region": "Unknown Region"},
            transaction_category="groceries",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=5.0,
            total_lifetime_donations=50.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        result = empty_agent.process(context)
        
        assert result["recommendation"] is None
        assert result["reason"] == "no_relevant_causes_found"


class TestDonationAmountOptimizerAgent:
    """Test cases for donation amount optimization"""
    
    def setup_method(self):
        self.agent = DonationAmountOptimizerAgent()
    
    def test_small_transaction_amounts(self):
        """Test amount suggestions for small transactions"""
        context = UserContext(
            user_id="test_user_9",
            transaction_amount=100.0,  # Small transaction
            wallet_balance=1000.0,
            location={"region": "NCR"},
            transaction_category="groceries",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=0.0,  # First-time donor
            total_lifetime_donations=0.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        result = self.agent.process(context)
        
        # Small transactions should suggest small amounts
        assert result["primary_amount"] <= 5.0
        assert all(amt <= 10.0 for amt in result["alternative_amounts"])
        assert result["round_up_amount"] > 0
    
    def test_large_transaction_amounts(self):
        """Test amount suggestions for large transactions"""
        context = UserContext(
            user_id="test_user_10",
            transaction_amount=5000.0,  # Large transaction
            wallet_balance=20000.0,
            location={"region": "NCR"},
            transaction_category="luxury",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=10.0,  # Established donor
            total_lifetime_donations=100.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        result = self.agent.process(context)
        
        # Large transactions should suggest larger amounts
        assert result["primary_amount"] >= 5.0
        assert any(amt >= 10.0 for amt in result["alternative_amounts"])
    
    def test_wallet_balance_constraint(self):
        """Test that suggestions respect wallet balance"""
        context = UserContext(
            user_id="test_user_11",
            transaction_amount=2000.0,
            wallet_balance=150.0,  # Low balance after transaction
            location={"region": "NCR"},
            transaction_category="groceries",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=5.0,
            total_lifetime_donations=50.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        result = self.agent.process(context)
        
        # Should not suggest more than 1% of wallet balance
        max_suggested = max([result["primary_amount"]] + result["alternative_amounts"])
        assert max_suggested <= context.wallet_balance * 0.01
    
    def test_first_time_donor_conservative_approach(self):
        """Test conservative amounts for first-time donors"""
        context = UserContext(
            user_id="test_user_12",
            transaction_amount=1000.0,
            wallet_balance=5000.0,
            location={"region": "NCR"},
            transaction_category="groceries",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=0.0,  # No donation history
            total_lifetime_donations=0.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        result = self.agent.process(context)
        
        # First-time donors should get smallest option as primary
        all_amounts = [result["primary_amount"]] + result["alternative_amounts"]
        assert result["primary_amount"] == min(all_amounts)


class TestImpactSenseOrchestrator:
    """Test cases for the main orchestrator"""
    
    def setup_method(self):
        self.orchestrator = ImpactSenseOrchestrator(NGOS)
    
    def test_successful_recommendation_generation(self):
        """Test complete recommendation generation"""
        context = UserContext(
            user_id="test_user_13",
            transaction_amount=689.50,
            wallet_balance=2500.0,
            location={"region": "NCR"},
            transaction_category="groceries",
            days_since_last_prompt=8,
            days_since_last_donation=30,
            average_donation_amount=3.50,
            total_lifetime_donations=45.0,
            preferred_causes=["Health", "Education"],
            notification_preferences={"impact_updates": True},
            demographic_hints={"age_range": "25-34"}
        )
        
        recommendation = self.orchestrator.generate_recommendation(context)
        
        assert recommendation is not None
        assert recommendation.primary_amount > 0
        assert recommendation.cause.ngo_id in [ngo["ngo_id"] for ngo in NGOS]
        assert recommendation.likelihood_score > 0
        assert len(recommendation.alternative_amounts) > 0
    
    def test_no_recommendation_when_inappropriate(self):
        """Test that no recommendation is generated when inappropriate"""
        context = UserContext(
            user_id="test_user_14",
            transaction_amount=25.0,
            wallet_balance=30.0,  # Very low balance
            location={"region": "NCR"},
            transaction_category="groceries",
            days_since_last_prompt=1,  # Too recent
            days_since_last_donation=2,
            average_donation_amount=0.0,
            total_lifetime_donations=0.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        recommendation = self.orchestrator.generate_recommendation(context)
        
        assert recommendation is None
    
    def test_error_handling(self):
        """Test orchestrator error handling"""
        # Invalid context should not crash the system
        invalid_context = UserContext(
            user_id="test_user_15",
            transaction_amount=-100.0,  # Invalid amount
            wallet_balance=-500.0,      # Invalid balance
            location={},  # Missing region
            transaction_category="",
            days_since_last_prompt=-1,
            days_since_last_donation=-1,
            average_donation_amount=-5.0,
            total_lifetime_donations=-10.0,
            preferred_causes=[],
            notification_preferences={},
            demographic_hints={}
        )
        
        # Should handle gracefully and return None
        recommendation = self.orchestrator.generate_recommendation(invalid_context)
        assert recommendation is None


# Integration Tests
class TestAgentIntegration:
    """Integration tests across multiple agents"""
    
    def test_agent_decision_consistency(self):
        """Test that agent decisions are consistent across multiple calls"""
        context = UserContext(
            user_id="test_user_16",
            transaction_amount=500.0,
            wallet_balance=2000.0,
            location={"region": "NCR"},
            transaction_category="groceries",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=5.0,
            total_lifetime_donations=50.0,
            preferred_causes=["Health"],
            notification_preferences={},
            demographic_hints={}
        )
        
        orchestrator = ImpactSenseOrchestrator(NGOS)
        
        # Generate multiple recommendations with same context
        recommendations = []
        for _ in range(5):
            rec = orchestrator.generate_recommendation(context)
            if rec:
                recommendations.append(rec)
        
        # All recommendations should be similar (allowing for some variation)
        if recommendations:
            first_rec = recommendations[0]
            for rec in recommendations[1:]:
                # Same NGO should be recommended
                assert rec.cause.ngo_id == first_rec.cause.ngo_id
                # Amount should be in similar range
                assert abs(rec.primary_amount - first_rec.primary_amount) <= 2.0
    
    def test_regional_preference_propagation(self):
        """Test that regional preferences propagate through the system"""
        palawan_context = UserContext(
            user_id="test_user_17",
            transaction_amount=500.0,
            wallet_balance=2000.0,
            location={"region": "Palawan"},  # Specific region
            transaction_category="healthcare",
            days_since_last_prompt=7,
            days_since_last_donation=30,
            average_donation_amount=5.0,
            total_lifetime_donations=50.0,
            preferred_causes=["Health"],
            notification_preferences={},
            demographic_hints={}
        )
        
        orchestrator = ImpactSenseOrchestrator(NGOS)
        recommendation = orchestrator.generate_recommendation(palawan_context)
        
        if recommendation:
            # Should prefer Palawan-focused NGOs or nationwide NGOs
            assert (recommendation.cause.region_focus == "Palawan" or 
                    recommendation.cause.region_focus == "Nationwide")


# Test Fixtures and Utilities
@pytest.fixture
def sample_user_context():
    """Standard user context for testing"""
    return UserContext(
        user_id="test_user_standard",
        transaction_amount=500.0,
        wallet_balance=2000.0,
        location={"region": "NCR"},
        transaction_category="groceries",
        days_since_last_prompt=7,
        days_since_last_donation=30,
        average_donation_amount=5.0,
        total_lifetime_donations=50.0,
        preferred_causes=["Health", "Education"],
        notification_preferences={"impact_updates": True},
        demographic_hints={"age_range": "25-34"}
    )

@pytest.fixture
def mock_ngo_database():
    """Mock NGO database for testing"""
    return [
        {
            "ngo_id": "TEST001",
            "ngo_name": "Test Health NGO",
            "category": "Health",
            "region_focus": "NCR"
        },
        {
            "ngo_id": "TEST002", 
            "ngo_name": "Test Education NGO",
            "category": "Education",
            "region_focus": "Nationwide"
        }
    ]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
