"""
Vybe Change Core Agent Framework
Autonomous AI agents for intelligent microdonation orchestration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserContext:
    """Comprehensive user context for agent decision-making"""
    user_id: str
    transaction_amount: float
    wallet_balance: float
    location: Dict[str, str]  # {"lat": "14.5995", "lng": "120.9842", "region": "NCR"}
    transaction_category: str
    days_since_last_prompt: int
    days_since_last_donation: int
    average_donation_amount: float
    total_lifetime_donations: float
    preferred_causes: List[str]
    notification_preferences: Dict[str, bool]
    demographic_hints: Dict[str, Any]

@dataclass
class CauseRecommendation:
    """NGO cause recommendation with relevance scoring"""
    ngo_id: str
    ngo_name: str
    cause_category: str
    region_focus: str
    relevance_score: float
    impact_description: str
    urgency_level: str
    recent_updates: bool

@dataclass
class DonationSuggestion:
    """Complete donation suggestion package"""
    primary_amount: float
    alternative_amounts: List[float]
    cause: CauseRecommendation
    message_tone: str
    community_context: str
    likelihood_score: float
    timing_optimal: bool

class BaseAgent(ABC):
    """Abstract base class for all ImpactSense agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"ImpactSense.{name}")
        
    @abstractmethod
    def process(self, context: UserContext) -> Dict[str, Any]:
        """Process user context and return agent-specific recommendations"""
        pass
    
    def log_decision(self, context: UserContext, decision: Dict[str, Any], reasoning: str):
        """Log agent decisions for learning and debugging"""
        self.logger.info(f"User {context.user_id}: {reasoning} -> {decision}")

class DonationLikelyScoreAgent(BaseAgent):
    """Determines optimal timing for donation prompts"""
    
    def __init__(self):
        super().__init__("DonationLikelyScoreAgent")
        self.cooldown_rules = {
            'declined_prompt': 14,  # days
            'accepted_prompt': 7,   # days  
            'low_wallet': 7,        # days
        }
    
    def process(self, context: UserContext) -> Dict[str, Any]:
        """Calculate donation likelihood score (0-100)"""
        score = 0
        reasoning = []
        
        # Recent prompt behavior
        if context.days_since_last_prompt >= 6:
            score += 30
            reasoning.append("sufficient_cooldown_period")
        elif context.days_since_last_prompt < 3:
            score -= 20
            reasoning.append("too_recent_prompt")
        
        # Transaction size consideration
        if context.transaction_amount > 500:
            score += 25
            reasoning.append("large_transaction_generosity")
        elif context.transaction_amount < 50:
            score -= 15
            reasoning.append("small_transaction_constraint")
        
        # Wallet balance check (critical)
        if context.wallet_balance < 100:
            score -= 40
            reasoning.append("low_wallet_balance")
        elif context.wallet_balance > 1000:
            score += 15
            reasoning.append("healthy_wallet_balance")
        
        # Purchase category psychology
        essential_categories = ['food', 'groceries', 'utilities', 'transportation']
        if context.transaction_category in essential_categories:
            score += 20
            reasoning.append("post_essential_purchase_openness")
        
        # Historical giving pattern
        if context.average_donation_amount > 0:
            score += 15
            reasoning.append("established_donor_pattern")
        
        # Timing factors (could be enhanced with real-time data)
        now = datetime.now()
        if now.weekday() in [4, 5]:  # Friday, Saturday (payday assumption)
            score += 10
            reasoning.append("likely_payday_timing")
        
        # Cap score at reasonable bounds
        final_score = max(0, min(100, score))
        
        # Decision logic
        if final_score >= 70:
            recommendation = "prompt_now"
        elif final_score >= 40:
            recommendation = "prompt_with_caution"
        else:
            recommendation = "skip_prompt"
        
        decision = {
            'likelihood_score': final_score,
            'recommendation': recommendation,
            'reasoning_factors': reasoning,
            'next_check_hours': self._calculate_next_check(final_score),
            'confidence_level': self._calculate_confidence(reasoning)
        }
        
        self.log_decision(context, decision, f"Score: {final_score}, Factors: {', '.join(reasoning)}")
        return decision
    
    def _calculate_next_check(self, score: float) -> int:
        """Calculate when to next evaluate this user"""
        if score >= 70:
            return 24 * 7  # 1 week
        elif score >= 40:
            return 24 * 3  # 3 days
        else:
            return 24 * 1  # 1 day
    
    def _calculate_confidence(self, reasoning: List[str]) -> str:
        """Assess confidence in the recommendation"""
        critical_factors = ['low_wallet_balance', 'too_recent_prompt']
        if any(factor in reasoning for factor in critical_factors):
            return 'high'
        elif len(reasoning) >= 3:
            return 'medium'
        else:
            return 'low'

class LocalCauseRecommenderAgent(BaseAgent):
    """Matches users with geographically and personally relevant causes"""
    
    def __init__(self, ngo_database):
        super().__init__("LocalCauseRecommenderAgent")
        self.ngo_database = ngo_database
        
    def process(self, context: UserContext) -> Dict[str, Any]:
        """Find the most relevant cause for the user"""
        # Get NGOs in user's region
        nearby_ngos = self._filter_by_geography(context.location)
        
        # Score each NGO for relevance
        scored_ngos = []
        for ngo in nearby_ngos:
            relevance_score = self._calculate_relevance_score(ngo, context)
            if relevance_score > 0:
                scored_ngos.append((ngo, relevance_score))
        
        # Sort by relevance score
        scored_ngos.sort(key=lambda x: x[1], reverse=True)
        
        if not scored_ngos:
            return {'recommendation': None, 'reason': 'no_relevant_causes_found'}
        
        # Return top recommendation
        top_ngo, top_score = scored_ngos[0]
        
        recommendation = CauseRecommendation(
            ngo_id=top_ngo['ngo_id'],
            ngo_name=top_ngo['ngo_name'],
            cause_category=top_ngo['category'],
            region_focus=top_ngo['region_focus'],
            relevance_score=top_score,
            impact_description=self._generate_impact_description(top_ngo, context),
            urgency_level=self._assess_urgency(top_ngo),
            recent_updates=self._check_recent_updates(top_ngo['ngo_id'])
        )
        
        decision = {
            'primary_recommendation': recommendation,
            'alternatives': [scored_ngos[i][0] for i in range(1, min(3, len(scored_ngos)))],
            'total_candidates': len(nearby_ngos),
            'geographic_match': self._assess_geographic_match(top_ngo, context.location)
        }
        
        self.log_decision(context, decision, f"Selected {top_ngo['ngo_name']} with score {top_score}")
        return decision
    
    def _filter_by_geography(self, location: Dict[str, str]) -> List[Dict]:
        """Filter NGOs by geographic relevance"""
        user_region = location.get('region', '')
        relevant_ngos = []
        
        for ngo in self.ngo_database:
            # Exact region match gets priority
            if ngo['region_focus'] == user_region:
                relevant_ngos.append(ngo)
            # Nationwide NGOs are always relevant
            elif ngo['region_focus'] == 'Nationwide':
                relevant_ngos.append(ngo)
            # Broader regional matches (e.g., "Luzon" contains "NCR")
            elif user_region in ngo['region_focus'] or ngo['region_focus'] in user_region:
                relevant_ngos.append(ngo)
        
        return relevant_ngos
    
    def _calculate_relevance_score(self, ngo: Dict, context: UserContext) -> float:
        """Calculate how relevant this NGO is to the user"""
        score = 0
        
        # Geographic relevance (0-40 points)
        if ngo['region_focus'] == context.location.get('region'):
            score += 40  # Exact regional match
        elif ngo['region_focus'] == 'Nationwide':
            score += 25  # Nationwide coverage
        else:
            score += 10  # Broader regional match
        
        # Category preference matching (0-30 points)
        if ngo['category'] in context.preferred_causes:
            score += 30
        elif any(pref in ngo['category'] for pref in context.preferred_causes):
            score += 15
        
        # Transaction context relevance (0-20 points)
        category_mappings = {
            'food': ['Nutrition', 'Poverty Alleviation'],
            'healthcare': ['Health', 'Women\'s Health', 'Pediatric Health'],
            'education': ['Education'],
            'transport': ['Environment'],  # Assuming eco-conscious transport users
        }
        
        if context.transaction_category in category_mappings:
            if ngo['category'] in category_mappings[context.transaction_category]:
                score += 20
        
        # Historical engagement (0-10 points)
        # This would require integration with donation history
        # For now, we'll use a placeholder
        if context.total_lifetime_donations > 0:
            score += 5  # Bonus for established donors
        
        return score
    
    def _generate_impact_description(self, ngo: Dict, context: UserContext) -> str:
        """Generate contextual impact description"""
        impact_templates = {
            'Nutrition': f"Help provide nutritious meals to families in {ngo['region_focus']}",
            'Education': f"Support educational programs for children in {ngo['region_focus']}",
            'Health': f"Fund healthcare services for communities in {ngo['region_focus']}",
            'Environment': f"Protect natural resources in {ngo['region_focus']}",
            'Poverty Alleviation': f"Create sustainable livelihoods in {ngo['region_focus']}",
        }
        
        return impact_templates.get(ngo['category'], 
                                  f"Support {ngo['category'].lower()} initiatives in {ngo['region_focus']}")
    
    def _assess_urgency(self, ngo: Dict) -> str:
        """Assess the urgency level of the cause"""
        urgent_categories = ['Disaster Relief', 'Health', 'Street Children']
        
        if ngo['category'] in urgent_categories:
            return 'high'
        elif ngo['category'] in ['Education', 'Poverty Alleviation']:
            return 'medium'
        else:
            return 'low'
    
    def _check_recent_updates(self, ngo_id: str) -> bool:
        """Check if NGO has recent impact updates"""
        # This would integrate with the proof collection system
        # For now, return a placeholder
        return True
    
    def _assess_geographic_match(self, ngo: Dict, location: Dict[str, str]) -> str:
        """Assess quality of geographic matching"""
        if ngo['region_focus'] == location.get('region'):
            return 'exact'
        elif ngo['region_focus'] == 'Nationwide':
            return 'nationwide'
        else:
            return 'regional'

class DonationAmountOptimizerAgent(BaseAgent):
    """Optimizes donation amount suggestions based on context and psychology"""
    
    def __init__(self):
        super().__init__("DonationAmountOptimizerAgent")
        
    def process(self, context: UserContext) -> Dict[str, Any]:
        """Calculate optimal donation amounts"""
        # Base calculation: round up to nearest peso
        round_up_amount = self._calculate_round_up(context.transaction_amount)
        
        # Generate amount options based on transaction size
        amount_options = self._generate_amount_options(context.transaction_amount, context)
        
        # Factor in user's historical giving pattern
        if context.average_donation_amount > 0:
            amount_options = self._adjust_for_user_pattern(amount_options, context.average_donation_amount)
        
        # Select primary recommendation
        primary_amount = self._select_primary_amount(amount_options, context)
        
        # Generate alternatives
        alternatives = [amt for amt in amount_options if amt != primary_amount][:2]
        
        decision = {
            'primary_amount': primary_amount,
            'alternative_amounts': alternatives,
            'round_up_amount': round_up_amount,
            'reasoning': self._generate_reasoning(primary_amount, context),
            'psychological_factor': self._assess_psychological_factor(context)
        }
        
        self.log_decision(context, decision, f"Recommended ₱{primary_amount} based on ₱{context.transaction_amount} transaction")
        return decision
    
    def _calculate_round_up(self, amount: float) -> float:
        """Calculate simple round-up amount"""
        import math
        return math.ceil(amount) - amount
    
    def _generate_amount_options(self, transaction_amount: float, context: UserContext) -> List[float]:
        """Generate donation amount options by rounding up to the nearest tens"""
        import math
        # Round up to nearest tens
        rounded_up = math.ceil(transaction_amount / 10) * 10
        donation_amount = rounded_up - transaction_amount
        # Always suggest the round-up amount, and optionally multiples of ten up to 30
        options = [donation_amount]
        for extra in [10, 20, 30]:
            if donation_amount + extra <= context.wallet_balance * 0.01:
                options.append(donation_amount + extra)
        # Filter out zero or negative options
        options = [amt for amt in options if amt > 0]
        # Ensure at least one option
        if not options:
            options = [10.0]
        return options
    
    def _adjust_for_user_pattern(self, amounts: List[float], avg_donation: float) -> List[float]:
        """Adjust amounts based on user's historical giving pattern"""
        # If user typically gives more, suggest higher amounts
        if avg_donation > 10:
            return [amt * 1.5 for amt in amounts if amt * 1.5 <= 100]  # Cap at ₱100
        elif avg_donation > 5:
            return [amt * 1.2 for amt in amounts]
        else:
            return amounts  # Keep original suggestions for small donors
    
    def _select_primary_amount(self, options: List[float], context: UserContext) -> float:
        """Select the primary recommended amount"""
        if not options:
            return 1.0
        
        # For first-time donors, suggest the smallest amount
        if context.total_lifetime_donations == 0:
            return min(options)
        
        # For regular donors, suggest middle option
        if len(options) >= 3:
            return options[1]  # Middle option
        elif len(options) == 2:
            return options[0]  # Smaller of two options
        else:
            return options[0]  # Only option
    
    def _generate_reasoning(self, amount: float, context: UserContext) -> str:
        """Generate human-readable reasoning for the amount suggestion"""
        if context.total_lifetime_donations == 0:
            return f"Perfect amount to start your giving journey"
        elif amount <= 5:
            return f"Small change, big impact"
        elif amount <= 20:
            return f"Meaningful contribution that fits your spending"
        else:
            return f"Generous giving that matches your transaction size"
    
    def _assess_psychological_factor(self, context: UserContext) -> str:
        """Assess psychological factors affecting donation willingness"""
        if context.transaction_category in ['luxury', 'entertainment', 'dining']:
            return 'post_indulgence_generosity'
        elif context.transaction_category in ['groceries', 'utilities']:
            return 'essential_purchase_completion'
        elif context.wallet_balance > context.transaction_amount * 10:
            return 'financial_comfort'
        else:
            return 'standard_willingness'

# Agent Orchestrator
class ImpactSenseOrchestrator:
    """Coordinates all agents to generate complete donation recommendations"""
    
    def __init__(self, ngo_database):
        self.likelihood_agent = DonationLikelyScoreAgent()
        self.cause_agent = LocalCauseRecommenderAgent(ngo_database)
        self.amount_agent = DonationAmountOptimizerAgent()
        self.logger = logging.getLogger("ImpactSense.Orchestrator")
    
    def generate_recommendation(self, context: UserContext) -> Optional[DonationSuggestion]:
        """Generate complete donation recommendation"""
        try:
            # Step 1: Check if user is likely to donate
            likelihood_result = self.likelihood_agent.process(context)
            
            if likelihood_result['recommendation'] == 'skip_prompt':
                self.logger.info(f"User {context.user_id}: Skipping prompt - {likelihood_result['reasoning_factors']}")
                return None
            
            # Step 2: Find relevant cause
            cause_result = self.cause_agent.process(context)
            
            if not cause_result.get('primary_recommendation'):
                self.logger.info(f"User {context.user_id}: No suitable causes found")
                return None
            
            # Step 3: Optimize donation amount
            amount_result = self.amount_agent.process(context)
            
            # Step 4: Combine into final recommendation
            recommendation = DonationSuggestion(
                primary_amount=amount_result['primary_amount'],
                alternative_amounts=amount_result['alternative_amounts'],
                cause=cause_result['primary_recommendation'],
                message_tone='casual',  # Would be determined by ToneCalibratorAgent
                community_context='',   # Would be determined by CommunityFramerAgent
                likelihood_score=likelihood_result['likelihood_score'],
                timing_optimal=likelihood_result['recommendation'] == 'prompt_now'
            )
            
            self.logger.info(f"User {context.user_id}: Generated complete recommendation")
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation for user {context.user_id}: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # This would be imported from constants.ngos in real implementation
    from constants.ngos import NGOS
    
    # Create orchestrator
    orchestrator = ImpactSenseOrchestrator(NGOS)
    
    # Example user context
    sample_context = UserContext(
        user_id="user_123",
        transaction_amount=689.50,
        wallet_balance=2500.00,
        location={"region": "NCR", "lat": "14.5995", "lng": "120.9842"},
        transaction_category="groceries",
        days_since_last_prompt=8,
        days_since_last_donation=30,
        average_donation_amount=3.50,
        total_lifetime_donations=45.00,
        preferred_causes=["Health", "Education"],
        notification_preferences={"impact_updates": True, "social_sharing": False},
        demographic_hints={"age_range": "25-34", "income_level": "middle"}
    )
    
    # Generate recommendation
    recommendation = orchestrator.generate_recommendation(sample_context)
    
    if recommendation:
        print(f"Recommendation: Donate ₱{recommendation.primary_amount} to {recommendation.cause.ngo_name}")
        print(f"Likelihood Score: {recommendation.likelihood_score}")
        print(f"Impact: {recommendation.cause.impact_description}")
    else:
        print("No recommendation generated at this time")
