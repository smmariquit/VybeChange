"""
Agent API Module
Handles all agent-related API endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Import our agents
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))

from donation_likelihood_agent import DonationLikelihoodAgent, UserContext, LikelihoodScore
from cause_recommendation_agent import CauseRecommendationAgent, UserProfile, CauseRecommendations
from amount_optimization_agent import AmountOptimizationAgent, FinancialContext, BehavioralProfile, OptimizedAmounts
from performance_tracking_agent import PerformanceTrackingAgent, AgentDecision, UserFeedback, PerformanceReport

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])

# Initialize agents
likelihood_agent = DonationLikelihoodAgent()
cause_agent = CauseRecommendationAgent()
amount_agent = AmountOptimizationAgent()
performance_agent = PerformanceTrackingAgent()

# API Models
class DonationLikelihoodRequest(BaseModel):
    user_id: str
    transaction_amount: float
    wallet_balance: float
    location: Dict[str, str]
    transaction_category: str
    days_since_last_prompt: int = 30
    days_since_last_donation: int = 45
    average_donation_amount: float = 0.0
    total_lifetime_donations: float = 0.0
    preferred_causes: List[str] = []
    notification_preferences: Dict[str, bool] = {"donation_prompts": True}
    demographic_hints: Dict[str, Any] = {}
    time_of_day: str = "unknown"
    day_of_week: str = "unknown"

class CauseRecommendationRequest(BaseModel):
    user_id: str
    location: Dict[str, str]
    demographic_hints: Dict[str, Any] = {}
    preferred_causes: List[str] = []
    donation_history: List[Dict[str, Any]] = []
    transaction_category: str
    transaction_amount: float
    interests: List[str] = []
    values: List[str] = []

class AmountOptimizationRequest(BaseModel):
    # Financial context
    user_id: str
    transaction_amount: float
    wallet_balance: float
    monthly_income_estimate: Optional[float] = None
    recent_transactions: List[Dict[str, Any]] = []
    donation_history: List[Dict[str, Any]] = []
    spending_patterns: Dict[str, float] = {}
    savings_indicators: Dict[str, Any] = {}
    payment_method: str = "unknown"
    time_since_payday: int = 15
    
    # Behavioral profile
    donation_frequency: str = "unknown"
    average_donation_amount: float = 0.0
    largest_donation: float = 0.0
    smallest_donation: float = 0.0
    preferred_amounts: List[float] = []
    response_to_suggestions: Dict[str, int] = {}
    price_sensitivity: str = "medium"
    generosity_score: float = 0.5

class FeedbackSubmission(BaseModel):
    decision_id: str
    user_id: str
    feedback_type: str  # accepted, rejected, modified, ignored
    actual_amount: Optional[float] = None
    selected_cause: Optional[str] = None
    timing_feedback: Optional[str] = None
    satisfaction_score: Optional[int] = None

class PerformanceAnalysisRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    agent_names: Optional[List[str]] = None

# Agent Endpoints

@router.post("/likelihood/analyze")
async def analyze_donation_likelihood(request: DonationLikelihoodRequest) -> Dict[str, Any]:
    """
    Analyze donation likelihood for a user transaction
    
    This endpoint uses the DonationLikelihoodAgent to determine if and when
    to prompt a user for donations based on their context and behavior.
    """
    try:
        # Convert request to UserContext
        user_context = UserContext(
            user_id=request.user_id,
            transaction_amount=request.transaction_amount,
            wallet_balance=request.wallet_balance,
            location=request.location,
            transaction_category=request.transaction_category,
            days_since_last_prompt=request.days_since_last_prompt,
            days_since_last_donation=request.days_since_last_donation,
            average_donation_amount=request.average_donation_amount,
            total_lifetime_donations=request.total_lifetime_donations,
            preferred_causes=request.preferred_causes,
            notification_preferences=request.notification_preferences,
            demographic_hints=request.demographic_hints,
            time_of_day=request.time_of_day,
            day_of_week=request.day_of_week
        )
        
        # Process with agent
        result = likelihood_agent.process(user_context)
        
        # Convert result to dict for JSON response
        return {
            "decision_id": f"likelihood_{request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "likelihood_score": result.likelihood_score,
            "should_prompt": result.should_prompt,
            "optimal_timing": result.optimal_timing,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
            "next_optimal_time": result.next_optimal_time.isoformat() if result.next_optimal_time else None,
            "risk_factors": result.risk_factors,
            "opportunity_factors": result.opportunity_factors,
            "agent_name": "DonationLikelihoodAgent",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing donation likelihood: {str(e)}")

@router.post("/causes/recommend")
async def recommend_causes(request: CauseRecommendationRequest) -> Dict[str, Any]:
    """
    Recommend optimal causes/NGOs for a user
    
    This endpoint uses the CauseRecommendationAgent to suggest relevant
    NGOs and causes based on user preferences and context.
    """
    try:
        # Convert request to UserProfile
        user_profile = UserProfile(
            user_id=request.user_id,
            location=request.location,
            demographic_hints=request.demographic_hints,
            preferred_causes=request.preferred_causes,
            donation_history=request.donation_history,
            transaction_category=request.transaction_category,
            transaction_amount=request.transaction_amount,
            interests=request.interests,
            values=request.values
        )
        
        # Process with agent
        result = cause_agent.process(user_profile)
        
        # Convert result to dict for JSON response
        def cause_match_to_dict(match):
            return {
                "ngo": {
                    "ngo_id": match.ngo.ngo_id,
                    "name": match.ngo.name,
                    "cause_category": match.ngo.cause_category,
                    "subcategories": match.ngo.subcategories,
                    "region_focus": match.ngo.region_focus,
                    "impact_metrics": match.ngo.impact_metrics,
                    "urgency_level": match.ngo.urgency_level,
                    "recent_updates": match.ngo.recent_updates,
                    "transparency_score": match.ngo.transparency_score,
                    "efficiency_rating": match.ngo.efficiency_rating
                },
                "relevance_score": match.relevance_score,
                "match_reasons": match.match_reasons,
                "impact_potential": match.impact_potential,
                "personalized_message": match.personalized_message,
                "suggested_amount": match.suggested_amount,
                "urgency_indicator": match.urgency_indicator,
                "social_proof": match.social_proof
            }
        
        return {
            "decision_id": f"cause_{request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "primary_recommendation": cause_match_to_dict(result.primary_recommendation),
            "alternative_recommendations": [cause_match_to_dict(alt) for alt in result.alternative_recommendations],
            "reasoning_summary": result.reasoning_summary,
            "personalization_factors": result.personalization_factors,
            "diversity_score": result.diversity_score,
            "confidence": result.confidence,
            "agent_name": "CauseRecommendationAgent",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recommending causes: {str(e)}")

@router.post("/amounts/optimize")
async def optimize_amounts(request: AmountOptimizationRequest) -> Dict[str, Any]:
    """
    Optimize donation amounts for a user
    
    This endpoint uses the AmountOptimizationAgent to suggest optimal
    donation amounts based on financial capacity and behavioral patterns.
    """
    try:
        # Convert request to FinancialContext and BehavioralProfile
        financial_context = FinancialContext(
            user_id=request.user_id,
            transaction_amount=request.transaction_amount,
            wallet_balance=request.wallet_balance,
            monthly_income_estimate=request.monthly_income_estimate,
            recent_transactions=request.recent_transactions,
            donation_history=request.donation_history,
            spending_patterns=request.spending_patterns,
            savings_indicators=request.savings_indicators,
            payment_method=request.payment_method,
            time_since_payday=request.time_since_payday
        )
        
        behavioral_profile = BehavioralProfile(
            donation_frequency=request.donation_frequency,
            average_donation_amount=request.average_donation_amount,
            largest_donation=request.largest_donation,
            smallest_donation=request.smallest_donation,
            preferred_amounts=request.preferred_amounts,
            response_to_suggestions=request.response_to_suggestions,
            price_sensitivity=request.price_sensitivity,
            generosity_score=request.generosity_score
        )
        
        # Process with agent
        result = amount_agent.process(financial_context, behavioral_profile)
        
        # Convert result to dict for JSON response
        def amount_suggestion_to_dict(suggestion):
            return {
                "amount": suggestion.amount,
                "confidence": suggestion.confidence,
                "reasoning": suggestion.reasoning,
                "psychological_anchor": suggestion.psychological_anchor,
                "conversion_probability": suggestion.conversion_probability,
                "impact_description": suggestion.impact_description
            }
        
        return {
            "decision_id": f"amount_{request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "primary_amount": amount_suggestion_to_dict(result.primary_amount),
            "alternative_amounts": [amount_suggestion_to_dict(alt) for alt in result.alternative_amounts],
            "micro_amount": amount_suggestion_to_dict(result.micro_amount),
            "stretch_amount": amount_suggestion_to_dict(result.stretch_amount),
            "optimization_strategy": result.optimization_strategy,
            "personalization_factors": result.personalization_factors,
            "a_b_test_variant": result.a_b_test_variant,
            "confidence_score": result.confidence_score,
            "agent_name": "AmountOptimizationAgent",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing amounts: {str(e)}")

@router.post("/recommendation/complete")
async def get_complete_recommendation(
    likelihood_request: DonationLikelihoodRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Get a complete donation recommendation using all agents
    
    This endpoint orchestrates all three agents to provide a comprehensive
    donation recommendation including likelihood, cause, and amount.
    """
    try:
        # Step 1: Check donation likelihood
        likelihood_context = UserContext(
            user_id=likelihood_request.user_id,
            transaction_amount=likelihood_request.transaction_amount,
            wallet_balance=likelihood_request.wallet_balance,
            location=likelihood_request.location,
            transaction_category=likelihood_request.transaction_category,
            days_since_last_prompt=likelihood_request.days_since_last_prompt,
            days_since_last_donation=likelihood_request.days_since_last_donation,
            average_donation_amount=likelihood_request.average_donation_amount,
            total_lifetime_donations=likelihood_request.total_lifetime_donations,
            preferred_causes=likelihood_request.preferred_causes,
            notification_preferences=likelihood_request.notification_preferences,
            demographic_hints=likelihood_request.demographic_hints,
            time_of_day=likelihood_request.time_of_day,
            day_of_week=likelihood_request.day_of_week
        )
        
        likelihood_result = likelihood_agent.process(likelihood_context)
        
        # If likelihood is too low, return early recommendation
        if not likelihood_result.should_prompt:
            return {
                "decision_id": f"complete_{likelihood_request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "should_prompt": False,
                "likelihood_score": likelihood_result.likelihood_score,
                "reasoning": likelihood_result.reasoning,
                "next_optimal_time": likelihood_result.next_optimal_time.isoformat() if likelihood_result.next_optimal_time else None,
                "agent_name": "DonationOrchestrator",
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 2: Get cause recommendations
        cause_profile = UserProfile(
            user_id=likelihood_request.user_id,
            location=likelihood_request.location,
            demographic_hints=likelihood_request.demographic_hints,
            preferred_causes=likelihood_request.preferred_causes,
            donation_history=[],  # Would be populated from database
            transaction_category=likelihood_request.transaction_category,
            transaction_amount=likelihood_request.transaction_amount,
            interests=[],  # Would be populated from user profile
            values=[]  # Would be populated from user profile
        )
        
        cause_result = cause_agent.process(cause_profile)
        
        # Step 3: Optimize amounts
        financial_context = FinancialContext(
            user_id=likelihood_request.user_id,
            transaction_amount=likelihood_request.transaction_amount,
            wallet_balance=likelihood_request.wallet_balance,
            monthly_income_estimate=None,  # Would be estimated from transaction patterns
            recent_transactions=[],  # Would be populated from database
            donation_history=[],  # Would be populated from database
            spending_patterns={},  # Would be analyzed from transaction history
            savings_indicators={},  # Would be inferred from behavior
            payment_method="unknown",
            time_since_payday=15  # Default assumption
        )
        
        behavioral_profile = BehavioralProfile(
            donation_frequency="unknown",
            average_donation_amount=likelihood_request.average_donation_amount,
            largest_donation=likelihood_request.average_donation_amount * 2,  # Estimated
            smallest_donation=max(10, likelihood_request.average_donation_amount * 0.5),  # Estimated
            preferred_amounts=[],
            response_to_suggestions={},
            price_sensitivity="medium",
            generosity_score=0.5  # Default neutral score
        )
        
        amount_result = amount_agent.process(financial_context, behavioral_profile)
        
        # Combine all results
        complete_recommendation = {
            "decision_id": f"complete_{likelihood_request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "should_prompt": True,
            "likelihood_analysis": {
                "score": likelihood_result.likelihood_score,
                "confidence": likelihood_result.confidence,
                "reasoning": likelihood_result.reasoning,
                "optimal_timing": likelihood_result.optimal_timing
            },
            "cause_recommendation": {
                "primary_ngo": {
                    "name": cause_result.primary_recommendation.ngo.name,
                    "cause_category": cause_result.primary_recommendation.ngo.cause_category,
                    "relevance_score": cause_result.primary_recommendation.relevance_score
                },
                "personalized_message": cause_result.primary_recommendation.personalized_message,
                "confidence": cause_result.confidence
            },
            "amount_optimization": {
                "primary_amount": amount_result.primary_amount.amount,
                "alternative_amounts": [alt.amount for alt in amount_result.alternative_amounts],
                "micro_amount": amount_result.micro_amount.amount,
                "confidence": amount_result.confidence_score,
                "strategy": amount_result.optimization_strategy
            },
            "combined_confidence": (likelihood_result.confidence + cause_result.confidence + amount_result.confidence_score) / 3,
            "agent_name": "DonationOrchestrator",
            "timestamp": datetime.now().isoformat()
        }
        
        # Log this decision for performance tracking (background task)
        background_tasks.add_task(
            log_orchestrated_decision,
            complete_recommendation,
            likelihood_result,
            cause_result,
            amount_result
        )
        
        return complete_recommendation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating complete recommendation: {str(e)}")

@router.post("/feedback/submit")
async def submit_feedback(feedback: FeedbackSubmission) -> Dict[str, Any]:
    """
    Submit user feedback for agent decisions
    
    This endpoint collects user feedback to improve agent performance
    through continuous learning.
    """
    try:
        # Create UserFeedback object
        user_feedback = UserFeedback(
            decision_id=feedback.decision_id,
            user_id=feedback.user_id,
            feedback_type=feedback.feedback_type,
            actual_amount=feedback.actual_amount,
            selected_cause=feedback.selected_cause,
            timing_feedback=feedback.timing_feedback,
            satisfaction_score=feedback.satisfaction_score,
            timestamp=datetime.now()
        )
        
        # Store feedback (in production, this would be saved to database)
        performance_agent.feedback_history.append(user_feedback)
        
        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "feedback_id": f"feedback_{feedback.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@router.post("/performance/analyze")
async def analyze_performance(request: PerformanceAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze agent performance over a specified time period
    
    This endpoint uses the PerformanceTrackingAgent to generate insights
    and recommendations based on agent decisions and user feedback.
    """
    try:
        # Filter decisions and feedback for the specified period
        period_decisions = [
            d for d in performance_agent.decision_history
            if request.start_date <= d.timestamp <= request.end_date
        ]
        
        period_feedback = [
            f for f in performance_agent.feedback_history
            if request.start_date <= f.timestamp <= request.end_date
        ]
        
        # Filter by specific agents if requested
        if request.agent_names:
            period_decisions = [
                d for d in period_decisions
                if d.agent_name in request.agent_names
            ]
        
        # Generate performance report
        analysis_period = {
            'start': request.start_date,
            'end': request.end_date
        }
        
        report = performance_agent.process(analysis_period, period_decisions, period_feedback)
        
        # Convert report to dict for JSON response
        def metrics_to_dict(metrics):
            return {
                "agent_name": metrics.agent_name,
                "time_period": metrics.time_period,
                "total_decisions": metrics.total_decisions,
                "accuracy_rate": metrics.accuracy_rate,
                "acceptance_rate": metrics.acceptance_rate,
                "conversion_rate": metrics.conversion_rate,
                "average_confidence": metrics.average_confidence,
                "improvement_trend": metrics.improvement_trend,
                "key_insights": metrics.key_insights,
                "recommendations": metrics.recommendations
            }
        
        return {
            "report_id": report.report_id,
            "generated_at": report.generated_at.isoformat(),
            "time_period": {
                "start": report.time_period['start'].isoformat(),
                "end": report.time_period['end'].isoformat()
            },
            "agent_metrics": [metrics_to_dict(m) for m in report.agent_metrics],
            "cross_agent_analysis": report.cross_agent_analysis,
            "learning_insights": {
                "successful_patterns": report.learning_insights.successful_patterns,
                "failure_patterns": report.learning_insights.failure_patterns,
                "user_segments": report.learning_insights.user_segments,
                "optimization_opportunities": report.learning_insights.optimization_opportunities,
                "model_drift_indicators": report.learning_insights.model_drift_indicators,
                "recommended_adjustments": report.learning_insights.recommended_adjustments
            },
            "business_impact": report.business_impact,
            "recommendations": report.recommendations,
            "confidence_score": report.confidence_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing performance: {str(e)}")

@router.get("/status")
async def get_agent_status() -> Dict[str, Any]:
    """
    Get the current status of all agents
    
    This endpoint provides health check and status information for all agents.
    """
    return {
        "agents": {
            "DonationLikelihoodAgent": {
                "status": "active",
                "last_decision": "N/A",
                "total_decisions": len([d for d in performance_agent.decision_history 
                                      if d.agent_name == "DonationLikelihoodAgent"])
            },
            "CauseRecommendationAgent": {
                "status": "active", 
                "last_decision": "N/A",
                "total_decisions": len([d for d in performance_agent.decision_history 
                                      if d.agent_name == "CauseRecommendationAgent"])
            },
            "AmountOptimizationAgent": {
                "status": "active",
                "last_decision": "N/A", 
                "total_decisions": len([d for d in performance_agent.decision_history 
                                      if d.agent_name == "AmountOptimizationAgent"])
            },
            "PerformanceTrackingAgent": {
                "status": "active",
                "total_feedback_records": len(performance_agent.feedback_history),
                "total_decision_records": len(performance_agent.decision_history)
            }
        },
        "system_health": "operational",
        "timestamp": datetime.now().isoformat()
    }

# Background task for logging orchestrated decisions
async def log_orchestrated_decision(complete_recommendation: Dict[str, Any],
                                  likelihood_result: LikelihoodScore,
                                  cause_result: CauseRecommendations,
                                  amount_result: OptimizedAmounts):
    """Log the orchestrated decision for performance tracking"""
    
    # Create AgentDecision records for each agent used
    decisions = [
        AgentDecision(
            decision_id=f"{complete_recommendation['decision_id']}_likelihood",
            agent_name="DonationLikelihoodAgent",
            user_id=complete_recommendation['decision_id'].split('_')[1],
            timestamp=datetime.now(),
            input_context={"transaction_amount": 0},  # Would include full context
            decision_output={"likelihood_score": likelihood_result.likelihood_score},
            confidence_score=likelihood_result.confidence / 100,
            reasoning=likelihood_result.reasoning
        ),
        AgentDecision(
            decision_id=f"{complete_recommendation['decision_id']}_cause",
            agent_name="CauseRecommendationAgent", 
            user_id=complete_recommendation['decision_id'].split('_')[1],
            timestamp=datetime.now(),
            input_context={"transaction_category": "unknown"},
            decision_output={"primary_ngo": cause_result.primary_recommendation.ngo.name},
            confidence_score=cause_result.confidence / 100,
            reasoning=["orchestrated_recommendation"]
        ),
        AgentDecision(
            decision_id=f"{complete_recommendation['decision_id']}_amount",
            agent_name="AmountOptimizationAgent",
            user_id=complete_recommendation['decision_id'].split('_')[1], 
            timestamp=datetime.now(),
            input_context={"transaction_amount": 0},
            decision_output={"primary_amount": amount_result.primary_amount.amount},
            confidence_score=amount_result.confidence_score / 100,
            reasoning=amount_result.primary_amount.reasoning
        )
    ]
    
    # Add to performance tracking
    performance_agent.decision_history.extend(decisions)
