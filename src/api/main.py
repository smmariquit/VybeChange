"""ImpactSense BPI VIBE Integration API
Endpoints for embedding donation prompts into BPI's payment flow
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid

from ..agents.core_agents import ImpactSenseOrchestrator, UserContext
from constants.ngos import NGOS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ImpactSense API",
    description="AI-powered microdonation layer for BPI VIBE",
    version="0.9.0"
)

# CORS middleware for BPI VIBE integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vibe.bpi.com.ph", "https://api.bpi.com.ph"],  # BPI domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

# Initialize agent orchestrator
orchestrator = ImpactSenseOrchestrator(NGOS)

# Request/Response Models
class TransactionContext(BaseModel):
    """Transaction context from BPI VIBE"""
    user_id: str = Field(..., description="BPI user identifier")
    transaction_id: str = Field(..., description="Unique transaction ID")
    amount: float = Field(..., gt=0, description="Transaction amount in PHP")
    merchant_category: str = Field(..., description="Transaction category")
    location: Dict[str, str] = Field(..., description="User location data")
    timestamp: datetime = Field(default_factory=datetime.now)

class UserProfile(BaseModel):
    """User profile data from BPI systems"""
    wallet_balance: float = Field(..., ge=0)
    last_donation_date: Optional[datetime] = None
    last_prompt_date: Optional[datetime] = None
    total_lifetime_donations: float = Field(default=0, ge=0)
    average_donation_amount: float = Field(default=0, ge=0)
    preferred_causes: List[str] = Field(default=[])
    notification_preferences: Dict[str, bool] = Field(default={})
    demographic_hints: Dict[str, Any] = Field(default={})

class DonationPromptRequest(BaseModel):
    """Request for donation prompt evaluation"""
    transaction: TransactionContext
    user_profile: UserProfile

class DonationPromptResponse(BaseModel):
    """Response with donation prompt or skip signal"""
    show_prompt: bool
    prompt_data: Optional[Dict[str, Any]] = None
    next_evaluation_hours: int
    reason: str

class DonationSubmission(BaseModel):
    """User donation submission"""
    user_id: str
    transaction_id: str
    ngo_id: str
    amount: float = Field(..., gt=0)
    cause_category: str
    user_message: Optional[str] = None

class DonationConfirmation(BaseModel):
    """Donation confirmation response"""
    donation_id: str
    status: str
    receipt_number: str
    estimated_impact: str
    proof_timeline: str

class ImpactUpdate(BaseModel):
    """Impact proof update from NGOs"""
    donation_id: str
    ngo_id: str
    update_type: str  # 'photo', 'milestone', 'completion'
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    gps_coordinates: Optional[Dict[str, float]] = None

# Helper Functions
def calculate_days_since(date: Optional[datetime]) -> int:
    """Calculate days since a given date"""
    if date is None:
        return 999  # Large number for "never"
    return (datetime.now() - date).days

def build_user_context(transaction: TransactionContext, profile: UserProfile) -> UserContext:
    """Convert API models to agent UserContext"""
    return UserContext(
        user_id=transaction.user_id,
        transaction_amount=transaction.amount,
        wallet_balance=profile.wallet_balance,
        location=transaction.location,
        transaction_category=transaction.merchant_category,
        days_since_last_prompt=calculate_days_since(profile.last_prompt_date),
        days_since_last_donation=calculate_days_since(profile.last_donation_date),
        average_donation_amount=profile.average_donation_amount,
        total_lifetime_donations=profile.total_lifetime_donations,
        preferred_causes=profile.preferred_causes,
        notification_preferences=profile.notification_preferences,
        demographic_hints=profile.demographic_hints
    )

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint for BPI monitoring"""
    return {
        "status": "healthy",
        "service": "ImpactSense",
        "version": "0.9.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/evaluate-donation-prompt", response_model=DonationPromptResponse)
async def evaluate_donation_prompt(request: DonationPromptRequest):
    """
    Evaluate whether to show donation prompt for a transaction
    
    Called by BPI VIBE after successful payment processing
    """
    try:
        # Build user context for agents
        user_context = build_user_context(request.transaction, request.user_profile)
        
        # Generate recommendation using agent orchestrator
        recommendation = orchestrator.generate_recommendation(user_context)
        
        if recommendation is None:
            return DonationPromptResponse(
                show_prompt=False,
                next_evaluation_hours=24,
                reason="user_not_receptive"
            )
        
        # Format prompt data for BPI VIBE UI
        prompt_data = {
            "ngo_name": recommendation.cause.ngo_name,
            "cause_category": recommendation.cause.cause_category,
            "impact_description": recommendation.cause.impact_description,
            "suggested_amount": recommendation.primary_amount,
            "alternative_amounts": recommendation.alternative_amounts,
            "urgency_level": recommendation.cause.urgency_level,
            "region_focus": recommendation.cause.region_focus,
            "ngo_id": recommendation.cause.ngo_id,
            "likelihood_score": recommendation.likelihood_score,
            "community_context": f"Join others in {request.transaction.location.get('region', 'your area')} supporting this cause"
        }
        
        return DonationPromptResponse(
            show_prompt=True,
            prompt_data=prompt_data,
            next_evaluation_hours=168,  # 1 week
            reason="high_likelihood_and_relevant_cause"
        )
        
    except Exception as e:
        logger.error(f"Error evaluating donation prompt: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/submit-donation", response_model=DonationConfirmation)
async def submit_donation(donation: DonationSubmission, background_tasks: BackgroundTasks):
    """
    Process user donation submission
    
    Called when user confirms donation through BPI VIBE UI
    """
    try:
        # Generate unique donation ID
        donation_id = str(uuid.uuid4())
        
        # Find NGO details
        ngo = next((ngo for ngo in NGOS if ngo['ngo_id'] == donation.ngo_id), None)
        if not ngo:
            raise HTTPException(status_code=400, detail="Invalid NGO ID")
        
        # Process donation (integrate with BPI payment processing)
        receipt_number = f"IS{datetime.now().strftime('%Y%m%d')}{donation_id[:8].upper()}"
        
        # Calculate estimated impact
        estimated_impact = calculate_estimated_impact(donation.amount, donation.cause_category)
        
        # Schedule background tasks
        background_tasks.add_task(notify_ngo_of_donation, donation_id, donation)
        background_tasks.add_task(update_user_profile, donation.user_id, donation)
        background_tasks.add_task(schedule_impact_follow_up, donation_id)
        
        return DonationConfirmation(
            donation_id=donation_id,
            status="confirmed",
            receipt_number=receipt_number,
            estimated_impact=estimated_impact,
            proof_timeline="Expect impact updates within 7 days"
        )
        
    except Exception as e:
        logger.error(f"Error processing donation: {str(e)}")
        raise HTTPException(status_code=500, detail="Donation processing failed")

@app.get("/user/{user_id}/impact-summary")
async def get_user_impact_summary(user_id: str, period: str = "year"):
    """
    Get user's impact summary for M-PACSense Wrap
    
    Returns aggregated impact data for gamification features
    """
    try:
        # This would integrate with your donation tracking database
        # For now, returning mock data structure
        
        impact_summary = {
            "total_donated": 234.50,
            "donations_count": 47,
            "causes_supported": ["Health", "Education", "Environment"],
            "ngos_partnered": 8,
            "regions_helped": 3,
            "specific_impacts": {
                "students_supported": 12,
                "meals_provided": 89,
                "trees_planted": 23,
                "clean_water_liters": 450
            },
            "badges_earned": ["First Donor", "Local Hero", "Consistent Giver"],
            "giving_streak": {
                "current_streak_months": 4,
                "longest_streak_months": 6
            },
            "community_rank": "Top 15% of donors in NCR",
            "next_milestone": {
                "target": "₱250 total donated",
                "progress": 93.8,
                "reward": "Community Champion badge"
            }
        }
        
        return impact_summary
        
    except Exception as e:
        logger.error(f"Error retrieving impact summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Unable to retrieve impact summary")

@app.post("/ngo/{ngo_id}/submit-proof")
async def submit_impact_proof(ngo_id: str, proof: ImpactUpdate, background_tasks: BackgroundTasks):
    """
    Receive impact proof submissions from NGO partners
    
    Called by NGOs to submit photos, videos, milestone updates
    """
    try:
        # Validate NGO
        ngo = next((ngo for ngo in NGOS if ngo['ngo_id'] == ngo_id), None)
        if not ngo:
            raise HTTPException(status_code=400, detail="Invalid NGO ID")
        
        # Process and validate proof
        proof_id = str(uuid.uuid4())
        
        # Schedule background validation and user notification
        background_tasks.add_task(validate_proof_authenticity, proof_id, proof)
        background_tasks.add_task(notify_donors_of_impact, proof.donation_id, proof)
        
        return {
            "proof_id": proof_id,
            "status": "received",
            "validation_timeline": "Processing within 24 hours",
            "donor_notification": "Will notify donors once validated"
        }
        
    except Exception as e:
        logger.error(f"Error processing impact proof: {str(e)}")
        raise HTTPException(status_code=500, detail="Proof submission failed")

@app.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """
    Analytics dashboard for BPI stakeholders
    
    Provides aggregate metrics on ImpactSense performance
    """
    try:
        # This would integrate with your analytics database
        analytics_data = {
            "total_donations": {
                "amount": 125420.75,
                "count": 3847,
                "growth_rate": 23.5  # % increase from last period
            },
            "user_engagement": {
                "opt_in_rate": 31.2,  # % of users who donate when prompted
                "retention_rate": 67.8,  # % who donate again within 30 days
                "avg_donations_per_user": 2.8
            },
            "ngo_performance": {
                "avg_proof_submission_time": 4.2,  # days
                "proof_quality_score": 87.3,  # % 
                "user_satisfaction": 4.6  # out of 5
            },
            "popular_causes": [
                {"category": "Education", "percentage": 28.5},
                {"category": "Health", "percentage": 24.1},
                {"category": "Poverty Alleviation", "percentage": 18.3},
                {"category": "Environment", "percentage": 15.7},
                {"category": "Disaster Relief", "percentage": 13.4}
            ],
            "regional_distribution": {
                "NCR": 42.1,
                "Luzon": 31.8,
                "Visayas": 16.3,
                "Mindanao": 9.8
            }
        }
        
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error retrieving analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Analytics unavailable")

# Background Task Functions
async def notify_ngo_of_donation(donation_id: str, donation: DonationSubmission):
    """Notify NGO of new donation"""
    logger.info(f"Notifying NGO {donation.ngo_id} of donation {donation_id}")
    # Implementation would send notification to NGO systems

async def update_user_profile(user_id: str, donation: DonationSubmission):
    """Update user profile with new donation data"""
    logger.info(f"Updating profile for user {user_id} with donation data")
    # Implementation would update user profile in database

async def schedule_impact_follow_up(donation_id: str):
    """Schedule follow-up for impact proof collection"""
    logger.info(f"Scheduling impact follow-up for donation {donation_id}")
    # Implementation would add to task queue for proof collection

async def validate_proof_authenticity(proof_id: str, proof: ImpactUpdate):
    """Validate submitted impact proof"""
    logger.info(f"Validating proof {proof_id}")
    # Implementation would run validation algorithms

async def notify_donors_of_impact(donation_id: str, proof: ImpactUpdate):
    """Notify donors when impact proof is available"""
    logger.info(f"Notifying donors of impact for donation {donation_id}")
    # Implementation would send push notifications to users

# Helper Functions
def calculate_estimated_impact(amount: float, category: str) -> str:
    """Calculate human-readable impact estimate"""
    impact_rates = {
        "Health": f"{int(amount * 2)} medical consultations funded",
        "Education": f"{int(amount / 5)} school supplies provided",
        "Nutrition": f"{int(amount * 3)} nutritious meals funded",
        "Environment": f"{int(amount / 10)} trees to be planted",
        "Poverty Alleviation": f"{int(amount / 2)} livelihood support hours",
        "Disaster Relief": f"{int(amount * 1.5)} emergency aid packages"
    }
    
    return impact_rates.get(category, f"₱{amount} in direct support")

# WebSocket endpoint for real-time updates (optional)
@app.websocket("/ws/user/{user_id}")
async def websocket_endpoint(websocket, user_id: str):
    """
    WebSocket for real-time impact updates
    
    Allows BPI VIBE to show live updates to users
    """
    await websocket.accept()
    try:
        while True:
            # Listen for impact updates for this user
            # Send real-time notifications
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
