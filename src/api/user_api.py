"""
User Management API Module
Handles user profiles, preferences, and transaction history
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import hashlib

router = APIRouter(prefix="/api/v1/users", tags=["users"])
security = HTTPBearer()

# In-memory storage (in production, this would be a database)
user_database = {}
user_sessions = {}

# API Models
class UserRegistration(BaseModel):
    email: EmailStr
    full_name: str
    phone_number: Optional[str] = None
    preferred_language: str = "en"
    location: Dict[str, str] = {}
    demographic_hints: Dict[str, Any] = {}

class UserProfile(BaseModel):
    user_id: str
    email: str
    full_name: str
    phone_number: Optional[str] = None
    preferred_language: str = "en"
    location: Dict[str, str]
    demographic_hints: Dict[str, Any]
    wallet_balance: float = 0.0
    preferred_causes: List[str] = []
    notification_preferences: Dict[str, bool] = {
        "donation_prompts": True,
        "cause_updates": True,
        "impact_reports": True,
        "promotional": False
    }
    privacy_settings: Dict[str, bool] = {
        "share_donation_history": False,
        "personalized_recommendations": True,
        "analytics_tracking": True
    }
    created_at: datetime
    last_active: datetime

class UserPreferencesUpdate(BaseModel):
    preferred_causes: Optional[List[str]] = None
    notification_preferences: Optional[Dict[str, bool]] = None
    privacy_settings: Optional[Dict[str, bool]] = None
    location: Optional[Dict[str, str]] = None
    demographic_hints: Optional[Dict[str, Any]] = None

class TransactionRecord(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    category: str
    merchant: str
    location: Dict[str, str]
    payment_method: str
    timestamp: datetime
    prompt_shown: bool = False
    donation_made: bool = False
    donation_amount: Optional[float] = None
    selected_ngo: Optional[str] = None

class DonationRecord(BaseModel):
    donation_id: str
    user_id: str
    ngo_id: str
    ngo_name: str
    cause_category: str
    amount: float
    transaction_id: Optional[str] = None
    agent_recommendation_id: Optional[str] = None
    timestamp: datetime
    status: str = "completed"  # pending, completed, failed, refunded
    impact_data: Optional[Dict[str, Any]] = None

class UserAnalytics(BaseModel):
    user_id: str
    period: str
    total_transactions: int
    total_donations: int
    total_donated_amount: float
    favorite_causes: List[str]
    donation_frequency: str
    average_donation_amount: float
    response_to_prompts: Dict[str, int]
    impact_summary: Dict[str, Any]

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract user ID from authorization token (simplified for demo)"""
    token = credentials.credentials
    # In production, this would verify JWT tokens
    # For demo, we'll use a simple hash-based approach
    if token.startswith("user_"):
        return token
    else:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

# User Management Endpoints

@router.post("/register")
async def register_user(registration: UserRegistration) -> Dict[str, Any]:
    """
    Register a new user
    
    Creates a new user profile with the provided information.
    """
    try:
        # Generate user ID
        user_id = f"user_{hashlib.md5(registration.email.encode()).hexdigest()[:8]}"
        
        # Check if user already exists
        if user_id in user_database:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create user profile
        user_profile = UserProfile(
            user_id=user_id,
            email=registration.email,
            full_name=registration.full_name,
            phone_number=registration.phone_number,
            preferred_language=registration.preferred_language,
            location=registration.location,
            demographic_hints=registration.demographic_hints,
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        # Store in database
        user_database[user_id] = user_profile
        
        # Create session token (simplified)
        session_token = f"user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_sessions[session_token] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=30)
        }
        
        return {
            "status": "success",
            "message": "User registered successfully",
            "user_id": user_id,
            "session_token": session_token,
            "profile": {
                "user_id": user_profile.user_id,
                "email": user_profile.email,
                "full_name": user_profile.full_name,
                "preferred_language": user_profile.preferred_language,
                "location": user_profile.location
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering user: {str(e)}")

@router.get("/profile")
async def get_user_profile(current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get user profile information
    
    Returns the complete profile for the authenticated user.
    """
    try:
        if current_user not in user_database:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_profile = user_database[current_user]
        
        # Update last active timestamp
        user_profile.last_active = datetime.now()
        
        return {
            "user_id": user_profile.user_id,
            "email": user_profile.email,
            "full_name": user_profile.full_name,
            "phone_number": user_profile.phone_number,
            "preferred_language": user_profile.preferred_language,
            "location": user_profile.location,
            "demographic_hints": user_profile.demographic_hints,
            "wallet_balance": user_profile.wallet_balance,
            "preferred_causes": user_profile.preferred_causes,
            "notification_preferences": user_profile.notification_preferences,
            "privacy_settings": user_profile.privacy_settings,
            "created_at": user_profile.created_at.isoformat(),
            "last_active": user_profile.last_active.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user profile: {str(e)}")

@router.put("/profile")
async def update_user_profile(
    updates: UserPreferencesUpdate,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update user profile and preferences
    
    Allows users to update their preferences, settings, and profile information.
    """
    try:
        if current_user not in user_database:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_profile = user_database[current_user]
        
        # Update provided fields
        if updates.preferred_causes is not None:
            user_profile.preferred_causes = updates.preferred_causes
        
        if updates.notification_preferences is not None:
            user_profile.notification_preferences.update(updates.notification_preferences)
        
        if updates.privacy_settings is not None:
            user_profile.privacy_settings.update(updates.privacy_settings)
        
        if updates.location is not None:
            user_profile.location.update(updates.location)
        
        if updates.demographic_hints is not None:
            user_profile.demographic_hints.update(updates.demographic_hints)
        
        # Update last active timestamp
        user_profile.last_active = datetime.now()
        
        return {
            "status": "success",
            "message": "Profile updated successfully",
            "updated_fields": [
                field for field, value in updates.dict().items() 
                if value is not None
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating user profile: {str(e)}")

@router.post("/transactions")
async def record_transaction(
    transaction: TransactionRecord,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Record a user transaction
    
    Stores transaction data for analysis and donation prompt optimization.
    """
    try:
        # Verify user owns this transaction
        if transaction.user_id != current_user:
            raise HTTPException(status_code=403, detail="Cannot record transaction for another user")
        
        # Validate transaction data
        if transaction.amount <= 0:
            raise HTTPException(status_code=400, detail="Transaction amount must be positive")
        
        # Store transaction (in production, this would be in a database)
        transaction_key = f"txn_{transaction.transaction_id}"
        
        # In a real system, we'd store this in a proper database
        # For demo, we'll just validate and acknowledge
        
        # Update user's wallet balance if it's a withdrawal
        if current_user in user_database:
            user_profile = user_database[current_user]
            if transaction.category in ['withdrawal', 'transfer_out']:
                user_profile.wallet_balance = max(0, user_profile.wallet_balance - transaction.amount)
            elif transaction.category in ['deposit', 'transfer_in']:
                user_profile.wallet_balance += transaction.amount
        
        return {
            "status": "success",
            "message": "Transaction recorded successfully",
            "transaction_id": transaction.transaction_id,
            "eligible_for_prompt": transaction.amount > 50 and not transaction.prompt_shown,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording transaction: {str(e)}")

@router.get("/transactions")
async def get_user_transactions(
    current_user: str = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get user transaction history
    
    Returns paginated transaction history for the authenticated user.
    """
    try:
        # In production, this would query a database
        # For demo, we'll return sample transactions
        
        sample_transactions = [
            {
                "transaction_id": f"txn_{current_user}_{i:03d}",
                "amount": 150.0 + (i * 25),
                "category": ["food", "transport", "shopping", "entertainment"][i % 4],
                "merchant": f"Merchant {i+1}",
                "location": {"region": "NCR", "city": "Manila"},
                "payment_method": "wallet",
                "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                "prompt_shown": i % 3 == 0,
                "donation_made": i % 5 == 0
            }
            for i in range(min(limit, 20))  # Return up to 20 sample transactions
        ]
        
        # Apply category filter if provided
        if category:
            sample_transactions = [
                txn for txn in sample_transactions 
                if txn["category"] == category
            ]
        
        # Apply pagination
        paginated_transactions = sample_transactions[offset:offset + limit]
        
        return {
            "transactions": paginated_transactions,
            "total_count": len(sample_transactions),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(sample_transactions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving transactions: {str(e)}")

@router.post("/donations")
async def record_donation(
    donation: DonationRecord,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Record a user donation
    
    Stores donation data and updates user profile with giving history.
    """
    try:
        # Verify user owns this donation
        if donation.user_id != current_user:
            raise HTTPException(status_code=403, detail="Cannot record donation for another user")
        
        # Validate donation data
        if donation.amount <= 0:
            raise HTTPException(status_code=400, detail="Donation amount must be positive")
        
        # Update user profile with donation history
        if current_user in user_database:
            user_profile = user_database[current_user]
            
            # Add to preferred causes if not already there
            if donation.cause_category not in user_profile.preferred_causes:
                user_profile.preferred_causes.append(donation.cause_category)
            
            # Deduct from wallet balance
            if user_profile.wallet_balance >= donation.amount:
                user_profile.wallet_balance -= donation.amount
            else:
                raise HTTPException(status_code=400, detail="Insufficient wallet balance")
        
        # Store donation record (in production, this would be in a database)
        donation_key = f"donation_{donation.donation_id}"
        
        # Generate impact data (simplified)
        impact_data = {
            "beneficiaries_helped": max(1, int(donation.amount / 25)),
            "impact_category": donation.cause_category,
            "estimated_reach": f"{donation.amount * 2:.0f} people indirectly helped",
            "ngo_efficiency_rating": 0.87
        }
        
        return {
            "status": "success",
            "message": "Donation recorded successfully",
            "donation_id": donation.donation_id,
            "impact_data": impact_data,
            "receipt_url": f"/api/v1/users/donations/{donation.donation_id}/receipt",
            "tax_deductible": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording donation: {str(e)}")

@router.get("/donations")
async def get_user_donations(
    current_user: str = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
    cause_category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get user donation history
    
    Returns paginated donation history for the authenticated user.
    """
    try:
        # In production, this would query a database
        # For demo, we'll return sample donations
        
        sample_donations = [
            {
                "donation_id": f"don_{current_user}_{i:03d}",
                "ngo_id": f"ngo_{(i % 5) + 1:03d}",
                "ngo_name": ["Gawad Kalinga", "World Vision", "Red Cross", "Teach for Philippines", "Bantay Bata"][i % 5],
                "cause_category": ["housing", "children", "disaster_relief", "education", "children"][i % 5],
                "amount": 25.0 + (i * 15),
                "timestamp": (datetime.now() - timedelta(days=i * 7)).isoformat(),
                "status": "completed",
                "impact_data": {
                    "beneficiaries_helped": max(1, int((25.0 + i * 15) / 25)),
                    "impact_description": f"Helped {max(1, int((25.0 + i * 15) / 25))} beneficiaries"
                }
            }
            for i in range(min(limit, 10))  # Return up to 10 sample donations
        ]
        
        # Apply category filter if provided
        if cause_category:
            sample_donations = [
                don for don in sample_donations 
                if don["cause_category"] == cause_category
            ]
        
        # Apply pagination
        paginated_donations = sample_donations[offset:offset + limit]
        
        # Calculate summary statistics
        total_donated = sum(don["amount"] for don in sample_donations)
        unique_causes = len(set(don["cause_category"] for don in sample_donations))
        total_beneficiaries = sum(
            don["impact_data"]["beneficiaries_helped"] 
            for don in sample_donations
        )
        
        return {
            "donations": paginated_donations,
            "summary": {
                "total_donated": total_donated,
                "total_donations": len(sample_donations),
                "unique_causes_supported": unique_causes,
                "total_beneficiaries_helped": total_beneficiaries,
                "average_donation": total_donated / max(1, len(sample_donations))
            },
            "total_count": len(sample_donations),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(sample_donations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving donations: {str(e)}")

@router.get("/analytics")
async def get_user_analytics(
    current_user: str = Depends(get_current_user),
    period: str = "30d"
) -> UserAnalytics:
    """
    Get user analytics and insights
    
    Returns comprehensive analytics about user's giving patterns and impact.
    """
    try:
        # Parse period parameter
        if period == "7d":
            days = 7
        elif period == "30d":
            days = 30
        elif period == "90d":
            days = 90
        elif period == "1y":
            days = 365
        else:
            days = 30
        
        # In production, this would aggregate data from database
        # For demo, we'll return calculated analytics
        
        analytics = UserAnalytics(
            user_id=current_user,
            period=period,
            total_transactions=45,
            total_donations=8,
            total_donated_amount=350.0,
            favorite_causes=["education", "children", "healthcare"],
            donation_frequency="bi-weekly",
            average_donation_amount=43.75,
            response_to_prompts={
                "accepted": 8,
                "rejected": 12,
                "modified": 3,
                "ignored": 7
            },
            impact_summary={
                "total_beneficiaries_helped": 28,
                "causes_supported": 3,
                "top_impact_area": "education",
                "social_reach": "approximately 84 people indirectly helped",
                "environmental_impact": "2.1 kg CO2 offset through sustainable projects"
            }
        )
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating user analytics: {str(e)}")

@router.get("/impact-report")
async def get_impact_report(
    current_user: str = Depends(get_current_user),
    period: str = "30d"
) -> Dict[str, Any]:
    """
    Generate a comprehensive impact report
    
    Returns detailed impact report showing how user donations have made a difference.
    """
    try:
        # Generate comprehensive impact report
        impact_report = {
            "user_id": current_user,
            "period": period,
            "generated_at": datetime.now().isoformat(),
            "overall_impact": {
                "total_donated": 350.0,
                "lives_touched": 28,
                "projects_supported": 5,
                "ngos_helped": 3
            },
            "cause_breakdown": [
                {
                    "cause": "education",
                    "amount_donated": 150.0,
                    "percentage": 42.9,
                    "impact": "Provided school supplies for 6 children",
                    "specific_outcomes": [
                        "3 children completed literacy program",
                        "2 students received scholarships",
                        "1 classroom equipped with learning materials"
                    ]
                },
                {
                    "cause": "children",
                    "amount_donated": 125.0,
                    "percentage": 35.7,
                    "impact": "Protected and fed 10 children",
                    "specific_outcomes": [
                        "5 children received medical care",
                        "8 children fed for a week",
                        "2 children rescued from dangerous situations"
                    ]
                },
                {
                    "cause": "healthcare",
                    "amount_donated": 75.0,
                    "percentage": 21.4,
                    "impact": "Medical supplies for 12 patients",
                    "specific_outcomes": [
                        "Emergency medical kit for 3 families",
                        "Vaccination supplies for 12 children",
                        "First aid training for 1 community"
                    ]
                }
            ],
            "ngo_partnerships": [
                {
                    "ngo_name": "Gawad Kalinga",
                    "total_donated": 125.0,
                    "projects_supported": 2,
                    "impact_highlights": "Helped build 1 home, supported 2 families"
                },
                {
                    "ngo_name": "World Vision Philippines",
                    "total_donated": 150.0,
                    "projects_supported": 2,
                    "impact_highlights": "Sponsored 2 children's education for 3 months"
                },
                {
                    "ngo_name": "Philippine Red Cross",
                    "total_donated": 75.0,
                    "projects_supported": 1,
                    "impact_highlights": "Emergency response kit for disaster victims"
                }
            ],
            "social_proof": {
                "ranking_among_peers": "Top 15% of donors in your area",
                "community_impact": "Your donations inspired 3 friends to start giving",
                "consistency_score": "85% - Very consistent giver"
            },
            "future_opportunities": [
                {
                    "cause": "disaster_relief",
                    "reason": "Upcoming typhoon season",
                    "potential_impact": "Help prepare communities for disasters",
                    "suggested_amount": 50.0
                },
                {
                    "cause": "environment",
                    "reason": "Climate action month",
                    "potential_impact": "Support reforestation efforts",
                    "suggested_amount": 35.0
                }
            ],
            "recognition": {
                "badges_earned": ["Consistent Giver", "Education Champion", "Child Protector"],
                "milestone_next": "500 PHP total donated",
                "impact_level": "Community Helper"
            }
        }
        
        return impact_report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating impact report: {str(e)}")

@router.delete("/profile")
async def delete_user_account(
    current_user: str = Depends(get_current_user),
    confirmation: str = ""
) -> Dict[str, Any]:
    """
    Delete user account and all associated data
    
    Permanently removes user account and all personal data (GDPR compliance).
    """
    try:
        if confirmation != "DELETE_MY_ACCOUNT":
            raise HTTPException(
                status_code=400, 
                detail="Must provide confirmation string 'DELETE_MY_ACCOUNT'"
            )
        
        if current_user not in user_database:
            raise HTTPException(status_code=404, detail="User not found")
        
        # In production, this would:
        # 1. Anonymize donation records (keep for NGO reporting)
        # 2. Delete all personal data
        # 3. Remove from all systems
        # 4. Send confirmation email
        
        # Remove user from database
        del user_database[current_user]
        
        # Remove all sessions for this user
        sessions_to_remove = [
            token for token, session in user_sessions.items()
            if session["user_id"] == current_user
        ]
        for token in sessions_to_remove:
            del user_sessions[token]
        
        return {
            "status": "success",
            "message": "Account deleted successfully",
            "deleted_at": datetime.now().isoformat(),
            "data_retention_note": "Anonymized donation records retained for NGO impact reporting"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting account: {str(e)}")

@router.get("/export")
async def export_user_data(current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Export all user data (GDPR compliance)
    
    Returns all user data in a portable format for data portability requirements.
    """
    try:
        if current_user not in user_database:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_profile = user_database[current_user]
        
        # Compile all user data
        export_data = {
            "export_generated_at": datetime.now().isoformat(),
            "data_format_version": "1.0",
            "user_profile": {
                "user_id": user_profile.user_id,
                "email": user_profile.email,
                "full_name": user_profile.full_name,
                "phone_number": user_profile.phone_number,
                "preferred_language": user_profile.preferred_language,
                "location": user_profile.location,
                "demographic_hints": user_profile.demographic_hints,
                "preferred_causes": user_profile.preferred_causes,
                "notification_preferences": user_profile.notification_preferences,
                "privacy_settings": user_profile.privacy_settings,
                "created_at": user_profile.created_at.isoformat(),
                "last_active": user_profile.last_active.isoformat()
            },
            "financial_data": {
                "current_wallet_balance": user_profile.wallet_balance
            },
            "donation_history": [
                # In production, this would include all donations
                {"note": "Donation history would be included here"}
            ],
            "transaction_history": [
                # In production, this would include all transactions
                {"note": "Transaction history would be included here"}
            ],
            "agent_interactions": [
                # In production, this would include all agent interactions
                {"note": "Agent interaction history would be included here"}
            ],
            "preferences_history": [
                # In production, this would include preference change history
                {"note": "Preference change history would be included here"}
            ]
        }
        
        return export_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting user data: {str(e)}")
