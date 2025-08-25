"""
NGO and Analytics API Module
Handles NGO information, analytics, and platform insights
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import statistics

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

# In-memory storage for demo (in production, this would be a database)
ngo_database = {}
platform_metrics = {}

# API Models
class NGOProfile(BaseModel):
    ngo_id: str
    name: str
    legal_name: str
    registration_number: str
    cause_category: str
    subcategories: List[str]
    description: str
    mission_statement: str
    website: Optional[str] = None
    contact_email: str
    phone_number: Optional[str] = None
    address: Dict[str, str]
    region_focus: str
    target_beneficiaries: List[str]
    programs: List[Dict[str, Any]]
    impact_metrics: Dict[str, float]
    financial_transparency: Dict[str, Any]
    efficiency_rating: float
    transparency_score: float
    verification_status: str
    created_at: datetime
    last_updated: datetime

class NGOUpdate(BaseModel):
    description: Optional[str] = None
    mission_statement: Optional[str] = None
    website: Optional[str] = None
    programs: Optional[List[Dict[str, Any]]] = None
    impact_metrics: Optional[Dict[str, float]] = None
    financial_transparency: Optional[Dict[str, Any]] = None

class DonationAnalytics(BaseModel):
    period: str
    total_donations: int
    total_amount: float
    unique_donors: int
    average_donation: float
    top_causes: List[Dict[str, Any]]
    geographic_distribution: Dict[str, float]
    time_series_data: List[Dict[str, Any]]
    conversion_metrics: Dict[str, float]

class NGOPerformance(BaseModel):
    ngo_id: str
    period: str
    total_received: float
    donor_count: int
    average_donation: float
    donor_retention_rate: float
    impact_efficiency: float
    transparency_improvements: List[str]
    top_programs: List[Dict[str, Any]]

class PlatformInsights(BaseModel):
    period: str
    user_engagement: Dict[str, float]
    agent_performance: Dict[str, float]
    cause_trends: List[Dict[str, Any]]
    geographic_insights: Dict[str, Any]
    prediction_accuracy: Dict[str, float]
    business_impact: Dict[str, float]

# Initialize sample NGO data
def initialize_ngo_data():
    """Initialize sample NGO data for demo"""
    sample_ngos = [
        {
            "ngo_id": "ngo_001",
            "name": "Gawad Kalinga",
            "legal_name": "Gawad Kalinga Community Development Foundation Inc.",
            "registration_number": "CN-2003-0123",
            "cause_category": "housing",
            "subcategories": ["community_development", "poverty_alleviation", "youth_empowerment"],
            "description": "Building communities and transforming lives through sustainable development programs",
            "mission_statement": "To end poverty by building sustainable communities",
            "website": "https://gk1world.com",
            "contact_email": "info@gk1world.com",
            "phone_number": "+63-2-123-4567",
            "address": {
                "street": "GK Building, 123 Community Ave",
                "city": "Quezon City",
                "region": "NCR",
                "postal_code": "1100"
            },
            "region_focus": "NCR",
            "target_beneficiaries": ["families_in_poverty", "youth", "communities"],
            "programs": [
                {
                    "name": "GK Village Building",
                    "description": "Building sustainable communities with decent homes",
                    "beneficiaries": 10000,
                    "budget": 50000000
                },
                {
                    "name": "Youth Leadership Program",
                    "description": "Developing young leaders in communities",
                    "beneficiaries": 2500,
                    "budget": 5000000
                }
            ],
            "impact_metrics": {
                "families_housed": 10000,
                "communities_built": 50,
                "youth_trained": 2500,
                "livelihood_programs": 100
            },
            "financial_transparency": {
                "program_expense_ratio": 0.85,
                "admin_expense_ratio": 0.12,
                "fundraising_expense_ratio": 0.03,
                "last_audit_date": "2024-03-15",
                "financial_statements_public": True
            },
            "efficiency_rating": 0.88,
            "transparency_score": 0.95,
            "verification_status": "verified"
        },
        {
            "ngo_id": "ngo_002",
            "name": "World Vision Philippines",
            "legal_name": "World Vision Development Foundation Inc.",
            "registration_number": "CN-1987-0456",
            "cause_category": "children",
            "subcategories": ["education", "healthcare", "nutrition", "child_protection"],
            "description": "Dedicated to working with children, families, and communities to overcome poverty and injustice",
            "mission_statement": "Our vision for every child, life in all its fullness. Our prayer for every heart, the will to make it so.",
            "website": "https://worldvision.org.ph",
            "contact_email": "info@worldvision.org.ph",
            "phone_number": "+63-2-987-6543",
            "address": {
                "street": "WV Building, 456 Children St",
                "city": "Makati City",
                "region": "NCR",
                "postal_code": "1200"
            },
            "region_focus": "nationwide",
            "target_beneficiaries": ["children", "families", "communities"],
            "programs": [
                {
                    "name": "Child Sponsorship Program",
                    "description": "Providing education and healthcare for sponsored children",
                    "beneficiaries": 50000,
                    "budget": 200000000
                },
                {
                    "name": "Emergency Response",
                    "description": "Disaster relief and emergency assistance",
                    "beneficiaries": 25000,
                    "budget": 75000000
                }
            ],
            "impact_metrics": {
                "children_helped": 50000,
                "schools_built": 25,
                "health_clinics": 15,
                "emergency_responses": 50
            },
            "financial_transparency": {
                "program_expense_ratio": 0.82,
                "admin_expense_ratio": 0.15,
                "fundraising_expense_ratio": 0.03,
                "last_audit_date": "2024-02-28",
                "financial_statements_public": True
            },
            "efficiency_rating": 0.90,
            "transparency_score": 0.92,
            "verification_status": "verified"
        }
    ]
    
    for ngo_data in sample_ngos:
        ngo_profile = NGOProfile(
            **ngo_data,
            created_at=datetime.now() - timedelta(days=365),
            last_updated=datetime.now() - timedelta(days=30)
        )
        ngo_database[ngo_data["ngo_id"]] = ngo_profile

# Initialize data on module load
initialize_ngo_data()

# NGO Endpoints

@router.get("/ngos")
async def get_ngos(
    cause_category: Optional[str] = None,
    region: Optional[str] = None,
    verified_only: bool = True,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get list of NGOs with filtering options
    
    Returns paginated list of NGOs with optional filtering by cause and region.
    """
    try:
        ngos = list(ngo_database.values())
        
        # Apply filters
        if cause_category:
            ngos = [ngo for ngo in ngos if ngo.cause_category == cause_category]
        
        if region:
            ngos = [ngo for ngo in ngos if ngo.region_focus == region or ngo.region_focus == "nationwide"]
        
        if verified_only:
            ngos = [ngo for ngo in ngos if ngo.verification_status == "verified"]
        
        # Sort by transparency score
        ngos.sort(key=lambda x: x.transparency_score, reverse=True)
        
        # Apply pagination
        paginated_ngos = ngos[offset:offset + limit]
        
        # Convert to dict for response
        ngo_list = []
        for ngo in paginated_ngos:
            ngo_list.append({
                "ngo_id": ngo.ngo_id,
                "name": ngo.name,
                "cause_category": ngo.cause_category,
                "subcategories": ngo.subcategories,
                "description": ngo.description,
                "region_focus": ngo.region_focus,
                "transparency_score": ngo.transparency_score,
                "efficiency_rating": ngo.efficiency_rating,
                "verification_status": ngo.verification_status,
                "impact_metrics": ngo.impact_metrics
            })
        
        return {
            "ngos": ngo_list,
            "total_count": len(ngos),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(ngos),
            "filters_applied": {
                "cause_category": cause_category,
                "region": region,
                "verified_only": verified_only
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving NGOs: {str(e)}")

@router.get("/ngos/{ngo_id}")
async def get_ngo_details(ngo_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific NGO
    
    Returns comprehensive NGO profile including programs and impact metrics.
    """
    try:
        if ngo_id not in ngo_database:
            raise HTTPException(status_code=404, detail="NGO not found")
        
        ngo = ngo_database[ngo_id]
        
        return {
            "ngo_id": ngo.ngo_id,
            "name": ngo.name,
            "legal_name": ngo.legal_name,
            "registration_number": ngo.registration_number,
            "cause_category": ngo.cause_category,
            "subcategories": ngo.subcategories,
            "description": ngo.description,
            "mission_statement": ngo.mission_statement,
            "website": ngo.website,
            "contact_email": ngo.contact_email,
            "phone_number": ngo.phone_number,
            "address": ngo.address,
            "region_focus": ngo.region_focus,
            "target_beneficiaries": ngo.target_beneficiaries,
            "programs": ngo.programs,
            "impact_metrics": ngo.impact_metrics,
            "financial_transparency": ngo.financial_transparency,
            "efficiency_rating": ngo.efficiency_rating,
            "transparency_score": ngo.transparency_score,
            "verification_status": ngo.verification_status,
            "created_at": ngo.created_at.isoformat(),
            "last_updated": ngo.last_updated.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving NGO details: {str(e)}")

@router.get("/ngos/{ngo_id}/performance")
async def get_ngo_performance(
    ngo_id: str,
    period: str = "30d"
) -> NGOPerformance:
    """
    Get NGO performance metrics
    
    Returns performance analytics for a specific NGO over the requested period.
    """
    try:
        if ngo_id not in ngo_database:
            raise HTTPException(status_code=404, detail="NGO not found")
        
        ngo = ngo_database[ngo_id]
        
        # Generate performance metrics (in production, this would query actual data)
        performance = NGOPerformance(
            ngo_id=ngo_id,
            period=period,
            total_received=125000.0,  # Sample data
            donor_count=145,
            average_donation=862.07,
            donor_retention_rate=0.68,
            impact_efficiency=ngo.efficiency_rating,
            transparency_improvements=[
                "Updated financial statements",
                "Added program impact photos",
                "Improved reporting frequency"
            ],
            top_programs=[
                {
                    "name": program["name"],
                    "donations_received": 50000.0,
                    "beneficiaries_helped": program["beneficiaries"],
                    "efficiency_score": 0.87
                }
                for program in ngo.programs[:3]
            ]
        )
        
        return performance
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving NGO performance: {str(e)}")

# Analytics Endpoints

@router.get("/donations")
async def get_donation_analytics(
    period: str = "30d",
    cause_category: Optional[str] = None,
    region: Optional[str] = None
) -> DonationAnalytics:
    """
    Get donation analytics
    
    Returns comprehensive analytics about donations on the platform.
    """
    try:
        # Parse period
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
        
        # Generate analytics (in production, this would aggregate real data)
        analytics = DonationAnalytics(
            period=period,
            total_donations=1250,
            total_amount=156750.0,
            unique_donors=425,
            average_donation=125.40,
            top_causes=[
                {"cause": "education", "amount": 52297.50, "percentage": 33.4, "donors": 142},
                {"cause": "children", "amount": 47025.00, "percentage": 30.0, "donors": 128},
                {"cause": "healthcare", "amount": 31350.00, "percentage": 20.0, "donors": 98},
                {"cause": "disaster_relief", "amount": 15675.00, "percentage": 10.0, "donors": 45},
                {"cause": "environment", "amount": 10402.50, "percentage": 6.6, "donors": 32}
            ],
            geographic_distribution={
                "NCR": 45.2,
                "CALABARZON": 18.7,
                "Central Luzon": 12.3,
                "Western Visayas": 8.9,
                "Central Visayas": 7.1,
                "Others": 7.8
            },
            time_series_data=[
                {
                    "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                    "donations": max(10, 50 - i + (i % 7) * 5),
                    "amount": max(1000, 6500 - i * 50 + (i % 7) * 250)
                }
                for i in range(min(days, 30))
            ],
            conversion_metrics={
                "prompt_to_donation_rate": 0.285,
                "first_time_donor_rate": 0.34,
                "repeat_donor_rate": 0.66,
                "average_time_to_donate": 2.3,  # days
                "mobile_vs_web_conversion": {"mobile": 0.31, "web": 0.24}
            }
        )
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving donation analytics: {str(e)}")

@router.get("/causes")
async def get_cause_analytics(period: str = "30d") -> Dict[str, Any]:
    """
    Get cause-specific analytics
    
    Returns detailed analytics about different cause categories.
    """
    try:
        cause_analytics = {
            "period": period,
            "cause_performance": [
                {
                    "cause_category": "education",
                    "total_donations": 415,
                    "total_amount": 52297.50,
                    "average_donation": 126.01,
                    "growth_rate": 0.15,
                    "top_ngos": [
                        {"name": "Teach for the Philippines", "amount": 31378.50, "share": 0.60},
                        {"name": "Scholarship Foundation", "amount": 20919.00, "share": 0.40}
                    ],
                    "urgency_score": 0.75,
                    "seasonal_trends": {
                        "peak_months": ["June", "November"],
                        "low_months": ["April", "December"]
                    }
                },
                {
                    "cause_category": "children",
                    "total_donations": 385,
                    "total_amount": 47025.00,
                    "average_donation": 122.14,
                    "growth_rate": 0.22,
                    "top_ngos": [
                        {"name": "World Vision Philippines", "amount": 28215.00, "share": 0.60},
                        {"name": "Bantay Bata 163", "amount": 18810.00, "share": 0.40}
                    ],
                    "urgency_score": 0.85,
                    "seasonal_trends": {
                        "peak_months": ["December", "May"],
                        "low_months": ["February", "August"]
                    }
                },
                {
                    "cause_category": "healthcare",
                    "total_donations": 295,
                    "total_amount": 31350.00,
                    "average_donation": 106.27,
                    "growth_rate": 0.08,
                    "top_ngos": [
                        {"name": "Philippine Red Cross", "amount": 20877.50, "share": 0.665},
                        {"name": "Medical Mission Group", "amount": 10472.50, "share": 0.335}
                    ],
                    "urgency_score": 0.70,
                    "seasonal_trends": {
                        "peak_months": ["October", "January"],
                        "low_months": ["May", "September"]
                    }
                }
            ],
            "emerging_causes": [
                {
                    "cause": "mental_health",
                    "growth_rate": 0.45,
                    "current_share": 0.02,
                    "predicted_share": 0.05
                },
                {
                    "cause": "elderly_care",
                    "growth_rate": 0.35,
                    "current_share": 0.01,
                    "predicted_share": 0.03
                }
            ],
            "cause_correlations": {
                "education_children": 0.72,
                "healthcare_children": 0.68,
                "disaster_relief_healthcare": 0.55
            }
        }
        
        return cause_analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cause analytics: {str(e)}")

@router.get("/platform/insights")
async def get_platform_insights(period: str = "30d") -> PlatformInsights:
    """
    Get platform-wide insights and metrics
    
    Returns comprehensive insights about platform performance and user behavior.
    """
    try:
        insights = PlatformInsights(
            period=period,
            user_engagement={
                "daily_active_users": 2485,
                "monthly_active_users": 15240,
                "session_duration_minutes": 8.7,
                "pages_per_session": 4.2,
                "bounce_rate": 0.23,
                "donation_page_conversion": 0.285
            },
            agent_performance={
                "likelihood_agent_accuracy": 0.847,
                "cause_agent_acceptance": 0.732,
                "amount_agent_conversion": 0.681,
                "overall_satisfaction": 0.789,
                "response_time_ms": 245,
                "uptime_percentage": 99.8
            },
            cause_trends=[
                {
                    "cause": "climate_action",
                    "trend": "rising",
                    "change_percentage": 35.2,
                    "influencing_factors": ["COP28", "typhoon_season", "youth_activism"]
                },
                {
                    "cause": "education",
                    "trend": "stable",
                    "change_percentage": 2.1,
                    "influencing_factors": ["school_year_start", "scholarship_drives"]
                },
                {
                    "cause": "disaster_relief",
                    "trend": "seasonal",
                    "change_percentage": 18.7,
                    "influencing_factors": ["typhoon_season", "earthquake_risk"]
                }
            ],
            geographic_insights={
                "highest_engagement_regions": ["NCR", "CALABARZON", "Central Luzon"],
                "fastest_growing_regions": ["Davao", "Cebu", "Iloilo"],
                "cause_preferences_by_region": {
                    "NCR": ["education", "children", "environment"],
                    "CALABARZON": ["education", "disaster_relief", "healthcare"],
                    "Central Luzon": ["agriculture", "education", "disaster_relief"]
                },
                "donation_patterns": {
                    "urban_vs_rural": {"urban": 0.73, "rural": 0.27},
                    "average_amounts": {"urban": 142.30, "rural": 89.50}
                }
            },
            prediction_accuracy={
                "donation_likelihood": 0.847,
                "amount_optimization": 0.732,
                "cause_recommendation": 0.789,
                "timing_prediction": 0.698,
                "user_segmentation": 0.823
            },
            business_impact={
                "total_platform_donations": 1567500.0,
                "platform_fee_revenue": 31350.0,  # 2% fee
                "user_acquisition_cost": 45.30,
                "user_lifetime_value": 287.60,
                "donation_growth_rate": 0.234,
                "ngo_retention_rate": 0.892
            }
        )
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving platform insights: {str(e)}")

@router.get("/metrics/dashboard")
async def get_dashboard_metrics() -> Dict[str, Any]:
    """
    Get key metrics for dashboard display
    
    Returns essential platform metrics for dashboard visualization.
    """
    try:
        dashboard_metrics = {
            "overview": {
                "total_donations_today": 47,
                "total_amount_today": 5875.00,
                "active_users_today": 1247,
                "conversion_rate_today": 0.297,
                "total_ngos_active": 156,
                "total_causes_supported": 12
            },
            "real_time": {
                "users_online_now": 127,
                "donations_last_hour": 8,
                "amount_last_hour": 950.00,
                "top_cause_last_hour": "education",
                "agent_decisions_last_hour": 45
            },
            "trends": {
                "donations_trend_7d": [
                    {"date": (datetime.now() - timedelta(days=i)).strftime("%m-%d"), 
                     "count": max(30, 50 - i * 2), 
                     "amount": max(3000, 6000 - i * 200)}
                    for i in range(7)
                ],
                "top_performing_ngos_today": [
                    {"name": "World Vision Philippines", "donations": 12, "amount": 1485.00},
                    {"name": "Gawad Kalinga", "donations": 9, "amount": 1125.00},
                    {"name": "Philippine Red Cross", "donations": 8, "amount": 960.00}
                ],
                "cause_distribution_today": {
                    "education": 28.7,
                    "children": 23.4,
                    "healthcare": 18.9,
                    "disaster_relief": 12.8,
                    "environment": 8.5,
                    "others": 7.7
                }
            },
            "agent_status": {
                "likelihood_agent": {"status": "active", "last_decision": "2 min ago", "accuracy": 0.847},
                "cause_agent": {"status": "active", "last_decision": "1 min ago", "accuracy": 0.732},
                "amount_agent": {"status": "active", "last_decision": "3 min ago", "accuracy": 0.681},
                "performance_agent": {"status": "active", "last_analysis": "5 min ago", "insights": 3}
            },
            "alerts": [
                {
                    "type": "info",
                    "message": "Donation volume 15% higher than yesterday",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "type": "warning", 
                    "message": "Amount optimization agent accuracy below threshold",
                    "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat()
                }
            ]
        }
        
        return dashboard_metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dashboard metrics: {str(e)}")

@router.get("/reports/impact")
async def generate_impact_report(
    period: str = "30d",
    format: str = "json"
) -> Dict[str, Any]:
    """
    Generate comprehensive impact report
    
    Returns detailed impact report showing platform's social impact.
    """
    try:
        impact_report = {
            "report_id": f"impact_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "period": period,
            "format": format,
            "executive_summary": {
                "total_impact_amount": 1567500.0,
                "total_beneficiaries": 12540,
                "active_ngos": 156,
                "unique_donors": 3247,
                "causes_supported": 12,
                "communities_reached": 89
            },
            "detailed_impact": {
                "education": {
                    "amount_donated": 522975.00,
                    "beneficiaries": 4185,
                    "specific_outcomes": [
                        "2,100 children received school supplies",
                        "850 students got scholarships",
                        "45 classrooms were equipped",
                        "15 schools received books and materials"
                    ]
                },
                "children": {
                    "amount_donated": 470250.00,
                    "beneficiaries": 3762,
                    "specific_outcomes": [
                        "1,890 children received healthcare",
                        "2,520 children got nutritious meals",
                        "315 children were protected from danger",
                        "127 families received support"
                    ]
                },
                "healthcare": {
                    "amount_donated": 313500.00,
                    "beneficiaries": 2508,
                    "specific_outcomes": [
                        "1,254 patients received medical supplies",
                        "420 families got emergency medical kits",
                        "63 communities received health education",
                        "18 health centers were supported"
                    ]
                }
            },
            "geographic_impact": {
                "regions_served": 17,
                "provinces_reached": 45,
                "municipalities_helped": 234,
                "rural_vs_urban": {"rural": 45.2, "urban": 54.8}
            },
            "innovation_metrics": {
                "ai_driven_matches": 8945,
                "personalization_accuracy": 0.847,
                "donor_satisfaction": 0.789,
                "ngo_efficiency_improvement": 0.156
            },
            "sustainability_indicators": {
                "donor_retention_rate": 0.683,
                "ngo_partnership_growth": 0.234,
                "platform_adoption_rate": 0.289,
                "impact_per_dollar": 3.67
            },
            "testimonials": [
                {
                    "type": "donor",
                    "name": "Maria Santos",
                    "quote": "ImpactSense makes giving so easy and meaningful. I love seeing exactly how my donations help.",
                    "location": "Quezon City"
                },
                {
                    "type": "ngo",
                    "name": "Juan Dela Cruz, Program Director",
                    "organization": "Education First Foundation",
                    "quote": "The platform has helped us reach more donors and fund critical education programs.",
                    "location": "Cebu"
                }
            ],
            "future_projections": {
                "predicted_impact_next_quarter": 2100000.0,
                "estimated_beneficiaries_next_quarter": 16800,
                "target_new_ngos": 50,
                "expansion_regions": ["Mindanao", "Remote Islands"]
            }
        }
        
        return impact_report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating impact report: {str(e)}")

@router.get("/health")
async def platform_health_check() -> Dict[str, Any]:
    """
    Platform health check
    
    Returns current health status of all platform components.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": {"status": "up", "response_time_ms": 45},
            "agents": {"status": "up", "active_count": 4},
            "api": {"status": "up", "response_time_ms": 23},
            "analytics": {"status": "up", "last_update": "2 min ago"}
        },
        "metrics": {
            "uptime_percentage": 99.8,
            "avg_response_time_ms": 156,
            "error_rate": 0.002,
            "active_connections": 247
        }
    }
