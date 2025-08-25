"""
Simplified Data Generator for ImpactSense Dashboard
Creates demo data for the dashboard without external dependencies
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import math


class SimplifiedDataGenerator:
    """Generate demo data for dashboard testing"""
    
    def __init__(self):
        self.causes = [
            'Education', 'Environment', 'Health', 'Disaster Relief', 
            'Poverty', 'Animal Welfare', 'Arts & Culture', 'Technology'
        ]
        
        self.locations = [
            {'city': 'Manila', 'region': 'NCR'},
            {'city': 'Cebu', 'region': 'Central Visayas'},
            {'city': 'Davao', 'region': 'Davao Region'},
            {'city': 'Quezon City', 'region': 'NCR'},
            {'city': 'Iloilo', 'region': 'Western Visayas'}
        ]
        
        self.user_segments = ['new', 'regular', 'major', 'inactive']
    
    def generate_users(self, num_users: int = 1000) -> List[Dict[str, Any]]:
        """Generate synthetic user data"""
        
        users = []
        
        for i in range(num_users):
            user = {
                'id': f'user_{i+1:04d}',
                'name': f'User {i+1}',
                'email': f'user{i+1}@example.com',
                'wallet_balance': round(random.uniform(100, 20000), 2),
                'segment': random.choice(self.user_segments),
                'registration_date': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                'location': random.choice(self.locations),
                'preferences': {
                    'causes': random.sample(self.causes, random.randint(1, 3)),
                    'max_donation': round(random.uniform(5, 100), 2),
                    'frequency': random.choice(['daily', 'weekly', 'monthly'])
                },
                'donation_history': [],
                'bpi_profile': {
                    'avg_transaction': round(random.uniform(200, 5000), 2),
                    'total_transactions': random.randint(10, 500),
                    'account_age_days': random.randint(30, 1095)
                }
            }
            
            # Generate donation history
            num_donations = random.randint(0, 50)
            for j in range(num_donations):
                donation = {
                    'amount': round(random.uniform(1, user['preferences']['max_donation']), 2),
                    'cause': random.choice(user['preferences']['causes']),
                    'timestamp': (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat(),
                    'ngo_id': f'ngo_{random.randint(1, 50):03d}'
                }
                user['donation_history'].append(donation)
            
            users.append(user)
        
        return users
    
    def generate_transactions(self, num_transactions: int = 10000) -> List[Dict[str, Any]]:
        """Generate synthetic transaction data"""
        
        transactions = []
        categories = [
            'groceries', 'dining', 'transportation', 'shopping', 'utilities',
            'entertainment', 'health', 'education', 'fuel', 'bills'
        ]
        
        for i in range(num_transactions):
            transaction = {
                'id': f'txn_{i+1:06d}',
                'user_id': f'user_{random.randint(1, 1000):04d}',
                'amount': round(random.uniform(50, 5000), 2),
                'category': random.choice(categories),
                'merchant': f'Merchant {random.randint(1, 100)}',
                'location': random.choice(self.locations),
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                'prompt_shown': random.choice([True, False]),
                'donation_made': False
            }
            
            # If prompt was shown, determine if donation was made
            if transaction['prompt_shown']:
                transaction['donation_made'] = random.random() < 0.25  # 25% conversion rate
            
            transactions.append(transaction)
        
        return transactions
    
    def generate_agent_decisions(self, num_decisions: int = 5000) -> List[Dict[str, Any]]:
        """Generate synthetic agent decision data"""
        
        decisions = []
        agents = [
            'DonationLikelyScoreAgent',
            'MLDonationLikelyScoreAgent',
            'LocalCauseRecommenderAgent',
            'DonationAmountOptimizerAgent',
            'MLDonationAmountOptimizerAgent'
        ]
        
        for i in range(num_decisions):
            agent = random.choice(agents)
            
            decision = {
                'id': f'decision_{i+1:06d}',
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                'agent_name': agent,
                'user_id': f'user_{random.randint(1, 1000):04d}',
                'decision_type': 'recommendation',
                'context': {
                    'transaction_amount': round(random.uniform(100, 3000), 2),
                    'wallet_balance': round(random.uniform(500, 15000), 2),
                    'user_segment': random.choice(self.user_segments)
                }
            }
            
            # Generate decision based on agent type
            if 'LikelyScore' in agent:
                likelihood_score = random.uniform(0, 100)
                decision['decision_data'] = {
                    'likelihood_score': round(likelihood_score, 2),
                    'recommendation': 'prompt_now' if likelihood_score > 60 else 'skip_prompt',
                    'confidence': random.uniform(0.5, 0.95)
                }
            elif 'Recommender' in agent:
                decision['decision_data'] = {
                    'primary_recommendation': {
                        'cause': random.choice(self.causes),
                        'ngo_id': f'ngo_{random.randint(1, 50):03d}',
                        'relevance_score': random.uniform(0.6, 0.95)
                    },
                    'alternatives': [
                        {
                            'cause': random.choice(self.causes),
                            'ngo_id': f'ngo_{random.randint(1, 50):03d}',
                            'relevance_score': random.uniform(0.4, 0.8)
                        } for _ in range(2)
                    ]
                }
            elif 'Amount' in agent:
                decision['decision_data'] = {
                    'primary_amount': round(random.uniform(5, 50), 2),
                    'alternatives': [round(random.uniform(1, 30), 2) for _ in range(3)],
                    'optimization_score': random.uniform(0.7, 0.95)
                }
            
            decisions.append(decision)
        
        return decisions
    
    def generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate performance metrics"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': {
                'total_users': random.randint(8000, 12000),
                'active_users_today': random.randint(300, 800),
                'total_donations': random.randint(15000, 25000),
                'total_amount_raised': round(random.uniform(150000, 300000), 2),
                'average_donation': round(random.uniform(12, 25), 2),
                'conversion_rate': round(random.uniform(0.20, 0.35), 3)
            },
            'agent_metrics': {
                'DonationLikelyScoreAgent': {
                    'decisions': random.randint(800, 1200),
                    'conversions': random.randint(200, 350),
                    'conversion_rate': round(random.uniform(0.20, 0.30), 3),
                    'avg_response_time': random.randint(50, 150)
                },
                'MLDonationLikelyScoreAgent': {
                    'decisions': random.randint(900, 1300),
                    'conversions': random.randint(250, 400),
                    'conversion_rate': round(random.uniform(0.25, 0.35), 3),
                    'avg_response_time': random.randint(80, 200)
                },
                'LocalCauseRecommenderAgent': {
                    'recommendations': random.randint(1000, 1500),
                    'accepted': random.randint(400, 700),
                    'acceptance_rate': round(random.uniform(0.35, 0.50), 3),
                    'avg_relevance_score': round(random.uniform(0.75, 0.90), 3)
                }
            },
            'cause_metrics': {
                cause: {
                    'donations': random.randint(100, 500),
                    'amount_raised': round(random.uniform(5000, 25000), 2),
                    'avg_donation': round(random.uniform(10, 30), 2),
                    'success_rate': round(random.uniform(0.20, 0.40), 3)
                } for cause in self.causes
            }
        }
    
    def save_data(self, output_dir: str = 'data'):
        """Generate and save all demo data"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔄 Generating demo data...")
        
        # Generate users
        print("👥 Generating users...")
        users = self.generate_users(1000)
        with open(f'{output_dir}/demo_users.json', 'w') as f:
            json.dump(users, f, indent=2)
        
        # Generate transactions
        print("💳 Generating transactions...")
        transactions = self.generate_transactions(10000)
        with open(f'{output_dir}/demo_transactions.json', 'w') as f:
            json.dump(transactions, f, indent=2)
        
        # Generate agent decisions
        print("🤖 Generating agent decisions...")
        decisions = self.generate_agent_decisions(5000)
        with open(f'{output_dir}/demo_decisions.json', 'w') as f:
            json.dump(decisions, f, indent=2)
        
        # Generate performance metrics
        print("📊 Generating performance metrics...")
        metrics = self.generate_performance_metrics()
        with open(f'{output_dir}/demo_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("✅ Demo data generation complete!")
        print(f"📁 Data saved to {output_dir}/ directory")
        
        return {
            'users': len(users),
            'transactions': len(transactions),
            'decisions': len(decisions),
            'output_dir': output_dir
        }


def main():
    """Main function to generate demo data"""
    
    generator = SimplifiedDataGenerator()
    result = generator.save_data()
    
    print(f"\n📈 Summary:")
    print(f"  Users: {result['users']:,}")
    print(f"  Transactions: {result['transactions']:,}")
    print(f"  Agent Decisions: {result['decisions']:,}")
    print(f"  Output Directory: {result['output_dir']}")
    
    print(f"\n🚀 Ready to run dashboard:")
    print(f"  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
