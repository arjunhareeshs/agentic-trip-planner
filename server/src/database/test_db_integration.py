import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database.db_manager import save_or_update_session, save_or_update_plan
from database.config import SessionLocal
from database.models import TripSession, TripPlan

def test_db():
    print("--- Testing Database Integration ---")
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 1. Test session save
    history = [
        {"role": "user", "parts": [{"text": "I want to go to Paris"}]},
        {"role": "model", "parts": [{"text": "Paris is lovely! What's your budget?"}]}
    ]
    print(f"Saving session: {session_id}...")
    save_or_update_session(session_id, history)
    
    # 2. Test plan save
    print("Saving plan...")
    plan_details = {"itinerary": "Day 1: Eiffel Tower", "budget": "2000 USD"}
    images = [{"url": "http://example.com/paris.jpg", "title": "Eiffel Tower"}]
    save_or_update_plan(session_id, "Paris", plan_details, images, confirmed=True)
    
    # 3. Verify
    db = SessionLocal()
    try:
        session = db.query(TripSession).filter(TripSession.session_id == session_id).first()
        plan = db.query(TripPlan).filter(TripPlan.session_id == session_id).first()
        
        if session and plan:
            print("SUCCESS: Data successfully persisted!")
            print(f"Session History: {session.conversation_history}")
            print(f"Plan Destination: {plan.destination}")
            print(f"Plan Images: {plan.images}")
        else:
            print("FAILURE: Data not found in database.")
    finally:
        db.close()

if __name__ == "__main__":
    try:
        test_db()
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nNote: Make sure PostgreSQL is running and credentials in .env are correct.")
