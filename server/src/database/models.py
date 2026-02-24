from sqlalchemy import Column, Integer, String, JSON, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .config import Base

class TripSession(Base):
    """
    Stores the full conversation history for a trip management session.
    """
    __tablename__ = "trip_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    conversation_history = Column(JSON)  # Stores the list of message dicts
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship to trip plans
    plans = relationship("TripPlan", back_populates="session")

class TripPlan(Base):
    """
    Stores the decision of destination, plan details, and images.
    """
    __tablename__ = "trip_plans"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("trip_sessions.session_id"))
    destination = Column(String, index=True)
    plan_details = Column(JSON)  # Stores itinerary, budget, etc.
    images = Column(JSON)        # Stores list of image URLs or metadata
    confirmed = Column(Boolean, default=False)  # Whether the trip is confirmed
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("TripSession", back_populates="plans")
