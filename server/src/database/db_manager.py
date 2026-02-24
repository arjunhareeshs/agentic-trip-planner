import logging
from contextlib import contextmanager
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from .config import SessionLocal, engine, Base, DB_CONFIGURED
from .models import TripSession, TripPlan

logger = logging.getLogger(__name__)

# Track whether DB is actually reachable
db_available = False

# Create tables if they don't exist — guarded so import never crashes if
# Postgres is unreachable at startup time.
if DB_CONFIGURED and engine is not None:
    try:
        Base.metadata.create_all(bind=engine)
        db_available = True
    except OperationalError as _e:
        logger.warning(
            "Database not reachable at startup; tables were NOT created: %s", _e
        )
else:
    logger.info("DB_PASSWORD not set — database features disabled (in-memory only).")


@contextmanager
def _get_db():
    """Proper context manager that always closes the session."""
    if not db_available or SessionLocal is None:
        raise RuntimeError("Database is not available")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_or_update_session(session_id: str, conversation_history: list):
    """
    Saves or updates a conversation session in the database.
    """
    with _get_db() as db:
        try:
            session = db.query(TripSession).filter(TripSession.session_id == session_id).first()
            if session:
                session.conversation_history = conversation_history
            else:
                session = TripSession(
                    session_id=session_id,
                    conversation_history=conversation_history
                )
                db.add(session)
            db.commit()
            db.refresh(session)
            return session
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
            db.rollback()
            raise

def save_or_update_plan(session_id: str, destination: str, plan_details: dict, images: list = None, confirmed: bool = False):
    """
    Saves or updates a trip plan associated with a session.
    """
    with _get_db() as db:
        try:
            plan = db.query(TripPlan).filter(TripPlan.session_id == session_id).first()
            if plan:
                plan.destination = destination
                plan.plan_details = plan_details
                if images is not None:
                    plan.images = images
                plan.confirmed = confirmed
            else:
                plan = TripPlan(
                    session_id=session_id,
                    destination=destination,
                    plan_details=plan_details,
                    images=images or [],
                    confirmed=confirmed
                )
                db.add(plan)
            db.commit()
            db.refresh(plan)
            return plan
        except Exception as e:
            logger.error(f"Error saving plan for session {session_id}: {e}")
            db.rollback()
            raise

def get_session(session_id: str):
    """
    Retrieves a session's conversation history from the database.
    """
    with _get_db() as db:
        try:
            session = db.query(TripSession).filter(TripSession.session_id == session_id).first()
            return session.conversation_history if session else None
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None

def get_plan(session_id: str):
    """
    Retrieves a trip plan associated with a session.
    """
    with _get_db() as db:
        try:
            plan = db.query(TripPlan).filter(TripPlan.session_id == session_id).first()
            if plan:
                return {
                    "destination": plan.destination,
                    "plan_details": plan.plan_details,
                    "images": plan.images,
                    "confirmed": plan.confirmed
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving plan for session {session_id}: {e}")
            return None

