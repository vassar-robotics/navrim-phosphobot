import sqlite3
import json
from typing import Dict, List, Optional, Any
import os
import threading
import atexit
from functools import wraps
from loguru import logger


def with_instance_lock(func):
    """Decorator to acquire instance-level lock for thread-safe database operations."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._inst_lock:
            return func(self, *args, **kwargs)

    return wrapper


class DatabaseManager:
    """SQLite database manager for training, AI control sessions, and servers."""

    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def get_instance():
        if DatabaseManager._instance is None:
            with DatabaseManager._lock:
                # Double-check locking pattern
                if DatabaseManager._instance is None:
                    DatabaseManager._instance = DatabaseManager()
        return DatabaseManager._instance

    def __init__(self, db_path: str = os.path.expanduser("~/navrim/userdata.db")):
        """Initialize database connection and create tables if they don't exist.

        Args:
            db_path: Path to the SQLite database file
        """
        if DatabaseManager._instance is not None:
            logger.warning("DatabaseManager instance already exists. Use get_instance() method instead.")

        self.db_path = db_path
        self.conn = None
        self._inst_lock = threading.Lock()
        self._connect()
        self._create_tables()

        # Register cleanup function to close connection on program exit
        atexit.register(self._cleanup)

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager.

        Note: The database connection is NOT closed here since this is a singleton.
        The connection will be closed automatically when the program exits.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)

        Returns:
            False to propagate any exceptions
        """
        return False

    def _connect(self):
        """Create a connection to the SQLite database."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _create_tables(self):
        """Create all required tables if they don't exist."""
        try:
            cursor = self.conn.cursor()

            # Create training table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training (
                    id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    terminated_at TEXT,
                    used_wandb BOOLEAN NOT NULL DEFAULT 0,
                    model_type TEXT NOT NULL,
                    training_params TEXT NOT NULL,  -- JSON string
                    modal_function_call_id TEXT
                )
            """)

            # Create ai_control_sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_control_sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    setup_success BOOLEAN NOT NULL DEFAULT 0,
                    user_email TEXT,
                    status TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    server_id INTEGER,
                    feedback TEXT,
                    FOREIGN KEY (server_id) REFERENCES servers(id)
                )
            """)

            # Create servers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS servers (
                    id INTEGER PRIMARY KEY,
                    url TEXT NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    region TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timeout INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    tcp_port INTEGER,
                    model_type TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    terminated_at TEXT
                )
            """)

            # Create indices for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_user_id ON training(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_status ON training(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON ai_control_sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_server_id ON ai_control_sessions(server_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_servers_user_id ON servers(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_servers_status ON servers(status)")

            self.conn.commit()
            logger.info("Database tables created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise

    @with_instance_lock
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def _cleanup(self):
        """Cleanup function called on program exit."""
        try:
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed on program exit")
        except Exception as e:
            logger.error(f"Error closing database connection on exit: {e}")

    # Training table operations

    @with_instance_lock
    def insert_training(self, training_data: Dict[str, Any]) -> int:
        """Insert a new training record.

        Args:
            training_data: Dictionary containing training data

        Returns:
            The ID of the inserted record
        """
        try:
            cursor = self.conn.cursor()

            # Convert training_params dict to JSON string
            training_params_json = json.dumps(training_data.get("training_params", {}))

            cursor.execute(
                """
                INSERT INTO training (
                    id, status, user_id, dataset_name, model_name,
                    requested_at, terminated_at, used_wandb, model_type,
                    training_params, modal_function_call_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    training_data.get("id"),
                    training_data.get("status"),
                    training_data.get("user_id"),
                    training_data.get("dataset_name"),
                    training_data.get("model_name"),
                    training_data.get("requested_at"),
                    training_data.get("terminated_at"),
                    training_data.get("used_wandb", False),
                    training_data.get("model_type"),
                    training_params_json,
                    training_data.get("modal_function_call_id"),
                ),
            )

            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error inserting training record: {e}")
            self.conn.rollback()
            raise

    @with_instance_lock
    def get_training(self, training_id: int) -> Optional[Dict[str, Any]]:
        """Get a training record by ID.

        Args:
            training_id: The ID of the training record

        Returns:
            Dictionary containing training data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM training WHERE id = ?", (training_id,))
            row = cursor.fetchone()

            if row:
                result = dict(row)
                # Parse JSON fields
                result["training_params"] = json.loads(result["training_params"])
                return result
            return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching training record: {e}")
            raise

    @with_instance_lock
    def get_trainings_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all training records for a user.

        Args:
            user_id: The user ID

        Returns:
            List of training records
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM training WHERE user_id = ? ORDER BY requested_at DESC", (user_id,))
            rows = cursor.fetchall()

            results = []
            for row in rows:
                result = dict(row)
                result["training_params"] = json.loads(result["training_params"])
                results.append(result)
            return results
        except sqlite3.Error as e:
            logger.error(f"Error fetching training records: {e}")
            raise

    @with_instance_lock
    def update_training_status(self, training_id: int, status: str, terminated_at: Optional[str] = None) -> bool:
        """Update the status of a training record.

        Args:
            training_id: The ID of the training record
            status: New status
            terminated_at: Termination timestamp (optional)

        Returns:
            True if update was successful
        """
        try:
            cursor = self.conn.cursor()

            if terminated_at:
                cursor.execute(
                    """
                    UPDATE training
                    SET status = ?, terminated_at = ?
                    WHERE id = ?
                """,
                    (status, terminated_at, training_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE training
                    SET status = ?
                    WHERE id = ?
                """,
                    (status, training_id),
                )

            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error updating training status: {e}")
            self.conn.rollback()
            raise

    # AI Control Sessions table operations

    @with_instance_lock
    def insert_ai_control_session(self, session_data: Dict[str, Any]) -> str:
        """Insert a new AI control session record.

        Args:
            session_data: Dictionary containing session data

        Returns:
            The ID of the inserted record
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute(
                """
                INSERT INTO ai_control_sessions (
                    id, created_at, user_id, model_id, prompt,
                    started_at, ended_at, setup_success, user_email,
                    status, model_type, server_id, feedback
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_data.get("id"),
                    session_data.get("created_at"),
                    session_data.get("user_id"),
                    session_data.get("model_id"),
                    session_data.get("prompt", ""),
                    session_data.get("started_at"),
                    session_data.get("ended_at"),
                    session_data.get("setup_success", False),
                    session_data.get("user_email"),
                    session_data.get("status"),
                    session_data.get("model_type"),
                    session_data.get("server_id"),
                    session_data.get("feedback"),
                ),
            )

            self.conn.commit()
            return session_data.get("id")
        except sqlite3.Error as e:
            logger.error(f"Error inserting AI control session: {e}")
            self.conn.rollback()
            raise

    @with_instance_lock
    def get_ai_control_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get an AI control session by ID.

        Args:
            session_id: The ID of the session

        Returns:
            Dictionary containing session data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM ai_control_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching AI control session: {e}")
            raise

    @with_instance_lock
    def get_ai_control_sessions_by_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get AI control sessions for a user.

        Args:
            user_id: The user ID
            limit: Maximum number of records to return

        Returns:
            List of session records
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM ai_control_sessions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (user_id, limit),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error fetching AI control sessions: {e}")
            raise

    @with_instance_lock
    def update_ai_control_session_status(self, session_id: str, status: str, ended_at: Optional[str] = None) -> bool:
        """Update the status of an AI control session.

        Args:
            session_id: The ID of the session
            status: New status
            ended_at: End timestamp (optional)

        Returns:
            True if update was successful
        """
        try:
            cursor = self.conn.cursor()

            if ended_at:
                cursor.execute(
                    """
                    UPDATE ai_control_sessions
                    SET status = ?, ended_at = ?
                    WHERE id = ?
                """,
                    (status, ended_at, session_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE ai_control_sessions
                    SET status = ?
                    WHERE id = ?
                """,
                    (status, session_id),
                )

            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error updating AI control session status: {e}")
            self.conn.rollback()
            raise

    @with_instance_lock
    def update_ai_control_session_feedback(self, session_id: str, feedback: str) -> bool:
        """Update the feedback for an AI control session.

        Args:
            session_id: The ID of the session
            feedback: Feedback text

        Returns:
            True if update was successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE ai_control_sessions
                SET feedback = ?
                WHERE id = ?
            """,
                (feedback, session_id),
            )

            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error updating AI control session feedback: {e}")
            self.conn.rollback()
            raise

    @with_instance_lock
    def update_ai_control_session_setup(self, session_id: str, setup_success: bool, server_id: int) -> bool:
        """Update the setup success flag and server ID for an AI control session.

        Args:
            session_id: The ID of the session
            setup_success: Whether setup was successful
            server_id: The ID of the associated server

        Returns:
            True if update was successful
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                UPDATE ai_control_sessions
                SET setup_success = ?, server_id = ?
                WHERE id = ?
            """,
                (setup_success, server_id, session_id),
            )

            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error updating AI control session setup: {e}")
            self.conn.rollback()
            raise

    # Servers table operations

    @with_instance_lock
    def insert_server(self, server_data: Dict[str, Any]) -> int:
        """Insert a new server record.

        Args:
            server_data: Dictionary containing server data

        Returns:
            The ID of the inserted record
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute(
                """
                INSERT INTO servers (
                    id, url, host, port, region, status, timeout,
                    user_id, model_id, tcp_port, model_type,
                    requested_at, terminated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    server_data.get("id"),
                    server_data.get("url"),
                    server_data.get("host"),
                    server_data.get("port"),
                    server_data.get("region"),
                    server_data.get("status"),
                    server_data.get("timeout"),
                    server_data.get("user_id"),
                    server_data.get("model_id"),
                    server_data.get("tcp_port"),
                    server_data.get("model_type"),
                    server_data.get("requested_at"),
                    server_data.get("terminated_at"),
                ),
            )

            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error inserting server record: {e}")
            self.conn.rollback()
            raise

    @with_instance_lock
    def get_server(self, server_id: int) -> Optional[Dict[str, Any]]:
        """Get a server record by ID.

        Args:
            server_id: The ID of the server

        Returns:
            Dictionary containing server data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM servers WHERE id = ?", (server_id,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching server record: {e}")
            raise

    @with_instance_lock
    def get_active_servers(self) -> List[Dict[str, Any]]:
        """Get all active servers.

        Returns:
            List of active server records
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM servers
                WHERE status IN ('running', 'active')
                ORDER BY requested_at DESC
            """)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error fetching active servers: {e}")
            raise

    @with_instance_lock
    def update_server_status(self, server_id: int, status: str, terminated_at: Optional[str] = None) -> bool:
        """Update the status of a server.

        Args:
            server_id: The ID of the server
            status: New status
            terminated_at: Termination timestamp (optional)

        Returns:
            True if update was successful
        """
        try:
            cursor = self.conn.cursor()

            if terminated_at:
                cursor.execute(
                    """
                    UPDATE servers
                    SET status = ?, terminated_at = ?
                    WHERE id = ?
                """,
                    (status, terminated_at, server_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE servers
                    SET status = ?
                    WHERE id = ?
                """,
                    (status, server_id),
                )

            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Error updating server status: {e}")
            self.conn.rollback()
            raise

    @with_instance_lock
    def get_servers_by_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get servers for a user.

        Args:
            user_id: The user ID
            limit: Maximum number of records to return

        Returns:
            List of server records
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM servers
                WHERE user_id = ?
                ORDER BY requested_at DESC
                LIMIT ?
            """,
                (user_id, limit),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error fetching servers by user: {e}")
            raise

    # Combined queries

    @with_instance_lock
    def get_ai_control_session_with_server(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get an AI control session with its associated server details.

        Args:
            session_id: The ID of the session

        Returns:
            Dictionary containing session and server data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT
                    s.*,
                    srv.id as server_id,
                    srv.url as server_url,
                    srv.host as server_host,
                    srv.port as server_port,
                    srv.region as server_region,
                    srv.status as server_status,
                    srv.timeout as server_timeout,
                    srv.tcp_port as server_tcp_port,
                    srv.requested_at as server_requested_at,
                    srv.terminated_at as server_terminated_at
                FROM ai_control_sessions s
                LEFT JOIN servers srv ON s.server_id = srv.id
                WHERE s.id = ?
            """,
                (session_id,),
            )

            row = cursor.fetchone()

            if row:
                result = dict(row)
                # Organize server data into nested dict
                if result.get("server_id"):
                    result["servers"] = {
                        "id": result.pop("server_id"),
                        "url": result.pop("server_url"),
                        "host": result.pop("server_host"),
                        "port": result.pop("server_port"),
                        "region": result.pop("server_region"),
                        "status": result.pop("server_status"),
                        "timeout": result.pop("server_timeout"),
                        "tcp_port": result.pop("server_tcp_port"),
                        "requested_at": result.pop("server_requested_at"),
                        "terminated_at": result.pop("server_terminated_at"),
                        "user_id": result["user_id"],
                        "model_id": result["model_id"],
                        "model_type": result["model_type"],
                    }
                return result
            return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching AI control session with server: {e}")
            raise

    @with_instance_lock
    def get_latest_ai_control_session_by_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest AI control session for a user with server details.

        Args:
            user_id: The user ID

        Returns:
            Dictionary containing session and server data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT
                    s.*,
                    srv.status as server_status
                FROM ai_control_sessions s
                LEFT JOIN servers srv ON s.server_id = srv.id
                WHERE s.user_id = ?
                ORDER BY s.created_at DESC
                LIMIT 1
            """,
                (user_id,),
            )

            row = cursor.fetchone()

            if row:
                result = dict(row)
                # Format as expected by the API
                if result.get("server_status"):
                    result["servers"] = {"status": result.pop("server_status")}
                return result
            return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching latest AI control session: {e}")
            raise

    # Utility methods

    @with_instance_lock
    def execute_raw_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query.

        Args:
            query: SQL query string
            params: Query parameters (optional)

        Returns:
            List of result rows as dictionaries
        """
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error executing raw query: {e}")
            raise

    @with_instance_lock
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary containing counts for each table
        """
        try:
            cursor = self.conn.cursor()

            stats = {}

            # Get training stats
            cursor.execute("SELECT COUNT(*) as total, COUNT(DISTINCT user_id) as unique_users FROM training")
            row = cursor.fetchone()
            stats["training"] = dict(row)

            # Get AI control sessions stats
            cursor.execute(
                "SELECT COUNT(*) as total, COUNT(DISTINCT user_id) as unique_users FROM ai_control_sessions"
            )
            row = cursor.fetchone()
            stats["ai_control_sessions"] = dict(row)

            # Get servers stats
            cursor.execute("SELECT COUNT(*) as total, COUNT(DISTINCT user_id) as unique_users FROM servers")
            row = cursor.fetchone()
            stats["servers"] = dict(row)

            # Get active servers count
            cursor.execute("SELECT COUNT(*) as active FROM servers WHERE status IN ('running', 'active')")
            stats["servers"]["active"] = cursor.fetchone()[0]

            return stats
        except sqlite3.Error as e:
            logger.error(f"Error getting database stats: {e}")
            raise
