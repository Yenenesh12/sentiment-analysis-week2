import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """PostgreSQL database connection manager with error handling"""
    
    def __init__(self):
        load_dotenv()
        self.config = {
            'dbname': os.getenv('DB_NAME', 'bank_reviews'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        self.connection = None
        
    def connect(self) -> Optional[psycopg2.extensions.connection]:
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                **self.config,
                cursor_factory=RealDictCursor
            )
            logger.info("Database connection established successfully")
            return self.connection
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            return None
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> Any:
        """Execute SQL query with error handling"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or ())
                if fetch:
                    if query.strip().upper().startswith('SELECT'):
                        return cursor.fetchall()
                    self.connection.commit()
                    return cursor.rowcount
                self.connection.commit()
        except psycopg2.Error as e:
            logger.error(f"Query execution failed: {e}")
            self.connection.rollback()
            raise
    
    def check_connection(self) -> bool:
        """Verify database connection"""
        try:
            self.connect()
            if self.connection and not self.connection.closed:
                return True
            return False
        except:
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def create_tables(db: DatabaseConnection) -> None:
    """Create database tables if they don't exist"""
    schema_file = os.path.join(os.path.dirname(__file__), 'schema.sql')
    
    with open(schema_file, 'r') as f:
        schema_sql = f.read()
    
    try:
        # Split SQL commands
        commands = schema_sql.split(';')
        for command in commands:
            if command.strip():
                db.execute_query(command, fetch=False)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise

def insert_banks_data(db: DatabaseConnection) -> None:
    """Insert bank data into banks table"""
    banks = [
        ('Commercial Bank of Ethiopia', 'CBE Birr'),
        ('Bank of Abyssinia', 'BOA Mobile'),
        ('Dashen Bank', 'Dashen Mobile')
    ]
    
    try:
        # Clear existing data
        db.execute_query("DELETE FROM banks", fetch=False)
        
        # Insert new data
        insert_query = """
        INSERT INTO banks (bank_name, app_name)
        VALUES (%s, %s)
        RETURNING bank_id
        """
        
        for bank in banks:
            result = db.execute_query(insert_query, bank)
            logger.info(f"Inserted bank: {bank[0]} with ID: {result[0]['bank_id']}")
            
    except Exception as e:
        logger.error(f"Failed to insert bank data: {e}")
        raise