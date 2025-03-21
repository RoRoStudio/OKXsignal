import psycopg2
import os
import time
from dotenv import load_dotenv
from config.config_loader import load_config
from psycopg2.extras import execute_values
from psycopg2.extensions import adapt

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", "credentials.env"))

# Load configuration settings
config = load_config()

DB_HOST = config["DB_HOST"]
DB_PORT = config["DB_PORT"]
DB_NAME = config["DB_NAME"]
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def get_connection():
    """Establishes and returns a PostgreSQL database connection."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def fetch_data(query, params=None):
    """Fetches data from PostgreSQL and returns results as a list of dictionaries."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    results = []
    try:
        cursor.execute(query, params or ())
        results = cursor.fetchall()
    except Exception as e:
        print(f"❌ Database fetch error: {e}")
    finally:
        cursor.close()
        conn.close()
    return results

def execute_query(query, params=None):
    """Executes a query (INSERT, UPDATE, DELETE) on PostgreSQL."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query, params or ())
        conn.commit()
    except Exception as e:
        print(f"❌ Database error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def execute_copy_update(temp_table_name, column_names, values, update_query):
    """
    Performs a high-speed COPY into a temp table and executes an UPDATE ... FROM ... join.
    """
    import io
    import time

    conn = get_connection()
    cursor = conn.cursor()

    try:
        start_all = time.time()

        print(f"🧪 Creating temporary table: {temp_table_name}")
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table_name}")
        create_stmt = f"""
        CREATE TEMP TABLE {temp_table_name} (
            {', '.join(f"{col} double precision" if col != 'id' else 'id bigint' for col in column_names)}
        ) ON COMMIT DROP;
        """
        cursor.execute(create_stmt)

        print("📥 Starting COPY INTO temp table...")
        output = io.StringIO()
        for row in values:
            output.write("\t".join("" if v is None else str(v) for v in row) + "\n")
        output.seek(0)

        copy_start = time.time()
        cursor.copy_from(output, temp_table_name, sep="\t", null="")
        print(f"✅ COPY completed in {time.time() - copy_start:.2f}s")

        print("🔁 Running UPDATE FROM temp table...")
        update_start = time.time()
        cursor.execute(update_query.format(temp_table=temp_table_name))
        print(f"✅ UPDATE completed in {time.time() - update_start:.2f}s")

        conn.commit()
        print(f"🎉 Total COPY + UPDATE time: {time.time() - start_all:.2f}s")

    except Exception as e:
        print(f"❌ Error during COPY+UPDATE: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()