import sqlite3

class HistoryDatabase:
    def __init__(self, db_path, table_name="history_data"):
        self.db_path = db_path
        self.table_name = table_name
        self._create_table()

    def _create_connection(self):
        """Private method to create and return a database connection."""
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        """Create the table if it does not exist."""
        with self._create_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    track_id INTEGER, 
                    class_name VARCHAR,
                    cam_id INTEGER, 
                    date VARCHAR, 
                    time VARCHAR,
                    x_coordinate INTEGER, 
                    y_coordinate INTEGER, 
                    z_coordinate INTEGER,
                    active VARCHAR,
                    velocity FLOAT
                )
            """)
            conn.commit()

    def add_many(self, data_list):
        """Insert multiple records into the history_data table."""
        with self._create_connection() as conn:
            cur = conn.cursor()
            cur.executemany(f"""
                INSERT INTO {self.table_name} 
                (track_id, class_name, cam_id, date, time, x_coordinate, y_coordinate, z_coordinate, active, velocity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_list)
            conn.commit()

    # def delete_data(self):
    #     """Delete all records from the history_data table."""
    #     conn = self._create_connection()
    #     cur = conn.cursor()
    #     cur.execute("DELETE FROM history_data")
    #     conn.commit()
    #     conn.close()

# Test database
if __name__ == "__main__":
    db = HistoryDatabase("history.db", "test_table")
    data = [
        (1, "excavator", 1, "2021-09-01", "12:00:00", 100, 200, 300, "active", 10.0),
        (2, "excavator", 1, "2021-09-01", "12:00:00", 100, 200, 300, "active", 10.0),
        (3, "excavator", 1, "2021-09-01", "12:00:00", 100, 200, 300, "active", 10.0),
    ]

    db.add_many(data)
