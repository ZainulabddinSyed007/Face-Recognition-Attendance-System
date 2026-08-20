import sqlite3

conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Create students table
cursor.execute("""
CREATE TABLE IF NOT EXISTS students(
    student_id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    year TEXT,
    email TEXT
)
""")

# Create attendance table
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    date TEXT,
    status TEXT
)
""")

conn.commit()
conn.close()

print("Database and tables created successfully!")