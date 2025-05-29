"""
Database migration script to add test_passed and test_details fields to Submission model
"""
from app import db
from models import Submission

# Check if the test_passed column exists
has_test_columns = False
for column in Submission.__table__.columns:
    if column.name == 'test_passed':
        has_test_columns = True
        break

# If the columns don't exist, add them
if not has_test_columns:
    print("Adding test_passed and test_details columns to Submission table...")
    with db.engine.connect() as conn:
        conn.execute('ALTER TABLE submission ADD COLUMN test_passed BOOLEAN DEFAULT FALSE')
        conn.execute('ALTER TABLE submission ADD COLUMN test_details TEXT')
    print("Database migration completed successfully!")
else:
    print("Migration not needed - columns already exist.")

print("Database schema is now up to date!")
