"""
Entry point for the CodePlagueDetector application
"""
import logging
logging.basicConfig(level=logging.DEBUG)

print("Starting CodePlagueDetector application...")
try:
    from CodePlagueDetector import create_app, db
    print("Successfully imported from CodePlagueDetector")
    
    app = create_app()
    print("App created successfully")
    
    if __name__ == "__main__":
        with app.app_context():
            print("Entering app context")
            # Import models here to avoid circular imports
            from CodePlagueDetector.models import Submission, PlagiarismReference
            print("Models imported successfully")
            
            # Drop and recreate tables to handle schema changes
            db.drop_all()
            print("Tables dropped")
            db.create_all()
            print("Tables created")
        
        print("Starting Flask app")
        app.run(debug=True)
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
