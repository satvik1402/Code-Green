from datetime import datetime
from CodePlagueDetector import db

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    problem_id = db.Column(db.String(50), nullable=False)
    code = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(20), nullable=False, default='python')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Plagiarism detection results
    exact_match = db.Column(db.Boolean, default=False)
    variable_renaming = db.Column(db.Boolean, default=False)
    structural_similarity = db.Column(db.Boolean, default=False)
    comment_similarity = db.Column(db.Boolean, default=False)
    semantic_similarity = db.Column(db.Float, default=0.0)
    
    # Feedback data stored as JSON string
    feedback_data = db.Column(db.Text)
    
    # Test case validation results
    test_passed = db.Column(db.Boolean, default=False)
    test_details = db.Column(db.Text)
    
    # Fast typing detection
    fast_typing_detected = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<Submission {self.id}>'

class PlagiarismReference(db.Model):
    """Store reference solutions for each DSA problem"""
    id = db.Column(db.Integer, primary_key=True)
    problem_id = db.Column(db.String(50), nullable=False)
    solution_code = db.Column(db.Text, nullable=False)
    solution_name = db.Column(db.String(100), nullable=False)
    language = db.Column(db.String(20), nullable=False, default='python')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PlagiarismReference {self.problem_id}>'
