import unittest
from run import app, db, User, ChatMessage
from werkzeug.security import generate_password_hash, check_password_hash
from flask import url_for
from datetime import datetime, timedelta, timezone

class FlaskTestCase(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False
        
        self.app = app.test_client()
        
        # Create all tables
        with app.app_context():
            db.create_all()
            
            # Add a test user
            test_user = User(
                user_id="test123",
                first_name="Test",
                last_name="User",
                national_id="123456789",
                email="test@example.com",
                password=generate_password_hash("password")
            )
            db.session.add(test_user)
            db.session.commit()
    
    def tearDown(self):
        """Clean up after each test"""
        with app.app_context():
            db.session.remove()
            db.drop_all()
    
    # Authentication Tests
    def test_login_success(self):
        """Test successful login"""
        response = self.app.post('/login', data=dict(
            email="test@example.com",
            password="password"
        ), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome', response.data)
    
    def test_login_failure(self):
        """Test failed login"""
        response = self.app.post('/login', data=dict(
            email="test@example.com",
            password="wrongpassword"
        ), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid credentials', response.data)
    
    def test_register(self):
        """Test user registration"""
        response = self.app.post('/register', data=dict(
            email="new@example.com",
            password="newpassword",
            first_name="New",
            last_name="User",
            national_id="987654321"
        ), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Registration successful', response.data)
        
        # Verify user was added to database
        with app.app_context():
            user = User.query.filter_by(email="new@example.com").first()
            self.assertIsNotNone(user)
            self.assertEqual(user.first_name, "New")
            self.assertEqual(user.last_name, "User")

    # Protected Route Tests
    def test_protected_route(self):
        """Test that protected routes require login"""
        response = self.app.get('/', follow_redirects=True)
        self.assertIn(b'login', response.data)
    
    # Chat Functionality Tests
    def test_chat_message(self):
        """Test sending and retrieving chat messages"""
        # First login
        self.app.post('/login', data=dict(
            email="test@example.com",
            password="password"
        ), follow_redirects=True)
        
        # Send message
        response = self.app.post('/chat/send', data=dict(
            message="Hello, world!"
        ))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'success')
        
        # Get messages
        response = self.app.get('/chat/messages')
        self.assertEqual(response.status_code, 200)
        messages = response.json
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['message'], "Hello, world!")
        self.assertEqual(messages[0]['name'], "Test User")
    
    # API Endpoint Tests
    def test_detections_endpoint(self):
        """Test the detections API endpoint"""
        # First login
        self.app.post('/login', data=dict(
            email="test@example.com",
            password="password"
        ), follow_redirects=True)
        
        response = self.app.get('/detections')
        self.assertEqual(response.status_code, 200)
        self.assertIn('objects', response.json)
        self.assertIn('barcodes', response.json)
        self.assertIn('fps', response.json)
    
    def test_camera_toggle(self):
        """Test camera toggle endpoint"""
        # First login
        self.app.post('/login', data=dict(
            email="test@example.com",
            password="password"
        ), follow_redirects=True)
        
        # Get initial status
        response = self.app.get('/camera/status')
        initial_status = response.json['active']
        
        # Toggle status
        response = self.app.post('/camera/toggle')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['active'], not initial_status)
        
        # Toggle back
        response = self.app.post('/camera/toggle')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['active'], initial_status)
    
    # Database Model Tests
    def test_user_model(self):
        """Test User model"""
        with app.app_context():
            user = User.query.filter_by(email="test@example.com").first()
            self.assertIsNotNone(user)
            self.assertEqual(user.first_name, "Test")
            self.assertEqual(user.last_name, "User")
            self.assertTrue(check_password_hash(user.password, "password"))
    
    def test_chat_message_model(self):
        """Test ChatMessage model"""
        with app.app_context():
            # Create a message
            message = ChatMessage(
                user_id="test123",
                message="Test message",
                first_name="Test",
                last_name="User",
                timestamp=datetime.now(timezone.utc)
            )
            db.session.add(message)
            db.session.commit()
            
            # Retrieve message
            saved_message = ChatMessage.query.first()
            self.assertIsNotNone(saved_message)
            self.assertEqual(saved_message.message, "Test message")
            self.assertEqual(saved_message.first_name, "Test")
            self.assertEqual(saved_message.last_name, "User")

    # Active Users Test
    def test_active_users(self):
        """Test active users endpoint"""
        # First login
        self.app.post('/login', data=dict(
            email="test@example.com",
            password="password"
        ), follow_redirects=True)
        
        response = self.app.get('/active_users')
        self.assertEqual(response.status_code, 200)
        users = response.json
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]['name'], "Test User")

    # Chat Message Time Filter Test
    def test_chat_message_time_filter(self):
        """Test chat messages are filtered by time"""
        with app.app_context():
            # Create old message (more than 1 hour ago)
            old_message = ChatMessage(
                user_id="test123",
                message="Old message",
                first_name="Test",
                last_name="User",
                timestamp=datetime.now(timezone.utc) - timedelta(hours=2)
            )
            db.session.add(old_message)
            
            # Create new message
            new_message = ChatMessage(
                user_id="test123",
                message="New message",
                first_name="Test",
                last_name="User",
                timestamp=datetime.now(timezone.utc)
            )
            db.session.add(new_message)
            db.session.commit()
        
        # First login
        self.app.post('/login', data=dict(
            email="test@example.com",
            password="password"
        ), follow_redirects=True)
        
        # Get messages - should only return the new one
        response = self.app.get('/chat/messages')
        self.assertEqual(response.status_code, 200)
        messages = response.json
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['message'], "New message")

if __name__ == '__main__':
    unittest.main()