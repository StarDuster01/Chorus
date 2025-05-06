import React, { useState } from 'react';
import { Card, Form, Button, Alert } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import axios from 'axios';
import logo from '../logo.svg';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const Register = ({ onLogin }) => {
  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
    confirmPassword: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setCredentials({
      ...credentials,
      [name]: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Validate passwords match
    if (credentials.password !== credentials.confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/auth/register`, {
        username: credentials.username,
        password: credentials.password
      });
      
      const { token, user } = response.data;
      onLogin(user, token);
    } catch (err) {
      setError(
        err.response?.data?.error || 
        'Failed to register. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <Card className="auth-card">
        <Card.Body>
          <img src={logo} alt="RAGBot Logo" className="auth-logo" />
          <h2 className="auth-title">Create an Account</h2>
          
          {error && (
            <Alert variant="danger" onClose={() => setError(null)} dismissible>
              {error}
            </Alert>
          )}
          
          <Form onSubmit={handleSubmit}>
            <Form.Group className="mb-3">
              <Form.Label>Username</Form.Label>
              <Form.Control
                type="text"
                name="username"
                value={credentials.username}
                onChange={handleChange}
                required
                className="rounded-pill"
              />
            </Form.Group>
            
            <Form.Group className="mb-3">
              <Form.Label>Password</Form.Label>
              <Form.Control
                type="password"
                name="password"
                value={credentials.password}
                onChange={handleChange}
                required
                minLength="6"
                className="rounded-pill"
              />
              <Form.Text className="text-muted">
                Password must be at least 6 characters long.
              </Form.Text>
            </Form.Group>
            
            <Form.Group className="mb-4">
              <Form.Label>Confirm Password</Form.Label>
              <Form.Control
                type="password"
                name="confirmPassword"
                value={credentials.confirmPassword}
                onChange={handleChange}
                required
                className="rounded-pill"
              />
            </Form.Group>
            
            <Button 
              variant="primary" 
              type="submit" 
              className="w-100 rounded-pill" 
              disabled={loading}
            >
              {loading ? 'Registering...' : 'Create Account'}
            </Button>
          </Form>
          
          <div className="text-center mt-4">
            <p>Already have an account? <Link to="/login">Login</Link></p>
          </div>
          
          <div className="text-center mt-3">
            <small className="text-muted">
              Powered by Retrieval Augmented Generation
            </small>
          </div>
        </Card.Body>
      </Card>
    </div>
  );
};

export default Register; 