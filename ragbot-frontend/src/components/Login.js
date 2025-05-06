import React, { useState } from 'react';
import { Card, Form, Button, Alert, Container } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import axios from 'axios';
import logo from '../logo.svg';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const Login = ({ onLogin }) => {
  const [credentials, setCredentials] = useState({
    username: '',
    password: ''
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

    try {
      const response = await axios.post(`${API_URL}/auth/login`, credentials);
      const { token, user } = response.data;
      onLogin(user, token);
    } catch (err) {
      setError(
        err.response?.data?.error || 
        'Failed to log in. Please check your credentials.'
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
          <h2 className="auth-title">Login to RAGBot</h2>
          
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
            
            <Form.Group className="mb-4">
              <Form.Label>Password</Form.Label>
              <Form.Control
                type="password"
                name="password"
                value={credentials.password}
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
              {loading ? 'Logging in...' : 'Login'}
            </Button>
          </Form>
          
          <div className="text-center mt-4">
            <p>Don't have an account? <Link to="/register">Register</Link></p>
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

export default Login; 