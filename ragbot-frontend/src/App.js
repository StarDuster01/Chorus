import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import { Container, Navbar, Nav, Button } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import logo from './logo.svg';

// Components
import BotPanel from './components/BotPanel';
import DatasetPanel from './components/DatasetPanel';
import ChatInterface from './components/ChatInterface';
import Login from './components/Login';
import Register from './components/Register';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);

  useEffect(() => {
    // Check if token exists in localStorage
    const token = localStorage.getItem('token');
    if (token) {
      setIsAuthenticated(true);
      try {
        const userData = JSON.parse(localStorage.getItem('user'));
        setUser(userData);
      } catch (e) {
        console.error('Error parsing user data:', e);
      }
    }
  }, []);

  const handleLogin = (userData, token) => {
    localStorage.setItem('token', token);
    localStorage.setItem('user', JSON.stringify(userData));
    setIsAuthenticated(true);
    setUser(userData);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setIsAuthenticated(false);
    setUser(null);
  };

  return (
    <Router>
      <div className="App">
        <Navbar bg="custom-primary" variant="dark" expand="lg">
          <Container>
            <Navbar.Brand as={Link} to="/">
              <img src={logo} alt="RAGBot Logo" />
              RAGBot
            </Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
              {isAuthenticated ? (
                <>
                  <Nav className="me-auto">
                    <Nav.Link as={Link} to="/bots">Bots</Nav.Link>
                    <Nav.Link as={Link} to="/datasets">Datasets</Nav.Link>
                  </Nav>
                  <Nav>
                    <Navbar.Text className="me-3 text-white">
                      Signed in as: <strong>{user?.username}</strong>
                    </Navbar.Text>
                    <Button variant="outline-light" onClick={handleLogout}>
                      Logout
                    </Button>
                  </Nav>
                </>
              ) : (
                <Nav className="ms-auto">
                  <Nav.Link as={Link} to="/login">Login</Nav.Link>
                  <Nav.Link as={Link} to="/register">Register</Nav.Link>
                </Nav>
              )}
            </Navbar.Collapse>
          </Container>
        </Navbar>

        <Container>
          <Routes>
            <Route 
              path="/" 
              element={isAuthenticated ? <Navigate to="/bots" /> : <Navigate to="/login" />} 
            />
            <Route 
              path="/login" 
              element={isAuthenticated ? <Navigate to="/bots" /> : <Login onLogin={handleLogin} />} 
            />
            <Route 
              path="/register" 
              element={isAuthenticated ? <Navigate to="/bots" /> : <Register onLogin={handleLogin} />} 
            />
            
            {/* Protected Routes */}
            <Route 
              path="/bots" 
              element={isAuthenticated ? <BotPanel /> : <Navigate to="/login" />} 
            />
            <Route 
              path="/datasets" 
              element={isAuthenticated ? <DatasetPanel /> : <Navigate to="/login" />} 
            />
            <Route 
              path="/chat/:botId" 
              element={isAuthenticated ? <ChatInterface /> : <Navigate to="/login" />} 
            />
          </Routes>
        </Container>
      </div>
    </Router>
  );
}

export default App; 