import React from 'react';
import './LoadingAnimation.css';

const LoadingAnimation = () => {
  return (
    <div className="loading-container">
      <div className="geometric-loader">
        <div className="hexagon hex1"></div>
        <div className="hexagon hex2"></div>
        <div className="hexagon hex3"></div>
        <div className="circle circle1"></div>
        <div className="circle circle2"></div>
        <div className="triangle"></div>
      </div>
    </div>
  );
};

export default LoadingAnimation; 