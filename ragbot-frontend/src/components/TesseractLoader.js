import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

const TesseractLoader = ({ size = 60 }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const frameRef = useRef(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Camera setup - moved further back to reduce zoom
    const camera = new THREE.PerspectiveCamera(
      45,
      1,
      0.1,
      1000
    );
    camera.position.z = 8;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true 
    });
    renderer.setSize(size, size);
    renderer.setClearColor(0x000000, 0);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Tesseract vertices in 4D
    const vertices4D = [];
    for (let i = 0; i < 16; i++) {
      vertices4D.push([
        (i & 1) * 2 - 1,
        ((i >> 1) & 1) * 2 - 1,
        ((i >> 2) & 1) * 2 - 1,
        ((i >> 3) & 1) * 2 - 1
      ]);
    }

    // Tesseract edges
    const edges = [];
    for (let i = 0; i < 16; i++) {
      for (let j = i + 1; j < 16; j++) {
        let diff = 0;
        for (let k = 0; k < 4; k++) {
          if (vertices4D[i][k] !== vertices4D[j][k]) diff++;
        }
        if (diff === 1) edges.push([i, j]);
      }
    }

    // Create line geometry
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(edges.length * 6);
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Random color selection - 1 in 10 chance for purple, otherwise dark green
    const isPurple = Math.random() < 0.1; // 10% chance
    const color = isPurple ? 0x6A1B9A : 0x1B5E20; // Purple or dark green
    
    // Material with randomly selected color
    const material = new THREE.LineBasicMaterial({ 
      color: color,
      linewidth: 3,
      transparent: true,
      opacity: 0.9
    });

    // Create line segments
    const tesseract = new THREE.LineSegments(geometry, material);
    scene.add(tesseract);

    // Animation variables
    let time = 0;
    const rotationSpeed = 0.01;

    // 4D rotation matrices
    const rotate4D = (vertex, angleXY, angleZW, angleXZ, angleYW) => {
      let [x, y, z, w] = vertex;
      
      // XY rotation
      const cosXY = Math.cos(angleXY);
      const sinXY = Math.sin(angleXY);
      [x, y] = [x * cosXY - y * sinXY, x * sinXY + y * cosXY];
      
      // ZW rotation
      const cosZW = Math.cos(angleZW);
      const sinZW = Math.sin(angleZW);
      [z, w] = [z * cosZW - w * sinZW, z * sinZW + w * cosZW];
      
      // XZ rotation
      const cosXZ = Math.cos(angleXZ);
      const sinXZ = Math.sin(angleXZ);
      [x, z] = [x * cosXZ - z * sinXZ, x * sinXZ + z * cosXZ];
      
      // YW rotation
      const cosYW = Math.cos(angleYW);
      const sinYW = Math.sin(angleYW);
      [y, w] = [y * cosYW - w * sinYW, y * sinYW + w * cosYW];
      
      return [x, y, z, w];
    };

    // Project 4D to 3D - adjusted projection distance for better view
    const project4Dto3D = (vertex4D) => {
      const distance = 3;
      const w = vertex4D[3];
      const scale = distance / (distance - w);
      return [
        vertex4D[0] * scale,
        vertex4D[1] * scale,
        vertex4D[2] * scale
      ];
    };

    // Animation loop
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      time += 1;

      // Update tesseract geometry
      const positions = geometry.attributes.position.array;
      let posIndex = 0;
      
      edges.forEach(([i, j]) => {
        const vertex1 = rotate4D(vertices4D[i], time * rotationSpeed, time * rotationSpeed * 0.7, time * rotationSpeed * 0.5, time * rotationSpeed * 0.3);
        const vertex2 = rotate4D(vertices4D[j], time * rotationSpeed, time * rotationSpeed * 0.7, time * rotationSpeed * 0.5, time * rotationSpeed * 0.3);
        
        const [x1, y1, z1] = project4Dto3D(vertex1);
        const [x2, y2, z2] = project4Dto3D(vertex2);
        
        positions[posIndex] = x1;
        positions[posIndex + 1] = y1;
        positions[posIndex + 2] = z1;
        positions[posIndex + 3] = x2;
        positions[posIndex + 4] = y2;
        positions[posIndex + 5] = z2;
        
        posIndex += 6;
      });
      
      geometry.attributes.position.needsUpdate = true;

      renderer.render(scene, camera);
    };

    animate();

    // Cleanup
    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
      geometry.dispose();
      material.dispose();
    };
  }, [size]);

  return (
    <div style={{ 
      display: 'inline-flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      width: size,
      height: size 
    }}>
      <div ref={mountRef} style={{ width: size, height: size }} />
    </div>
  );
};

export default TesseractLoader; 