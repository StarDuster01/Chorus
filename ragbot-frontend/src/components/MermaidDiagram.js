import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { Modal, Button, ButtonGroup } from 'react-bootstrap';
import { FaSearchPlus, FaSearchMinus, FaRedo, FaDownload } from 'react-icons/fa';
import './MermaidDiagram.css';

const MermaidDiagram = ({ chart }) => {
  const mermaidRef = useRef(null);
  const fullscreenRef = useRef(null);
  const [svgContent, setSvgContent] = useState('');
  const [showFullscreen, setShowFullscreen] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [renderError, setRenderError] = useState(null);

  useEffect(() => {
    // Initialize mermaid with robust config
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: {
        useMaxWidth: false,
        htmlLabels: true,
        curve: 'basis' // Use curved lines for better aesthetics
      },
      logLevel: 1, // Keep log level minimal for production
    });

    const renderDiagram = async () => {
      if (!chart || !mermaidRef.current) return;
      
      // Clear previous renders and errors
      mermaidRef.current.innerHTML = '';
      setRenderError(null);
      
      try {
        console.log("Rendering Mermaid chart...");
        // Create unique ID to avoid conflicts with multiple charts
        const id = `mermaid-diagram-${Date.now()}`;
        
        // Parse the diagram to ensure it's valid
        const { svg } = await mermaid.render(id, chart);
        
        // Store SVG content for the modal
        setSvgContent(svg);
        
        // Set the rendered SVG
        mermaidRef.current.innerHTML = svg;
        
        // Add click handler to the diagram container
        mermaidRef.current.addEventListener('click', () => {
          setShowFullscreen(true);
          setZoomLevel(1); // Reset zoom level when opening fullscreen
        });
        
        // Add cursor style to indicate it's clickable
        const svgElement = mermaidRef.current.querySelector('svg');
        if (svgElement) {
          svgElement.style.cursor = 'pointer';
          svgElement.setAttribute('title', 'Click to expand');
          // Make sure SVG fills the container but maintains aspect ratio
          svgElement.style.maxWidth = '100%';
          svgElement.style.height = 'auto';
        }
      } catch (err) {
        console.error('Mermaid rendering error:', err);
        setRenderError(err.message || 'Unknown rendering error');
        
        // Display a useful error message
        mermaidRef.current.innerHTML = `
          <div class="alert alert-danger">
            <strong>Error rendering diagram:</strong> ${err.message || 'Unknown error'}
            <hr/>
            <pre style="max-height: 200px; overflow: auto; font-size: 12px;">${chart}</pre>
          </div>
        `;
      }
    };

    renderDiagram();
  }, [chart]);
  
  // When the fullscreen modal is shown, render the diagram again in the modal
  useEffect(() => {
    if (showFullscreen && fullscreenRef.current) {
      if (svgContent) {
        fullscreenRef.current.innerHTML = svgContent;
        // Apply zoom level
        applyZoom();
        
        // Make sure SVG is properly sized in fullscreen mode
        const svgElement = fullscreenRef.current.querySelector('svg');
        if (svgElement) {
          svgElement.style.maxWidth = '100%';
          svgElement.style.height = 'auto';
          // Center the SVG in its container
          svgElement.style.margin = '0 auto';
          svgElement.style.display = 'block';
        }
      } else if (renderError) {
        fullscreenRef.current.innerHTML = `
          <div class="alert alert-danger">
            <strong>Error rendering diagram:</strong> ${renderError}
            <hr/>
            <pre style="max-height: 400px; overflow: auto; font-size: 12px;">${chart}</pre>
          </div>
        `;
      }
    }
  }, [showFullscreen, svgContent, renderError, chart, zoomLevel]);
  
  const zoomIn = () => {
    setZoomLevel(prevZoom => Math.min(prevZoom + 0.2, 3));
  };
  
  const zoomOut = () => {
    setZoomLevel(prevZoom => Math.max(prevZoom - 0.2, 0.5));
  };
  
  const resetZoom = () => {
    setZoomLevel(1);
  };
  
  const applyZoom = () => {
    if (fullscreenRef.current) {
      const svgElement = fullscreenRef.current.querySelector('svg');
      if (svgElement) {
        svgElement.style.transform = `scale(${zoomLevel})`;
        svgElement.style.transformOrigin = 'top center';
      }
    }
  };
  
  const downloadSVG = () => {
    if (!svgContent) return;
    
    // Create a blob from the SVG content
    const blob = new Blob([svgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    
    // Create a link and trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = 'flowchart.svg';
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const retryRender = () => {
    if (chart) {
      // Force re-render by clearing and then setting the chart
      mermaidRef.current.innerHTML = 
        '<div class="text-center p-3"><small>Retrying diagram render...</small></div>';
      
      // Re-initialize mermaid with different settings that might help
      mermaid.initialize({
        startOnLoad: false,
        theme: 'default',
        securityLevel: 'loose',
        flowchart: {
          useMaxWidth: false,
          htmlLabels: true,
          curve: 'basis'
        },
        logLevel: 1
      });
      
      setTimeout(() => {
        try {
          const id = `mermaid-retry-${Date.now()}`;
          mermaid.render(id, chart).then(({ svg }) => {
            mermaidRef.current.innerHTML = svg;
            setSvgContent(svg);
            setRenderError(null);
            
            // Ensure SVG is properly sized
            const svgElement = mermaidRef.current.querySelector('svg');
            if (svgElement) {
              svgElement.style.maxWidth = '100%';
              svgElement.style.height = 'auto';
            }
          }).catch(err => {
            console.error('Retry render failed:', err);
            setRenderError(err.message || 'Retry failed');
          });
        } catch (err) {
          console.error('Retry setup failed:', err);
        }
      }, 100);
    }
  };

  return (
    <>
      <div className="mermaid-wrapper">
        <div 
          ref={mermaidRef} 
          className="mermaid-container"
          title={renderError ? "Rendering error" : "Click to view fullscreen"}
        ></div>
        {renderError ? (
          <div className="error-actions text-center mt-2">
            <Button variant="outline-danger" size="sm" onClick={retryRender}>
              Retry Rendering
            </Button>
          </div>
        ) : (
          <div className="zoom-hint">
            <small>Click diagram to view fullscreen</small>
          </div>
        )}
      </div>
      
      <Modal 
        show={showFullscreen} 
        onHide={() => setShowFullscreen(false)}
        size="xl"
        dialogClassName="flowchart-modal-fullscreen"
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>Decision Process Flowchart</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div className="zoom-controls">
            <ButtonGroup>
              <Button variant="outline-secondary" onClick={zoomOut} title="Zoom Out">
                <FaSearchMinus />
              </Button>
              <Button variant="outline-secondary" onClick={resetZoom} title="Reset Zoom">
                <FaRedo />
              </Button>
              <Button variant="outline-secondary" onClick={zoomIn} title="Zoom In">
                <FaSearchPlus />
              </Button>
              <Button variant="outline-secondary" onClick={downloadSVG} title="Download SVG">
                <FaDownload />
              </Button>
            </ButtonGroup>
            <span className="zoom-level ms-2">Zoom: {Math.round(zoomLevel * 100)}%</span>
          </div>
          <div className="mermaid-fullscreen">
            <div ref={fullscreenRef}></div>
          </div>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowFullscreen(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

export default MermaidDiagram; 