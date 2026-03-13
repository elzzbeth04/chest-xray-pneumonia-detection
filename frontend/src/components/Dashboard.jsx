import React, { useState } from 'react';
import UploadXray from './UploadXray';
import DiagnosisResult from './DiagnosisResult';
import PatientHistory from './PatientHistory';
import '../styles/dashboard.css';

const Dashboard = () => {
  const [currentResult, setCurrentResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const handleUpload = async (file, previewUrl) => {
    setIsLoading(true);
    setCurrentResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Connect to the Flask backend
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      
      if (data.error) {
        console.error("Backend error:", data.error);
        alert(`Error: ${data.error}`);
        setIsLoading(false);
        return;
      }

      // Add to current result
      setCurrentResult({
        prediction: data.prediction,
        confidence: data.confidence,
        heatmap: data.heatmap
      });

      // Add to history
      const newRecord = {
        id: Date.now().toString(),
        patientId: `PT-${Math.floor(Math.random() * 90000) + 10000}`,
        date: new Date().toLocaleString(),
        prediction: data.prediction,
        confidence: data.confidence,
      };

      setHistory(prev => [newRecord, ...prev].slice(0, 10)); // Keep last 10
      
    } catch (error) {
      console.error('Error uploading image:', error);
      alert('Error connecting to the prediction API. Make sure the Flask backend is running on port 5000.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>AI Pneumonia Detection</h1>
        <p>Upload a chest X-ray image for instant AI-powered analysis with Grad-CAM explainability.</p>
      </header>
      
      <div className="dashboard-grid">
        <div className="left-column">
          <UploadXray onUpload={handleUpload} isLoading={isLoading} />
          <PatientHistory history={history} />
        </div>
        
        <div className="right-column">
          <DiagnosisResult result={currentResult} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
