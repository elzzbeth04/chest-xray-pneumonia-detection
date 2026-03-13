import React from 'react';

const DiagnosisResult = ({ result, isLoading }) => {
  if (isLoading) {
    return (
      <div className="card">
        <h2>Diagnosis Result</h2>
        <div className="result-container" style={{ minHeight: '300px', justifyContent: 'center' }}>
          <div className="loader"></div>
          <p style={{ marginTop: '1rem', color: 'var(--text-light)' }}>Analyzing X-ray using AI model...</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card">
        <h2>Diagnosis Result</h2>
        <div className="result-container" style={{ minHeight: '300px', justifyContent: 'center' }}>
          <p style={{ color: 'var(--text-light)', textAlign: 'center' }}>
            Upload an X-ray image and click "Analyze Image" to see the AI diagnosis.
          </p>
        </div>
      </div>
    );
  }

  const { prediction, confidence, heatmap, report } = result;
  
  // Format confidence as percentage
  const confidencePercent = (confidence * 100).toFixed(1);
  const isPneumonia = prediction === "PNEUMONIA";

  const handleDownloadReport = () => {
    if (!report) return;
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Medical_Report_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card">
      <h2>Diagnosis Result</h2>
      
      <div className="result-container">
        <div className={`prediction-badge ${isPneumonia ? 'pneumonia' : 'normal'}`}>
          {prediction}
        </div>
        
        <div style={{ width: '100%', marginBottom: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontWeight: 600 }}>AI Confidence</span>
            <span style={{ fontWeight: 600 }}>{confidencePercent}%</span>
          </div>
          
          <div className="confidence-bar-container">
            <div 
              className="confidence-bar" 
              style={{ 
                width: `${confidencePercent}%`,
                backgroundColor: isPneumonia ? 'var(--danger)' : 'var(--success)'
              }}
            ></div>
          </div>
        </div>

        {heatmap && (
          <div className="heatmap-container">
            <h3 style={{ fontSize: '1.1rem', marginBottom: '10px', color: 'var(--text-dark)' }}>
              Grad-CAM Activation Map
            </h3>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-light)', marginBottom: '15px' }}>
              Regions of interest identified by the AI model.
            </p>
            <img src={heatmap} alt="Grad-CAM Heatmap" className="heatmap-image" />
          </div>
        )}

        {report && (
          <div style={{ marginTop: '20px', width: '100%', textAlign: 'center' }}>
            <button 
              className="btn-primary" 
              onClick={handleDownloadReport}
              style={{ width: '100%', padding: '12px' }}
            >
              📄 Download Medical Report
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DiagnosisResult;
