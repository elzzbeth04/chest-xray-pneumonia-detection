import React from 'react';

const PatientHistory = ({ history }) => {
  return (
    <div className="card" style={{ marginTop: '2rem' }}>
      <h2>Recent Analyses</h2>
      
      {history.length === 0 ? (
        <p style={{ color: 'var(--text-light)', textAlign: 'center', padding: '1rem 0' }}>
          No previous scans analyzed during this session.
        </p>
      ) : (
        <ul className="history-list">
          {history.map((record) => {
            const isPneumonia = record.prediction === "PNEUMONIA";
            
            return (
              <li key={record.id} className="history-item">
                <div className="patient-info">
                  <span className="patient-id">{record.patientId}</span>
                  <span className="record-date">{record.date}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                  <span style={{ fontSize: '0.9rem', color: 'var(--text-light)' }}>
                    Conf: {(record.confidence * 100).toFixed(1)}%
                  </span>
                  <span className={`status-indicator ${isPneumonia ? 'pneumonia' : 'normal'}`}>
                    {record.prediction}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

export default PatientHistory;
