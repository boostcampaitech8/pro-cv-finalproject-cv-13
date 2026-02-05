import React, { useState, useEffect, useCallback } from 'react';

const API_BASE_URL = '/api';

interface Study {
  studyInstanceUID: string;
  patientName: string;
  studyDate: string;
  studyDescription: string;
  modalities: string;
}

/**
 * Worklist Layout with Upload Panel on left and Study List in center
 */
function WorklistLayout({
  leftPanels = [],
  servicesManager,
  extensionManager,
  commandsManager,
}: {
  leftPanels?: string[];
  servicesManager: any;
  extensionManager: any;
  commandsManager: any;
}): React.ReactElement {
  const [studies, setStudies] = useState<Study[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch studies from Orthanc
  const fetchStudies = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/dicom-web/studies');
      if (!response.ok) throw new Error('Failed to fetch studies');

      const data = await response.json();
      const formattedStudies: Study[] = data.map((study: any) => ({
        studyInstanceUID: study['0020000D']?.Value?.[0] || '',
        patientName: study['00100010']?.Value?.[0]?.Alphabetic || 'Unknown',
        studyDate: study['00080020']?.Value?.[0] || '',
        studyDescription: study['00081030']?.Value?.[0] || 'No description',
        modalities: study['00080061']?.Value?.join(', ') || 'Unknown',
      }));
      setStudies(formattedStudies);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStudies();
    // Refresh every 10 seconds
    const interval = setInterval(fetchStudies, 10000);
    return () => clearInterval(interval);
  }, [fetchStudies]);

  const handleViewStudy = (studyUID: string) => {
    window.location.href = `/nerve-assessment?StudyInstanceUIDs=${studyUID}`;
  };

  const handleDeleteStudy = async (studyUID: string) => {
    if (!confirm('Delete this study?')) return;
    try {
      await fetch(`${API_BASE_URL}/orthanc/studies/${studyUID}`, { method: 'DELETE' });
      fetchStudies();
    } catch (err) {
      console.error('Failed to delete study:', err);
    }
  };

  const formatDate = (dateStr: string) => {
    if (!dateStr || dateStr.length !== 8) return dateStr;
    return `${dateStr.substring(0, 4)}-${dateStr.substring(4, 6)}-${dateStr.substring(6, 8)}`;
  };

  // Get the panel component
  const getPanelComponent = (panelId: string) => {
    try {
      const panelModule = extensionManager.getModuleEntry(panelId);
      if (panelModule && panelModule.component) {
        const PanelComponent = panelModule.component;
        return (
          <PanelComponent
            commandsManager={commandsManager}
            servicesManager={servicesManager}
            extensionManager={extensionManager}
          />
        );
      }
    } catch (e) {
      console.warn('Failed to load panel:', panelId, e);
    }
    return null;
  };

  const containerStyle: React.CSSProperties = {
    display: 'flex',
    height: '100vh',
    backgroundColor: '#0a0a0a',
    color: '#fff',
  };

  const leftPanelStyle: React.CSSProperties = {
    width: '350px',
    borderRight: '1px solid #333',
    overflowY: 'auto',
    backgroundColor: '#111',
  };

  const mainContentStyle: React.CSSProperties = {
    flex: 1,
    padding: '24px',
    overflowY: 'auto',
  };

  const headerStyle: React.CSSProperties = {
    marginBottom: '24px',
  };

  const titleStyle: React.CSSProperties = {
    fontSize: '2rem',
    fontWeight: 'bold',
    marginBottom: '8px',
    background: 'linear-gradient(90deg, #60a5fa 0%, #a78bfa 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  };

  const subtitleStyle: React.CSSProperties = {
    color: '#888',
    fontSize: '1rem',
  };

  const tableStyle: React.CSSProperties = {
    width: '100%',
    borderCollapse: 'collapse',
  };

  const thStyle: React.CSSProperties = {
    textAlign: 'left',
    padding: '12px 16px',
    borderBottom: '2px solid #333',
    color: '#888',
    fontSize: '0.85rem',
    textTransform: 'uppercase',
  };

  const tdStyle: React.CSSProperties = {
    padding: '16px',
    borderBottom: '1px solid #222',
  };

  const btnStyle: React.CSSProperties = {
    padding: '8px 16px',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: '500',
    marginRight: '8px',
  };

  const viewBtnStyle: React.CSSProperties = {
    ...btnStyle,
    backgroundColor: '#3b82f6',
    color: '#fff',
  };

  const deleteBtnStyle: React.CSSProperties = {
    ...btnStyle,
    backgroundColor: '#ef4444',
    color: '#fff',
  };

  return (
    <div style={containerStyle}>
      {/* Left Panel - Upload */}
      <div style={leftPanelStyle}>
        {leftPanels.map((panelId, index) => (
          <div key={index}>{getPanelComponent(panelId)}</div>
        ))}
      </div>

      {/* Main Content - Study List */}
      <div style={mainContentStyle}>
        <div style={headerStyle}>
          <h1 style={titleStyle}>Nerve Risk Assessment</h1>
          <p style={subtitleStyle}>CT Analysis & Visualization Platform</p>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ fontSize: '1.2rem', color: '#ccc' }}>Available Studies</h2>
          <button
            onClick={fetchStudies}
            style={{ ...btnStyle, backgroundColor: '#374151', color: '#fff' }}
          >
            Refresh
          </button>
        </div>

        {loading && <div style={{ color: '#888', padding: '20px' }}>Loading studies...</div>}

        {error && (
          <div style={{ color: '#ef4444', padding: '20px', backgroundColor: 'rgba(239,68,68,0.1)', borderRadius: '8px' }}>
            Error: {error}
          </div>
        )}

        {!loading && !error && studies.length === 0 && (
          <div style={{
            textAlign: 'center',
            padding: '60px 20px',
            backgroundColor: '#111',
            borderRadius: '12px',
            border: '1px dashed #333',
          }}>
            <div style={{ fontSize: '3rem', marginBottom: '16px' }}>📁</div>
            <div style={{ color: '#888', fontSize: '1.1rem' }}>No studies available</div>
            <div style={{ color: '#666', marginTop: '8px' }}>Use the upload panel on the left to analyze CT data</div>
          </div>
        )}

        {!loading && !error && studies.length > 0 && (
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Patient Name</th>
                <th style={thStyle}>Study Date</th>
                <th style={thStyle}>Description</th>
                <th style={thStyle}>Modalities</th>
                <th style={thStyle}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {studies.map((study) => (
                <tr key={study.studyInstanceUID} style={{ cursor: 'pointer' }}>
                  <td style={tdStyle}>
                    <span style={{ fontWeight: '500' }}>{study.patientName}</span>
                  </td>
                  <td style={tdStyle}>{formatDate(study.studyDate)}</td>
                  <td style={tdStyle}>{study.studyDescription}</td>
                  <td style={tdStyle}>
                    <span style={{
                      padding: '4px 8px',
                      backgroundColor: '#1e3a5f',
                      borderRadius: '4px',
                      fontSize: '0.85rem',
                    }}>
                      {study.modalities}
                    </span>
                  </td>
                  <td style={tdStyle}>
                    <button
                      style={viewBtnStyle}
                      onClick={() => handleViewStudy(study.studyInstanceUID)}
                    >
                      View
                    </button>
                    <button
                      style={deleteBtnStyle}
                      onClick={() => handleDeleteStudy(study.studyInstanceUID)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

/**
 * Layout Template Module for Nerve Assessment Extension
 */
function getLayoutTemplateModule({
  servicesManager,
  extensionManager,
  commandsManager,
}: {
  servicesManager: any;
  extensionManager: any;
  commandsManager: any;
}) {
  return [
    {
      name: 'worklistLayout',
      id: 'worklistLayout',
      component: (props: any) => (
        <WorklistLayout
          {...props}
          servicesManager={servicesManager}
          extensionManager={extensionManager}
          commandsManager={commandsManager}
        />
      ),
    },
  ];
}

export default getLayoutTemplateModule;
