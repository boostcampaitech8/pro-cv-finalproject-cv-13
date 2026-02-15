import React, { useState, useEffect, useCallback } from 'react';

const API_BASE_URL = '/api';

interface StudyListPanelProps {
  commandsManager?: any;
  extensionManager?: any;
  servicesManager?: any;
}

interface Study {
  ID: string;
  MainDicomTags?: {
    PatientName?: string;
    StudyDescription?: string;
    StudyInstanceUID?: string;
  };
}

interface AnalysisResult {
  status: string;
  study_uid?: string;
  ohif_url?: string;
  message?: string;
  error?: string;
}

function StudyListPanel({ commandsManager, extensionManager, servicesManager }: StudyListPanelProps): React.ReactElement {
  const [studies, setStudies] = useState<Study[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Analysis state
  const [analyzingStudy, setAnalyzingStudy] = useState<Study | null>(null);
  const [segmentationDir, setSegmentationDir] = useState<string>('/data/test_data/segrap_0001');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [showAnalyzeModal, setShowAnalyzeModal] = useState<boolean>(false);

  const fetchStudies = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/orthanc/studies`);
      if (response.ok) {
        const data = await response.json();
        setStudies(data.studies || []);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStudies();
  }, [fetchStudies]);

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this study?')) return;
    try {
      const res = await fetch(`${API_BASE_URL}/orthanc/studies/${id}`, { method: 'DELETE' });
      if (!res.ok) {
        alert(`Delete failed: ${res.status}`);
        return;
      }
      setStudies(s => s.filter(x => x.ID !== id));
    } catch (err) {
      alert('Delete failed: network error');
    }
  };

  const handleOpen = (study: Study) => {
    const uid = study.MainDicomTags?.StudyInstanceUID;
    if (uid) window.location.href = `/nerve-assessment?StudyInstanceUIDs=${uid}`;
  };

  const handleAnalyzeClick = (study: Study) => {
    setAnalyzingStudy(study);
    setAnalysisResult(null);
    setShowAnalyzeModal(true);
  };

  const handleAnalyze = async () => {
    if (!analyzingStudy) return;

    const studyUid = analyzingStudy.MainDicomTags?.StudyInstanceUID;
    if (!studyUid) {
      setAnalysisResult({ status: 'error', error: 'Study UID not found' });
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze-study`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          study_instance_uid: studyUid,
          segmentation_dir: segmentationDir,
          patient_name: analyzingStudy.MainDicomTags?.PatientName || 'Anonymous',
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Analysis failed');
      }

      setAnalysisResult(data);

      // 분석 성공 시 자동으로 페이지 reload하여 SEG 오버레이 표시
      if (data.status === 'success') {
        setTimeout(() => {
          window.location.reload();
        }, 1500); // 1.5초 후 reload (성공 메시지 잠깐 보여주고)
      }
    } catch (err: any) {
      setAnalysisResult({ status: 'error', error: err.message });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const closeModal = () => {
    setShowAnalyzeModal(false);
    setAnalyzingStudy(null);
    setAnalysisResult(null);
  };

  const modalOverlayStyle: React.CSSProperties = {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  };

  const modalStyle: React.CSSProperties = {
    backgroundColor: '#1e293b',
    padding: '20px',
    borderRadius: '8px',
    width: '400px',
    maxWidth: '90%',
  };

  const inputStyle: React.CSSProperties = {
    width: '100%',
    padding: '8px',
    marginBottom: '12px',
    backgroundColor: '#2c3e50',
    border: '1px solid #4a5568',
    borderRadius: '4px',
    color: 'white',
    fontSize: '12px',
  };

  return (
    <div style={{ padding: '16px', color: 'white' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '16px', margin: 0 }}>Studies</h3>
        <button
          onClick={fetchStudies}
          style={{
            background: '#475569',
            border: 'none',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          {isLoading ? '...' : '↻'}
        </button>
      </div>
      {studies.length === 0 ? (
        <p style={{ color: '#64748b', textAlign: 'center' }}>No studies found</p>
      ) : (
        studies.map(s => (
          <div
            key={s.ID}
            style={{
              padding: '10px',
              marginBottom: '8px',
              background: '#1e293b',
              borderRadius: '4px',
            }}
          >
            <div style={{ fontWeight: 'bold' }}>
              {s.MainDicomTags?.PatientName || 'Unknown'}
            </div>
            <div style={{ fontSize: '12px', color: '#64748b' }}>
              {s.MainDicomTags?.StudyDescription || 'No description'}
            </div>
            <div style={{ marginTop: '8px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button
                onClick={() => handleOpen(s)}
                style={{
                  flex: 1,
                  minWidth: '60px',
                  padding: '6px',
                  background: '#3b82f6',
                  border: 'none',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '12px',
                }}
              >
                View
              </button>
              <button
                onClick={() => handleAnalyzeClick(s)}
                style={{
                  flex: 1,
                  minWidth: '60px',
                  padding: '6px',
                  background: '#10b981',
                  border: 'none',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '12px',
                }}
              >
                Analyze
              </button>
              <button
                onClick={() => handleDelete(s.ID)}
                style={{
                  padding: '6px 10px',
                  background: '#ef4444',
                  border: 'none',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '12px',
                }}
              >
                ✕
              </button>
            </div>
          </div>
        ))
      )}

      {/* Analyze Modal */}
      {showAnalyzeModal && (
        <div style={modalOverlayStyle} onClick={closeModal}>
          <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
            <h4 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>
              Nerve Analysis
            </h4>

            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                {analyzingStudy?.MainDicomTags?.PatientName || 'Unknown'}
              </div>
              <div style={{ fontSize: '11px', color: '#64748b', wordBreak: 'break-all' }}>
                UID: {analyzingStudy?.MainDicomTags?.StudyInstanceUID?.substring(0, 40)}...
              </div>
            </div>

            <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: '#a0aec0' }}>
              Segmentation Directory:
            </label>
            <input
              type="text"
              value={segmentationDir}
              onChange={(e) => setSegmentationDir(e.target.value)}
              style={inputStyle}
              disabled={isAnalyzing}
              placeholder="/data/test_data/segrap_0001"
            />
            <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '12px' }}>
              normal_structure/, tumor/ 하위 폴더를 포함하는 경로
            </div>

            {analysisResult && (
              <div style={{
                padding: '10px',
                marginBottom: '12px',
                backgroundColor: analysisResult.status === 'success' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                border: `1px solid ${analysisResult.status === 'success' ? '#10b981' : '#ef4444'}`,
                borderRadius: '4px',
                fontSize: '12px',
              }}>
                {analysisResult.status === 'success' ? (
                  <>
                    <div style={{ color: '#10b981', fontWeight: 'bold', marginBottom: '8px' }}>
                      Analysis Complete!
                    </div>
                    {analysisResult.ohif_url && (
                      <button
                        onClick={() => window.location.href = analysisResult.ohif_url!}
                        style={{
                          width: '100%',
                          padding: '8px',
                          background: '#10b981',
                          border: 'none',
                          borderRadius: '4px',
                          color: 'white',
                          cursor: 'pointer',
                          fontWeight: 'bold',
                        }}
                      >
                        Open Results in Viewer
                      </button>
                    )}
                  </>
                ) : (
                  <div style={{ color: '#ef4444' }}>
                    {analysisResult.error || analysisResult.message || 'Analysis failed'}
                  </div>
                )}
              </div>
            )}

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                style={{
                  flex: 1,
                  padding: '10px',
                  background: isAnalyzing ? '#6b7280' : '#10b981',
                  border: 'none',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                  fontWeight: 'bold',
                }}
              >
                {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
              </button>
              <button
                onClick={closeModal}
                disabled={isAnalyzing}
                style={{
                  padding: '10px 20px',
                  background: '#475569',
                  border: 'none',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default StudyListPanel;
