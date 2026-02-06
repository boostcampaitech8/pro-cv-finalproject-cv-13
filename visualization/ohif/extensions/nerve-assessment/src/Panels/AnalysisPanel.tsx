import React, { useState, useEffect, useCallback } from 'react';
import {
  getSegmentStates,
  toggleSegmentVisibility,
} from '../utils/onDemandSegmentationManager';
import {
  loadSurfaceMeshesToViewport,
  removeSurfaceMeshes,
  setVolumeHUThreshold,
  startAutoRotation,
  stopAutoRotation,
  setRotationSpeed as setSurfaceRotationSpeed,
} from '../utils/surfaceLoader';

const BACKEND_URL = '/api';

interface AnalysisPanelProps {
  commandsManager?: any;
  extensionManager?: any;
  servicesManager?: any;
}

interface Nerve {
  name: string;
  nerve?: string;
  side?: string;
  type?: string;
  method?: string;
  reference?: string;
  uncertainty_mm?: number;
  risk_level?: string;
  min_distance_to_tumor?: number;
  effective_distance_mm?: number;
  path_length?: number;
  overlap?: boolean;
  overlap_ratio?: number;
  has_risk_assessment?: boolean;
}

interface NerveResults {
  nerves?: Nerve[];
  analysis_summary?: {
    total_nerves?: number;
    high_risk_count?: number;
    moderate_count?: number;
    low_count?: number;
  };
}

interface RiskReport {
  status: string;
  message?: string;
  total_nerves?: number;
  total_risks_assessed?: number;
  high_risk_count?: number;
  moderate_risk_count?: number;
  low_risk_count?: number;
  high_risk_nerves?: string[];
  overall_risk?: string;
}

interface AnalysisResult {
  status: string;
  session_id?: string;
  study_uid?: string;
  nerve_results?: NerveResults;
  risk_report?: RiskReport;
  ohif_url?: string;
  steps?: Record<string, any>;
  message?: string;
}

interface SegmentItem {
  labelIndex: number;
  label: string;
  color: number[];
  isVisible: boolean;
}

type TabMode = 'segmentation' | 'analysis';

function AnalysisPanel({ servicesManager }: AnalysisPanelProps): React.ReactElement {
  const [tabMode, setTabMode] = useState<TabMode>('segmentation');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [studyInstanceUID, setStudyInstanceUID] = useState<string>('');
  const [segmentationDir, setSegmentationDir] = useState<string>('');
  const [progressMessage, setProgressMessage] = useState<string>('');
  const [segments, setSegments] = useState<SegmentItem[]>([]);
  const [showSegments, setShowSegments] = useState<boolean>(true);
  const [is3DLoading, setIs3DLoading] = useState<boolean>(false);
  const [is3DLoaded, setIs3DLoaded] = useState<boolean>(false);
  const [meshError, setMeshError] = useState<string | null>(null);
  const [huMin, setHuMin] = useState<number>(300);
  const [huOpacity, setHuOpacity] = useState<number>(15);
  const [showHUControls, setShowHUControls] = useState<boolean>(false);
  const [isRotating, setIsRotating] = useState<boolean>(false);
  const [rotationSpeed, setRotationSpeed] = useState<number>(50);

  // Stop auto-rotation on unmount
  useEffect(() => {
    return () => {
      stopAutoRotation();
    };
  }, []);

  // Get StudyInstanceUID from URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const uid = params.get('StudyInstanceUIDs') || '';
    setStudyInstanceUID(uid);
    console.log('[AnalysisPanel] StudyInstanceUID from URL:', uid);
  }, []);

  // Refresh segment list periodically
  const refreshSegments = useCallback(() => {
    if (!studyInstanceUID) return;
    const states = getSegmentStates(studyInstanceUID);
    setSegments(states.map(s => ({
      labelIndex: s.labelIndex,
      label: s.label,
      color: s.color,
      isVisible: s.isVisible,
    })));
  }, [studyInstanceUID]);

  useEffect(() => {
    refreshSegments();
    const interval = setInterval(refreshSegments, 1000);
    return () => clearInterval(interval);
  }, [refreshSegments]);

  const handleToggleSegment = async (segmentLabel: string) => {
    if (!studyInstanceUID || !servicesManager) return;
    const { cornerstoneViewportService } = servicesManager.services;
    const viewportIds = cornerstoneViewportService?.getViewportIds?.() || [];
    const viewportId = viewportIds[0] || 'mpr-axial';
    await toggleSegmentVisibility(segmentLabel, studyInstanceUID, viewportId, servicesManager);
    refreshSegments();
  };

  const load3DSurfaces = async () => {
    if (!studyInstanceUID || !servicesManager) {
      setMeshError('Study or services not available');
      return;
    }

    setIs3DLoading(true);
    setMeshError(null);

    try {
      const success = await loadSurfaceMeshesToViewport(studyInstanceUID, servicesManager);

      if (success) {
        setIs3DLoaded(true);
      } else {
        setMeshError('Failed to load 3D surfaces. Run nerve analysis first.');
      }
    } catch (error: any) {
      console.error('[AnalysisPanel] 3D loading error:', error);
      setMeshError(error.message || 'Unknown error loading 3D surfaces');
    }

    setIs3DLoading(false);
  };

  const remove3DSurfaces = async () => {
    if (!studyInstanceUID || !servicesManager) return;

    try {
      await removeSurfaceMeshes(studyInstanceUID, servicesManager);
      setIs3DLoaded(false);
    } catch (error) {
      console.error('[AnalysisPanel] Failed to remove 3D surfaces:', error);
    }
  };

  const runNerveAnalysis = async () => {
    if (!studyInstanceUID) {
      setError('StudyInstanceUID not found in URL');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setAnalysisResult(null);
    setProgressMessage('DICOM 다운로드 중...');

    try {
      console.log('[AnalysisPanel] Starting nerve analysis for:', studyInstanceUID);

      let url = `${BACKEND_URL}/run-nerve-analysis/${studyInstanceUID}`;
      if (segmentationDir.trim()) {
        url += `?segmentation_dir=${encodeURIComponent(segmentationDir.trim())}`;
      }

      setProgressMessage('신경 추정 실행 중...');

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();
      console.log('[AnalysisPanel] Analysis result:', data);

      if (response.ok) {
        setAnalysisResult(data);

        if (data.status === 'success') {
          setProgressMessage('분석 완료!');
        } else if (data.status === 'incomplete') {
          setProgressMessage('분할 결과가 필요합니다');
        }
      } else {
        setError(data.detail || 'Analysis failed');
      }
    } catch (e: any) {
      console.error('[AnalysisPanel] Analysis error:', e);
      setError(e.message || 'Failed to connect to backend');
    }

    setIsAnalyzing(false);
    setProgressMessage('');
  };

  const getRiskColor = (level?: string): string => {
    const colors: Record<string, string> = {
      high: '#ef4444',
      moderate: '#3b82f6',
      low: '#10b981',
    };
    return colors[level?.toLowerCase() || ''] || '#64748b';
  };

  const getRiskIcon = (level?: string): string => {
    const icons: Record<string, string> = {
      high: '🔴',
      moderate: '🟡',
      low: '🟢',
    };
    return icons[level?.toLowerCase() || ''] || '⚪';
  };

  const getRiskBadge = (level?: string): React.ReactElement => {
    const color = getRiskColor(level);
    return (
      <span
        style={{
          padding: '2px 8px',
          borderRadius: '12px',
          fontSize: '10px',
          fontWeight: 'bold',
          backgroundColor: `${color}22`,
          color: color,
          border: `1px solid ${color}`,
        }}
      >
        {(level || 'UNKNOWN').toUpperCase()}
      </span>
    );
  };

  const refreshPage = () => {
    window.location.reload();
  };

  const tabStyle = (active: boolean): React.CSSProperties => ({
    flex: 1,
    padding: '10px 8px',
    backgroundColor: active ? '#3b82f6' : '#1e293b',
    border: 'none',
    borderRadius: '4px 4px 0 0',
    color: 'white',
    cursor: 'pointer',
    fontSize: '12px',
    fontWeight: active ? 'bold' : 'normal',
  });

  return (
    <div style={{ padding: '16px', color: 'white', height: '100%', overflow: 'auto' }}>
      {/* Study Info (common) */}
      <div style={{
        marginBottom: '16px',
        padding: '10px',
        background: '#0f172a',
        borderRadius: '6px',
      }}>
        <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '4px' }}>Current Study</div>
        <div style={{ fontSize: '10px', color: '#64748b', wordBreak: 'break-all' }}>
          {studyInstanceUID ? studyInstanceUID.slice(0, 50) + '...' : 'No study loaded'}
        </div>
      </div>

      {/* Tab Buttons */}
      <div style={{ display: 'flex', gap: '2px', marginBottom: 0 }}>
        <button style={tabStyle(tabMode === 'segmentation')}
          onClick={() => setTabMode('segmentation')}>
          Segmentation
        </button>
        <button style={tabStyle(tabMode === 'analysis')}
          onClick={() => setTabMode('analysis')}>
          Nerve Analysis
        </button>
      </div>

      {/* Tab Content Container */}
      <div style={{
        background: '#1e293b',
        borderRadius: '0 0 6px 6px',
        padding: '12px',
        marginBottom: '16px',
      }}>

      {tabMode === 'segmentation' && (
        <div>
          {/* Segment Visibility Controls */}
          {segments.length > 0 && (
            <div style={{
              marginBottom: '16px',
              background: '#0f172a',
              borderRadius: '6px',
              overflow: 'hidden',
            }}>
              <div
                onClick={() => setShowSegments(!showSegments)}
                style={{
                  padding: '10px 12px',
                  background: '#334155',
                  cursor: 'pointer',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <span style={{ fontSize: '12px', fontWeight: 'bold' }}>
                  Segments ({segments.length})
                </span>
                <span style={{ fontSize: '10px', color: '#94a3b8' }}>
                  {showSegments ? '▼' : '▶'} {segments.filter(s => s.isVisible).length} visible
                </span>
              </div>

              {showSegments && (
                <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                  {segments.map((seg) => (
                    <div
                      key={seg.labelIndex}
                      onClick={() => handleToggleSegment(seg.label)}
                      style={{
                        padding: '8px 12px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        cursor: 'pointer',
                        borderBottom: '1px solid #334155',
                        background: seg.isVisible ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                      }}
                    >
                      <div style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '3px',
                        background: `rgb(${seg.color[0]}, ${seg.color[1]}, ${seg.color[2]})`,
                        opacity: seg.isVisible ? 1 : 0.3,
                        flexShrink: 0,
                      }} />

                      <span style={{
                        fontSize: '11px',
                        color: seg.isVisible ? '#e2e8f0' : '#64748b',
                        flex: 1,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}>
                        {seg.label}
                      </span>

                      <div style={{
                        width: '32px',
                        height: '16px',
                        borderRadius: '8px',
                        background: seg.isVisible ? '#3b82f6' : '#475569',
                        position: 'relative',
                        flexShrink: 0,
                      }}>
                        <div style={{
                          width: '12px',
                          height: '12px',
                          borderRadius: '50%',
                          background: 'white',
                          position: 'absolute',
                          top: '2px',
                          left: seg.isVisible ? '18px' : '2px',
                          transition: 'left 0.2s',
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {segments.length === 0 && (
            <div style={{ textAlign: 'center', color: '#64748b', padding: '16px 0' }}>
              <div style={{ fontSize: '32px', marginBottom: '8px' }}>📦</div>
              <p style={{ fontSize: '11px', margin: 0 }}>No segments loaded</p>
              <p style={{ fontSize: '10px', marginTop: '4px' }}>
                Run analysis or load segmentation data first
              </p>
            </div>
          )}

          {/* 3D Surface Loading Section */}
          <div style={{
            marginBottom: '16px',
            padding: '12px',
            background: '#0f172a',
            borderRadius: '6px',
          }}>
            <div style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '8px' }}>
              3D Surface Rendering
            </div>
            <div style={{ fontSize: '10px', color: '#94a3b8', marginBottom: '8px' }}>
              Load 3D mesh from server
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={load3DSurfaces}
                disabled={is3DLoading || !studyInstanceUID}
                style={{
                  flex: 1,
                  padding: '10px',
                  background: is3DLoading ? '#475569' : is3DLoaded ? '#059669' : '#2563eb',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  fontSize: '12px',
                  fontWeight: 'bold',
                  cursor: is3DLoading || !studyInstanceUID ? 'not-allowed' : 'pointer',
                }}
              >
                {is3DLoading ? (
                  <>
                    <span className="spinner">⏳</span> Loading...
                  </>
                ) : is3DLoaded ? (
                  '✅ 3D Loaded'
                ) : (
                  'Load 3D Mesh'
                )}
              </button>

              {is3DLoaded && (
                <button
                  onClick={remove3DSurfaces}
                  style={{
                    padding: '10px',
                    background: '#dc2626',
                    border: 'none',
                    borderRadius: '6px',
                    color: 'white',
                    fontSize: '12px',
                    cursor: 'pointer',
                  }}
                >
                  Remove
                </button>
              )}
            </div>

            {/* Volume Rendering Controls */}
            <div style={{ marginTop: '10px' }}>
              {/* HU Threshold toggle */}
              <div
                onClick={() => setShowHUControls(!showHUControls)}
                style={{
                  marginTop: '8px',
                  padding: '6px 8px',
                  background: '#334155',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '10px',
                  color: '#94a3b8',
                }}
              >
                <span>HU Threshold</span>
                <span>{showHUControls ? '▼' : '▶'}</span>
              </div>

              {showHUControls && (
                <div style={{ padding: '8px', background: '#0f172a', borderRadius: '0 0 4px 4px' }}>
                  <div style={{ marginBottom: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94a3b8', marginBottom: '2px' }}>
                      <span>Min HU (bone threshold)</span>
                      <span>{huMin} HU</span>
                    </div>
                    <input
                      type="range"
                      min={-200}
                      max={800}
                      step={10}
                      value={huMin}
                      onChange={(e) => {
                        const val = Number(e.target.value);
                        setHuMin(val);
                        if (servicesManager) {
                          setVolumeHUThreshold(servicesManager, val, 3000, huOpacity / 100, 100);
                        }
                      }}
                      style={{ width: '100%', accentColor: '#3b82f6' }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#64748b' }}>
                      <span>-200 (soft)</span>
                      <span>800 (dense bone)</span>
                    </div>
                  </div>

                  <div style={{ marginBottom: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94a3b8', marginBottom: '2px' }}>
                      <span>Opacity</span>
                      <span>{huOpacity}%</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={100}
                      step={1}
                      value={huOpacity}
                      onChange={(e) => {
                        const val = Number(e.target.value);
                        setHuOpacity(val);
                        if (servicesManager) {
                          setVolumeHUThreshold(servicesManager, huMin, 3000, val / 100, 100);
                        }
                      }}
                      style={{ width: '100%', accentColor: '#3b82f6' }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#64748b' }}>
                      <span>1% (faint)</span>
                      <span>100% (solid)</span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                    {[
                      { label: 'Bone Only', min: 400, op: 12 },
                      { label: 'Bone+Soft', min: 100, op: 8 },
                      { label: 'Faint Bone', min: 300, op: 5 },
                      { label: 'Dense Only', min: 600, op: 20 },
                    ].map((p) => (
                      <button
                        key={p.label}
                        onClick={() => {
                          setHuMin(p.min);
                          setHuOpacity(p.op);
                          if (servicesManager) {
                            setVolumeHUThreshold(servicesManager, p.min, 3000, p.op / 100, 100);
                          }
                        }}
                        style={{
                          padding: '3px 8px',
                          background: '#1e293b',
                          border: '1px solid #475569',
                          borderRadius: '3px',
                          color: '#94a3b8',
                          fontSize: '9px',
                          cursor: 'pointer',
                        }}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Auto Rotation Controls */}
            <div style={{ marginTop: '8px' }}>
              <div
                style={{
                  padding: '6px 8px',
                  background: '#334155',
                  borderRadius: '4px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '10px',
                  color: '#94a3b8',
                }}
              >
                <span>Auto Rotation</span>
                <button
                  onClick={() => {
                    if (isRotating) {
                      stopAutoRotation();
                      setIsRotating(false);
                    } else {
                      setSurfaceRotationSpeed(0.1 + (rotationSpeed / 100) * 1.9);
                      startAutoRotation(servicesManager, () => setIsRotating(false));
                      setIsRotating(true);
                    }
                  }}
                  disabled={!is3DLoaded}
                  style={{
                    padding: '2px 10px',
                    background: !is3DLoaded ? '#475569' : isRotating ? '#dc2626' : '#2563eb',
                    border: 'none',
                    borderRadius: '3px',
                    color: 'white',
                    fontSize: '11px',
                    cursor: !is3DLoaded ? 'not-allowed' : 'pointer',
                  }}
                >
                  {isRotating ? '⏸ Stop' : '▶ Play'}
                </button>
              </div>

              <div style={{ padding: '8px', background: '#0f172a', borderRadius: '0 0 4px 4px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94a3b8', marginBottom: '2px' }}>
                  <span>Speed</span>
                  <span>{rotationSpeed}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={rotationSpeed}
                  disabled={!is3DLoaded}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setRotationSpeed(val);
                    const deg = 0.1 + (val / 100) * 1.9;
                    setSurfaceRotationSpeed(deg);
                  }}
                  style={{ width: '100%', accentColor: '#3b82f6' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#64748b' }}>
                  <span>Slow</span>
                  <span>Fast</span>
                </div>
              </div>
            </div>

            {meshError && (
              <div style={{
                marginTop: '8px',
                padding: '8px',
                background: '#7f1d1d',
                borderRadius: '4px',
                fontSize: '10px',
                color: '#fca5a5',
              }}>
                {meshError}
              </div>
            )}
          </div>
        </div>
      )}

      {tabMode === 'analysis' && (
        <div>
          {/* Segmentation Directory Input (optional - for server path data) */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>
              Segmentation Directory (optional)
            </label>
            <input
              type="text"
              value={segmentationDir}
              onChange={(e) => setSegmentationDir(e.target.value)}
              placeholder="/data/test_data/case_001"
              style={{
                width: '100%',
                padding: '8px',
                background: '#0f172a',
                border: '1px solid #475569',
                borderRadius: '4px',
                color: 'white',
                fontSize: '11px',
                boxSizing: 'border-box',
              }}
              disabled={isAnalyzing}
            />
            <div style={{ fontSize: '10px', color: '#64748b', marginTop: '4px' }}>
              Upload 사용 시 자동 설정됨. Server path 사용 시 입력 (normal_structure 폴더 포함 경로)
            </div>
          </div>

          <button
            onClick={runNerveAnalysis}
            disabled={isAnalyzing || !studyInstanceUID}
            style={{
              width: '100%',
              padding: '16px',
              background: isAnalyzing ? '#475569' : !studyInstanceUID ? '#374151' : '#10b981',
              border: 'none',
              borderRadius: '8px',
              color: 'white',
              fontSize: '16px',
              fontWeight: 'bold',
              cursor: isAnalyzing || !studyInstanceUID ? 'not-allowed' : 'pointer',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              transition: 'background-color 0.2s',
            }}
          >
            {isAnalyzing ? (
              <>
                <span className="spinner">⏳</span>
                {progressMessage || 'Analyzing...'}
              </>
            ) : (
              'Run Analysis'
            )}
          </button>

          {error && (
            <div style={{
              padding: '12px',
              background: '#7f1d1d',
              borderRadius: '6px',
              marginBottom: '16px',
              fontSize: '12px',
            }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Incomplete Status (needs segmentation) */}
          {analysisResult?.status === 'incomplete' && (
            <div style={{
              padding: '12px',
              background: '#422006',
              borderRadius: '6px',
              marginBottom: '16px',
              fontSize: '12px',
            }}>
              <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>Segmentation Required</div>
              <div style={{ color: '#fbbf24' }}>{analysisResult.message}</div>
            </div>
          )}

          {/* Analysis Status & Results */}
          {analysisResult?.status === 'success' && (
            <div>
              <div style={{
                padding: '12px',
                background: '#052e16',
                borderRadius: '6px',
                marginBottom: '16px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}>
                <span style={{ fontSize: '14px' }}>Analysis Complete</span>
                <button
                  onClick={refreshPage}
                  style={{
                    padding: '6px 12px',
                    background: '#1d4ed8',
                    border: 'none',
                    borderRadius: '4px',
                    color: 'white',
                    fontSize: '11px',
                    cursor: 'pointer',
                  }}
                >
                  Refresh View
                </button>
              </div>

              {/* No Tumor Warning */}
              {analysisResult.risk_report && analysisResult.risk_report.status === 'no_tumor' && (
                <div style={{
                  marginBottom: '16px',
                  padding: '12px',
                  background: '#422006',
                  borderRadius: '6px',
                  fontSize: '12px',
                }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '4px', color: '#fbbf24' }}>
                    No Tumor Detected
                  </div>
                  <div style={{ color: '#fcd34d' }}>
                    {analysisResult.risk_report.message || 'Risk assessment requires tumor mask'}
                  </div>
                  <div style={{ color: '#94a3b8', marginTop: '4px', fontSize: '11px' }}>
                    Total nerves estimated: {analysisResult.risk_report.total_nerves || 0}
                  </div>
                </div>
              )}

              {/* Risk Report Summary */}
              {analysisResult.risk_report && analysisResult.risk_report.status === 'analyzed' && (
                <div style={{
                  marginBottom: '16px',
                  padding: '12px',
                  background: '#0f172a',
                  borderRadius: '6px',
                }}>
                  <div style={{ fontSize: '13px', fontWeight: 'bold', marginBottom: '12px' }}>
                    Risk Summary
                  </div>

                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    marginBottom: '12px',
                    padding: '8px',
                    background: '#1e293b',
                    borderRadius: '4px',
                  }}>
                    <span style={{ fontSize: '12px', color: '#94a3b8' }}>Overall Risk:</span>
                    <span style={{
                      padding: '4px 12px',
                      borderRadius: '12px',
                      fontSize: '12px',
                      fontWeight: 'bold',
                      backgroundColor: `${getRiskColor(analysisResult.risk_report.overall_risk)}22`,
                      color: getRiskColor(analysisResult.risk_report.overall_risk),
                      border: `1px solid ${getRiskColor(analysisResult.risk_report.overall_risk)}`,
                    }}>
                      {getRiskIcon(analysisResult.risk_report.overall_risk)} {analysisResult.risk_report.overall_risk}
                    </span>
                  </div>

                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '8px',
                    fontSize: '12px',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ color: '#ef4444' }}>●</span>
                      <span>High: {analysisResult.risk_report.high_risk_count || 0}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ color: '#3b82f6' }}>●</span>
                      <span>Moderate: {analysisResult.risk_report.moderate_risk_count || 0}</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ color: '#10b981' }}>●</span>
                      <span>Low: {analysisResult.risk_report.low_risk_count || 0}</span>
                    </div>
                  </div>

                  {analysisResult.risk_report.high_risk_nerves && analysisResult.risk_report.high_risk_nerves.length > 0 && (
                    <div style={{ marginTop: '12px', padding: '8px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '4px' }}>
                      <div style={{ fontSize: '11px', color: '#ef4444', marginBottom: '4px', fontWeight: 'bold' }}>
                        High Risk Nerves:
                      </div>
                      <div style={{ fontSize: '11px', color: '#fca5a5' }}>
                        {analysisResult.risk_report.high_risk_nerves.join(', ')}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Detailed Nerve Results */}
              {analysisResult.nerve_results?.nerves && analysisResult.nerve_results.nerves.length > 0 && (
                <div>
                  <div style={{ fontSize: '13px', fontWeight: 'bold', marginBottom: '12px', color: '#94a3b8' }}>
                    Nerve Details ({analysisResult.nerve_results.nerves.length})
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {analysisResult.nerve_results.nerves.map((nerve, i) => (
                      <div
                        key={i}
                        style={{
                          padding: '10px',
                          background: '#0f172a',
                          borderRadius: '6px',
                          borderLeft: `3px solid ${getRiskColor(nerve.risk_level)}`,
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                          <span style={{ fontWeight: 'bold', fontSize: '12px' }}>
                            {getRiskIcon(nerve.risk_level)} {nerve.name}
                          </span>
                          {getRiskBadge(nerve.risk_level)}
                        </div>

                        {/* Overlap Warning */}
                        {nerve.overlap && (
                          <div style={{
                            padding: '4px 8px',
                            marginBottom: '6px',
                            background: 'rgba(239, 68, 68, 0.2)',
                            borderRadius: '4px',
                            fontSize: '10px',
                            color: '#ef4444',
                          }}>
                            OVERLAP: {nerve.overlap_ratio !== undefined ? `${(nerve.overlap_ratio * 100).toFixed(0)}%` : 'detected'}
                          </div>
                        )}

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '10px', color: '#94a3b8' }}>
                          {nerve.min_distance_to_tumor !== undefined && nerve.min_distance_to_tumor !== null && (
                            <div>Distance: {nerve.min_distance_to_tumor.toFixed(1)}mm</div>
                          )}
                          {nerve.effective_distance_mm !== undefined && nerve.effective_distance_mm !== null && (
                            <div>Effective: {nerve.effective_distance_mm.toFixed(1)}mm</div>
                          )}
                          {nerve.path_length !== undefined && nerve.path_length !== null && (
                            <div>Length: {nerve.path_length.toFixed(1)}mm</div>
                          )}
                          {nerve.uncertainty_mm !== undefined && (
                            <div>+/-{nerve.uncertainty_mm}mm uncertainty</div>
                          )}
                        </div>

                        {/* Method info */}
                        {nerve.method && (
                          <div style={{ fontSize: '9px', color: '#64748b', marginTop: '4px' }}>
                            Method: {nerve.method}
                          </div>
                        )}

                        {/* No risk assessment warning */}
                        {!nerve.has_risk_assessment && (
                          <div style={{ fontSize: '9px', color: '#f59e0b', marginTop: '4px' }}>
                            No tumor for risk assessment
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {!analysisResult && !isAnalyzing && !error && (
            <div style={{ textAlign: 'center', color: '#64748b', marginTop: '24px' }}>
              <p style={{ fontSize: '12px', margin: 0 }}>
                Click "Run Analysis" to start
              </p>
              <p style={{ fontSize: '11px', marginTop: '4px' }}>
                nerve estimation
              </p>
            </div>
          )}
        </div>
      )}

      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .spinner {
          display: inline-block;
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
}

export default AnalysisPanel;
