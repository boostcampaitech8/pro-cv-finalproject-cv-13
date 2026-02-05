import React, { useState, useCallback, useRef, useEffect } from 'react';

const BACKEND_URL = '/api';

interface UploadPanelProps {
  commandsManager?: any;
  extensionManager?: any;
  servicesManager?: any;
}

interface ProcessResult {
  status: string;
  study_uid?: string;
  message?: string;
  ohif_url?: string;
  steps?: Record<string, any>;
}

interface Study {
  ID: string;
  MainDicomTags: {
    StudyInstanceUID: string;
    PatientName?: string;
    StudyDate?: string;
    StudyDescription?: string;
  };
  Series?: string[];
}

type TabMode = 'studies' | 'upload' | 'serverpath';

function UploadPanel({ commandsManager, extensionManager, servicesManager }: UploadPanelProps): React.ReactElement {
  // Tab mode
  const [tabMode, setTabMode] = useState<TabMode>('studies');

  // File upload state (Upload Tab)
  const [ctFile, setCtFile] = useState<File | null>(null);
  const [ptFile, setPtFile] = useState<File | null>(null);
  const [runTumor, setRunTumor] = useState<boolean>(false);
  const ctInputRef = useRef<HTMLInputElement>(null);
  const ptInputRef = useRef<HTMLInputElement>(null);

  // Server path state (Server Path Tab) - 단일 경로
  const [dataDir, setDataDir] = useState<string>('/data/test_data/hecktor_CHUM-021');

  // Common state
  const [patientName, setPatientName] = useState<string>('Test Patient');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progressMessage, setProgressMessage] = useState<string>('');

  // Study list state
  const [studies, setStudies] = useState<Study[]>([]);
  const [isLoadingStudies, setIsLoadingStudies] = useState<boolean>(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Load studies on mount and when tab changes
  useEffect(() => {
    if (tabMode === 'studies') {
      loadStudies();
    }
  }, [tabMode]);

  const loadStudies = async () => {
    setIsLoadingStudies(true);
    try {
      const response = await fetch(`${BACKEND_URL}/orthanc/studies`);
      if (response.ok) {
        const data = await response.json();
        setStudies(data.studies || []);
      }
    } catch (err) {
      console.error('Failed to load studies:', err);
    }
    setIsLoadingStudies(false);
  };

  const deleteStudy = async (studyId: string) => {
    if (!confirm('Are you sure you want to delete this study?')) {
      return;
    }

    setDeletingId(studyId);
    try {
      const response = await fetch(`${BACKEND_URL}/orthanc/studies/${studyId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        await loadStudies();
      } else {
        alert('Failed to delete study');
      }
    } catch (err) {
      console.error('Failed to delete study:', err);
      alert('Failed to delete study');
    }
    setDeletingId(null);
  };

  const openStudy = (studyUid: string) => {
    window.location.href = `/nerve-assessment?StudyInstanceUIDs=${studyUid}`;
  };

  // Upload Tab: CT Upload + Segmentation
  const handleUploadAndSegment = useCallback(async () => {
    if (!ctFile) {
      setError('CT 파일을 선택하세요 (NIfTI 또는 DICOM ZIP)');
      return;
    }

    // Tumor 분할 요청했는데 PT 없으면 경고
    if (runTumor && !ptFile) {
      setError('Tumor 분할을 위해서는 PT (PET) 파일이 필요합니다');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);
    setProgressMessage('파일 업로드 중...');

    try {
      const formData = new FormData();
      formData.append('ct_file', ctFile);
      formData.append('patient_name', patientName);
      formData.append('run_tumor', String(runTumor));

      if (ptFile) {
        formData.append('pt_file', ptFile);
      }

      setProgressMessage(runTumor
        ? '분할 모델 실행 중 (TSv2 + nnUNet + Tumor)...'
        : '분할 모델 실행 중 (TSv2 + nnUNet)...'
      );

      const response = await fetch(`${BACKEND_URL}/upload-ct-and-segment`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Upload and segmentation failed');
      }

      const data = await response.json();
      setResult(data);

      if (data.status === 'success' && data.study_uid) {
        setProgressMessage('Success! Redirecting to viewer...');
        setTimeout(() => {
          window.location.href = `/nerve-assessment?StudyInstanceUIDs=${data.study_uid}`;
        }, 1500);
      }
    } catch (err: any) {
      setError(err.message || 'Processing failed');
    } finally {
      setIsProcessing(false);
      setProgressMessage('');
    }
  }, [ctFile, ptFile, patientName, runTumor]);

  // Server Path Tab: Import existing data
  const handleImportData = useCallback(async () => {
    if (!dataDir.trim()) {
      setError('데이터 폴더 경로를 입력하세요');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);
    setProgressMessage('데이터 가져오는 중...');

    try {
      const response = await fetch(`${BACKEND_URL}/import-existing-data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_dir: dataDir,
          patient_name: patientName,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Import failed');
      }

      const data = await response.json();
      setResult(data);

      if (data.status === 'success' && data.study_uid) {
        setProgressMessage('Success! Redirecting to viewer...');
        setTimeout(() => {
          window.location.href = `/nerve-assessment?StudyInstanceUIDs=${data.study_uid}`;
        }, 1500);
      }
    } catch (err: any) {
      setError(err.message || 'Import failed');
    } finally {
      setIsProcessing(false);
      setProgressMessage('');
    }
  }, [dataDir, patientName]);

  // Styles
  const inputStyle: React.CSSProperties = {
    width: '100%',
    padding: '8px',
    marginBottom: '8px',
    backgroundColor: '#2c3e50',
    border: '1px solid #4a5568',
    borderRadius: '4px',
    color: 'white',
    fontSize: '12px',
  };

  const buttonStyle: React.CSSProperties = {
    width: '100%',
    padding: '12px',
    backgroundColor: '#3b82f6',
    border: 'none',
    borderRadius: '4px',
    color: 'white',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold',
  };

  const buttonDisabledStyle: React.CSSProperties = {
    ...buttonStyle,
    backgroundColor: '#6b7280',
    cursor: 'not-allowed',
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
    textAlign: 'center',
  });

  const fileButtonStyle: React.CSSProperties = {
    width: '100%',
    padding: '16px',
    backgroundColor: '#1e293b',
    border: '2px dashed #4a5568',
    borderRadius: '4px',
    color: '#a0aec0',
    cursor: 'pointer',
    fontSize: '12px',
    textAlign: 'center',
    marginBottom: '12px',
  };

  const formatDate = (dateStr?: string) => {
    if (!dateStr || dateStr.length !== 8) return dateStr || '-';
    return `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}`;
  };

  return (
    <div style={{ padding: '16px', color: 'white', height: '100%', overflow: 'auto' }}>
      {/* Main Tabs - 3 tabs */}
      <div style={{ display: 'flex', gap: '2px', marginBottom: '0' }}>
        <button style={tabStyle(tabMode === 'studies')} onClick={() => setTabMode('studies')}>
          Studies
        </button>
        <button style={tabStyle(tabMode === 'upload')} onClick={() => setTabMode('upload')}>
          Upload
        </button>
        <button style={tabStyle(tabMode === 'serverpath')} onClick={() => setTabMode('serverpath')}>
          Server Path
        </button>
      </div>

      <div style={{
        backgroundColor: '#1e293b',
        padding: '16px',
        borderRadius: '0 0 4px 4px',
        minHeight: '300px',
      }}>
                {tabMode === 'studies' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <h4 style={{ margin: 0, fontSize: '14px' }}>Orthanc Studies</h4>
              <button
                onClick={loadStudies}
                disabled={isLoadingStudies}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#475569',
                  border: 'none',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '12px',
                }}
              >
                {isLoadingStudies ? '...' : 'Refresh'}
              </button>
            </div>

            {isLoadingStudies ? (
              <div style={{ textAlign: 'center', padding: '20px', color: '#64748b' }}>
                Loading studies...
              </div>
            ) : studies.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '20px', color: '#64748b' }}>
                No studies found
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {studies.map((study) => (
                  <div
                    key={study.ID}
                    style={{
                      padding: '12px',
                      backgroundColor: '#0f172a',
                      borderRadius: '6px',
                      borderLeft: '3px solid #3b82f6',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 'bold', fontSize: '13px', marginBottom: '4px' }}>
                          {study.MainDicomTags?.PatientName || 'Unknown Patient'}
                        </div>
                        <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '2px' }}>
                          {formatDate(study.MainDicomTags?.StudyDate)}
                        </div>
                        <div style={{ fontSize: '10px', color: '#64748b', wordBreak: 'break-all' }}>
                          {study.MainDicomTags?.StudyDescription || study.MainDicomTags?.StudyInstanceUID?.slice(0, 40) + '...'}
                        </div>
                        <div style={{ fontSize: '10px', color: '#475569', marginTop: '4px' }}>
                          {study.Series?.length || 0} series
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: '4px', marginLeft: '8px' }}>
                        <button
                          onClick={() => openStudy(study.MainDicomTags?.StudyInstanceUID)}
                          style={{
                            padding: '6px 10px',
                            backgroundColor: '#10b981',
                            border: 'none',
                            borderRadius: '4px',
                            color: 'white',
                            cursor: 'pointer',
                            fontSize: '11px',
                          }}
                        >
                          Open
                        </button>
                        <button
                          onClick={() => deleteStudy(study.ID)}
                          disabled={deletingId === study.ID}
                          style={{
                            padding: '6px 10px',
                            backgroundColor: deletingId === study.ID ? '#6b7280' : '#ef4444',
                            border: 'none',
                            borderRadius: '4px',
                            color: 'white',
                            cursor: deletingId === study.ID ? 'not-allowed' : 'pointer',
                            fontSize: '11px',
                          }}
                        >
                          {deletingId === study.ID ? '...' : 'Del'}
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

                {tabMode === 'upload' && (
          <div>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '14px' }}>CT (+PT) Upload + Auto Segmentation</h4>
            <p style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '16px' }}>
              CT 파일을 업로드하면 분할 모델이 자동 실행됩니다. NIfTI(.nii.gz) 또는 DICOM(.zip) 지원.
            </p>

            {/* CT File Selection */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: '#a0aec0' }}>
                CT File (NIfTI .nii.gz 또는 DICOM .zip) - 필수:
              </label>
              <input
                ref={ctInputRef}
                type="file"
                accept=".nii,.nii.gz,.gz,.zip,.dcm"
                style={{ display: 'none' }}
                onChange={(e) => setCtFile(e.target.files?.[0] || null)}
                disabled={isProcessing}
              />
              <div
                style={fileButtonStyle}
                onClick={() => !isProcessing && ctInputRef.current?.click()}
              >
                {ctFile ? (
                  <span style={{ color: '#10b981' }}>{ctFile.name}</span>
                ) : (
                  'Click to select CT file (NIfTI or DICOM ZIP)'
                )}
              </div>
            </div>

            {/* PT File Selection */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: '#a0aec0' }}>
                PT (PET) File (NIfTI .nii.gz 또는 DICOM .zip) - Tumor 시 필수:
              </label>
              <input
                ref={ptInputRef}
                type="file"
                accept=".nii,.nii.gz,.gz,.zip,.dcm"
                style={{ display: 'none' }}
                onChange={(e) => setPtFile(e.target.files?.[0] || null)}
                disabled={isProcessing}
              />
              <div
                style={{
                  ...fileButtonStyle,
                  borderColor: runTumor && !ptFile ? '#ef4444' : '#4a5568',
                }}
                onClick={() => !isProcessing && ptInputRef.current?.click()}
              >
                {ptFile ? (
                  <span style={{ color: '#10b981' }}>{ptFile.name}</span>
                ) : (
                  'Click to select PT file (NIfTI or DICOM ZIP)'
                )}
              </div>
            </div>

            {/* Run Tumor Checkbox */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '12px' }}>
                <input
                  type="checkbox"
                  checked={runTumor}
                  onChange={(e) => setRunTumor(e.target.checked)}
                  disabled={isProcessing}
                  style={{ marginRight: '8px' }}
                />
                <span style={{ color: runTumor ? '#f59e0b' : '#a0aec0' }}>
                  Tumor 분할 실행 (CT+PT 필요)
                </span>
              </label>
              {runTumor && !ptFile && (
                <div style={{ fontSize: '11px', color: '#ef4444', marginTop: '4px', marginLeft: '20px' }}>
                  ⚠️ Tumor 분할을 위해 PT 파일을 선택하세요
                </div>
              )}
            </div>

            {/* Patient Name */}
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: '#a0aec0' }}>
                Patient Name:
              </label>
              <input
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                style={inputStyle}
                disabled={isProcessing}
              />
            </div>

            {/* Upload Button */}
            <button
              onClick={handleUploadAndSegment}
              disabled={isProcessing || !ctFile || (runTumor && !ptFile)}
              style={isProcessing || !ctFile || (runTumor && !ptFile) ? buttonDisabledStyle : { ...buttonStyle, backgroundColor: '#10b981' }}
            >
              {isProcessing ? progressMessage || 'Processing...' : 'Upload & Run Segmentation'}
            </button>

            {/* Info Box */}
            <div style={{
              marginTop: '16px',
              padding: '12px',
              backgroundColor: '#0f172a',
              borderRadius: '6px',
              fontSize: '11px',
              color: '#64748b',
            }}>
              <div style={{ marginBottom: '4px', color: '#94a3b8' }}>Supported Formats:</div>
              <ul style={{ margin: '0 0 8px 0', paddingLeft: '16px' }}>
                <li>NIfTI: .nii, .nii.gz</li>
                <li>DICOM: .zip (DICOM 시리즈 포함)</li>
              </ul>
              <div style={{ marginBottom: '4px', color: '#94a3b8' }}>Processing Pipeline:</div>
              <ol style={{ margin: '0', paddingLeft: '16px' }}>
                <li>CT (+PT) 업로드 & DICOM→NIfTI 변환</li>
                <li>Normal Structure 분할 (TSv2 + nnUNet)</li>
                {runTumor && <li style={{ color: '#f59e0b' }}>Tumor 분할 (STU-Net)</li>}
                <li>DICOM 변환 및 Orthanc 업로드</li>
                <li>Viewer로 자동 이동</li>
              </ol>
            </div>
          </div>
        )}

                {tabMode === 'serverpath' && (
          <div>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '14px' }}>Import Existing Data</h4>
            <p style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '16px' }}>
              이미 분할이 완료된 데이터를 서버 경로로 가져옵니다.
            </p>

            {/* Data Directory (단일 경로) */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: '#a0aec0' }}>
                Data Directory (server path):
              </label>
              <input
                type="text"
                value={dataDir}
                onChange={(e) => setDataDir(e.target.value)}
                placeholder="/data/test_data/hecktor_CHUM-021"
                style={inputStyle}
                disabled={isProcessing}
              />
            </div>

            {/* Patient Name */}
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: '#a0aec0' }}>
                Patient Name:
              </label>
              <input
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                style={inputStyle}
                disabled={isProcessing}
              />
            </div>

            {/* Import Button */}
            <button
              onClick={handleImportData}
              disabled={isProcessing || !dataDir.trim()}
              style={isProcessing || !dataDir.trim() ? buttonDisabledStyle : buttonStyle}
            >
              {isProcessing ? progressMessage || 'Processing...' : 'Import Data'}
            </button>

            {/* Info Box */}
            <div style={{
              marginTop: '16px',
              padding: '12px',
              backgroundColor: '#0f172a',
              borderRadius: '6px',
              fontSize: '11px',
              color: '#64748b',
            }}>
              <div style={{ marginBottom: '4px', color: '#94a3b8' }}>Expected folder structure:</div>
              <pre style={{ margin: '4px 0', fontSize: '10px', color: '#64748b' }}>
{`data_dir/
├── ct.nii.gz          (CT 파일)
├── normal_structure/  (정상 구조물)
│   ├── trachea.nii.gz
│   ├── esophagus.nii.gz
│   └── ...
└── tumor/             (optional)
    └── tumor.nii.gz`}
              </pre>
            </div>
          </div>
        )}

                {error && (
          <div style={{
            marginTop: '12px',
            padding: '10px',
            backgroundColor: 'rgba(239, 68, 68, 0.2)',
            border: '1px solid #ef4444',
            borderRadius: '4px',
            color: '#ef4444',
            fontSize: '12px',
          }}>
            {error}
          </div>
        )}

        {result && (
          <div style={{
            marginTop: '12px',
            padding: '12px',
            backgroundColor: '#0f172a',
            borderRadius: '4px',
          }}>
            <div style={{
              display: 'inline-block',
              padding: '4px 8px',
              backgroundColor: result.status === 'success' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
              color: result.status === 'success' ? '#10b981' : '#f59e0b',
              borderRadius: '4px',
              fontSize: '11px',
              fontWeight: 'bold',
              marginBottom: '8px',
            }}>
              {result.status?.toUpperCase()}
            </div>

            {result.study_uid && (
              <div style={{ fontSize: '12px', color: '#10b981' }}>
                Redirecting to viewer...
              </div>
            )}

            {result.message && (
              <div style={{ fontSize: '12px', marginTop: '4px' }}>
                {result.message}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default UploadPanel;
