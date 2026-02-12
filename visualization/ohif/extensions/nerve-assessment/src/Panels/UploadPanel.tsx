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
type UploadMode = 'single' | 'batch';

interface DetectedCase {
  id: string;
  type: 'nifti' | 'dicom_zip';
  files: File[];
  ptFile?: File;
  status: 'pending' | 'processing' | 'success' | 'failed';
  error?: string;
  studyUid?: string;
}

interface BatchProgress {
  total: number;
  completed: number;
  failed: number;
  current: string | null;
}

const CT_NAME_PATTERNS = [/^ct\.nii(\.gz)?$/i, /^image\.nii(\.gz)?$/i, /^ct_image\.nii(\.gz)?$/i];
const PT_NAME_PATTERNS = [/^pt\.nii(\.gz)?$/i, /^pet\.nii(\.gz)?$/i, /^pt_image\.nii(\.gz)?$/i];

function sanitizePatientName(name: string): string {
  return name
    .replace(/[^a-zA-Z0-9가-힣_\-]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '')
    .slice(0, 64);
}

function detectCases(files: FileList): DetectedCase[] {
  const cases: DetectedCase[] = [];
  const byFolder = new Map<string, File[]>();

  for (const file of Array.from(files)) {
    const parts = file.webkitRelativePath.split('/');

    if (parts.length === 2) {
      // Top-level files
      const name = parts[1];
      if (name.endsWith('.nii.gz') || name.endsWith('.nii')) {
        cases.push({
          id: name.replace(/\.nii(\.gz)?$/, ''),
          type: 'nifti',
          files: [file],
          status: 'pending',
        });
      } else if (name.endsWith('.zip')) {
        cases.push({
          id: name.replace(/\.zip$/, ''),
          type: 'dicom_zip',
          files: [file],
          status: 'pending',
        });
      }
    } else if (parts.length >= 3) {
      // Subfolder files
      const subFolder = parts[1];
      if (!byFolder.has(subFolder)) byFolder.set(subFolder, []);
      byFolder.get(subFolder)!.push(file);
    }
  }

  // Analyze subfolders (case-insensitive CT file matching)
  for (const [folderName, folderFiles] of byFolder) {
    // Priority 1: Name pattern matching
    let ctFile = folderFiles.find(f =>
      CT_NAME_PATTERNS.some(p => p.test(f.name))
    );

    // Priority 2: Fallback — if exactly one .nii.gz in folder, treat as CT
    if (!ctFile) {
      const niftiFiles = folderFiles.filter(f =>
        f.name.endsWith('.nii.gz') || f.name.endsWith('.nii')
      );
      if (niftiFiles.length === 1) {
        ctFile = niftiFiles[0];
      }
    }

    if (ctFile) {
      const ptFile = folderFiles.find(f =>
        PT_NAME_PATTERNS.some(p => p.test(f.name))
      );
      cases.push({
        id: folderName,
        type: 'nifti',
        files: [ctFile],
        ptFile: ptFile || undefined,
        status: 'pending',
      });
    }
  }

  return cases.sort((a, b) => a.id.localeCompare(b.id));
}

function UploadPanel({ commandsManager, extensionManager, servicesManager }: UploadPanelProps): React.ReactElement {
  // Tab mode
  const [tabMode, setTabMode] = useState<TabMode>('studies');

  // Upload mode (single vs batch)
  const [uploadMode, setUploadMode] = useState<UploadMode>('single');

  // File upload state (Upload Tab - Single)
  const [ctFile, setCtFile] = useState<File | null>(null);
  const [ptFile, setPtFile] = useState<File | null>(null);
  const [runTumor, setRunTumor] = useState<boolean>(false);
  const ctInputRef = useRef<HTMLInputElement>(null);
  const ptInputRef = useRef<HTMLInputElement>(null);

  // Batch upload state
  const [detectedCases, setDetectedCases] = useState<DetectedCase[]>([]);
  const [batchProgress, setBatchProgress] = useState<BatchProgress | null>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

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

  // Set webkitdirectory attribute when batch input is rendered
  useEffect(() => {
    if (uploadMode === 'batch' && folderInputRef.current) {
      folderInputRef.current.setAttribute('webkitdirectory', '');
      folderInputRef.current.setAttribute('directory', '');
    }
  }, [uploadMode]);

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

  // Batch folder selection handler
  const handleFolderSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const cases = detectCases(files);
    setDetectedCases(cases);
    setBatchProgress(null);
  }, []);

  // Batch processing
  const runBatch = useCallback(async () => {
    if (detectedCases.length === 0) return;

    // Snapshot to avoid stale closure during async loop
    const casesSnapshot = [...detectedCases];

    setIsProcessing(true);
    setError(null);
    setResult(null);
    const progress: BatchProgress = {
      total: casesSnapshot.length,
      completed: 0,
      failed: 0,
      current: null,
    };
    setBatchProgress({ ...progress });

    for (let i = 0; i < casesSnapshot.length; i++) {
      const caseItem = casesSnapshot[i];
      progress.current = caseItem.id;
      setBatchProgress({ ...progress });

      setDetectedCases(prev => prev.map((c, idx) =>
        idx === i ? { ...c, status: 'processing' } : c
      ));

      try {
        const formData = new FormData();
        formData.append('ct_file', caseItem.files[0]);
        formData.append('patient_name', sanitizePatientName(`${patientName}_${caseItem.id}`));
        formData.append('run_tumor', String(runTumor));
        if (caseItem.ptFile) {
          formData.append('pt_file', caseItem.ptFile);
        }

        const response = await fetch(`${BACKEND_URL}/upload-ct-and-segment`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Upload failed for ${caseItem.id}`);
        }

        const result = await response.json();

        setDetectedCases(prev => prev.map((c, idx) =>
          idx === i ? { ...c, status: 'success', studyUid: result.study_uid } : c
        ));
        progress.completed++;
      } catch (e: any) {
        setDetectedCases(prev => prev.map((c, idx) =>
          idx === i ? { ...c, status: 'failed', error: e.message } : c
        ));
        progress.failed++;
      }

      setBatchProgress({ ...progress });
    }

    progress.current = null;
    setBatchProgress({ ...progress });
    setIsProcessing(false);
  }, [detectedCases, patientName, runTumor]);

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
      {/* Main Tabs - 3 tabs (locked during batch processing only) */}
      {(() => { const batchLocked = isProcessing && uploadMode === 'batch'; return (
      <div style={{ display: 'flex', gap: '2px', marginBottom: '0' }}>
        <button
          style={{ ...tabStyle(tabMode === 'studies'), opacity: batchLocked && tabMode !== 'studies' ? 0.5 : 1 }}
          onClick={() => !batchLocked && setTabMode('studies')}
          disabled={batchLocked}
        >
          Studies
        </button>
        <button
          style={{ ...tabStyle(tabMode === 'upload'), opacity: batchLocked && tabMode !== 'upload' ? 0.5 : 1 }}
          onClick={() => !batchLocked && setTabMode('upload')}
          disabled={batchLocked}
        >
          Upload{batchLocked && batchProgress ? ` (${batchProgress.completed + batchProgress.failed}/${batchProgress.total})` : ''}
        </button>
        <button
          style={{ ...tabStyle(tabMode === 'serverpath'), opacity: batchLocked && tabMode !== 'serverpath' ? 0.5 : 1 }}
          onClick={() => !batchLocked && setTabMode('serverpath')}
          disabled={batchLocked}
        >
          Server Path
        </button>
      </div>
      ); })()}

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
                          {study.PatientMainDicomTags?.PatientName || study.MainDicomTags?.PatientName || 'Unknown Patient'}
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
            <h4 style={{ margin: '0 0 8px 0', fontSize: '14px' }}>CT Upload + Auto Segmentation</h4>

            {/* Mode Toggle */}
            <div style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '12px' }}>
                <input
                  type="radio"
                  name="uploadMode"
                  checked={uploadMode === 'single'}
                  onChange={() => setUploadMode('single')}
                  disabled={isProcessing}
                  style={{ marginRight: '6px' }}
                />
                단일 케이스
              </label>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '12px' }}>
                <input
                  type="radio"
                  name="uploadMode"
                  checked={uploadMode === 'batch'}
                  onChange={() => setUploadMode('batch')}
                  disabled={isProcessing}
                  style={{ marginRight: '6px' }}
                />
                폴더 일괄
              </label>
            </div>

            {uploadMode === 'single' && (
              <div>
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
                      'Click to select CT file'
                    )}
                  </div>
                </div>

                {/* PT File Selection */}
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: '#a0aec0' }}>
                    PT (PET) File (NIfTI .nii.gz 또는 DICOM .zip) - 선택사항:
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
                      borderColor: '#4a5568',
                    }}
                    onClick={() => !isProcessing && ptInputRef.current?.click()}
                  >
                    {ptFile ? (
                      <span style={{ color: '#10b981' }}>{ptFile.name}</span>
                    ) : (
                      'Click to select PT file'
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
                      Tumor 분할 실행
                    </span>
                  </label>
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
                  disabled={isProcessing || !ctFile}
                  style={isProcessing || !ctFile ? buttonDisabledStyle : { ...buttonStyle, backgroundColor: '#10b981' }}
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

            {uploadMode === 'batch' && (
              <div>
                <p style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '16px' }}>
                  폴더를 선택하면 CT 케이스를 자동 감지하여 일괄 처리합니다.
                </p>

                {/* Folder Selection */}
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: '#a0aec0' }}>
                    Select Folder:
                  </label>
                  <input
                    ref={folderInputRef}
                    type="file"
                    style={{ display: 'none' }}
                    onChange={handleFolderSelect}
                    disabled={isProcessing}
                  />
                  <div
                    style={fileButtonStyle}
                    onClick={() => !isProcessing && folderInputRef.current?.click()}
                  >
                    {detectedCases.length > 0 ? (
                      <span style={{ color: '#10b981' }}>
                        {detectedCases.length} case{detectedCases.length !== 1 ? 's' : ''} detected
                      </span>
                    ) : (
                      'Click to select folder'
                    )}
                  </div>
                </div>

                {/* Detected Cases List */}
                {detectedCases.length > 0 && (
                  <div style={{ marginBottom: '12px' }}>
                    <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: '#a0aec0' }}>
                      Detected Cases ({detectedCases.length}):
                    </label>
                    <div style={{
                      maxHeight: '200px',
                      overflowY: 'auto',
                      backgroundColor: '#0f172a',
                      borderRadius: '4px',
                      padding: '8px',
                    }}>
                      {detectedCases.map((c, idx) => (
                        <div
                          key={idx}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            padding: '4px 8px',
                            fontSize: '11px',
                            borderBottom: idx < detectedCases.length - 1 ? '1px solid #1e293b' : 'none',
                          }}
                        >
                          <span style={{ width: '16px', textAlign: 'center' }}>
                            {c.status === 'pending' && '⏸'}
                            {c.status === 'processing' && '⏳'}
                            {c.status === 'success' && '✅'}
                            {c.status === 'failed' && '❌'}
                          </span>
                          <span style={{
                            flex: 1,
                            color: c.status === 'success' ? '#10b981'
                              : c.status === 'failed' ? '#ef4444'
                              : c.status === 'processing' ? '#f59e0b'
                              : '#e2e8f0',
                          }}>
                            {c.id}
                          </span>
                          <span style={{ color: '#64748b', fontSize: '10px' }}>
                            {c.type === 'nifti' ? 'NIfTI' : 'DICOM ZIP'}
                          </span>
                          {c.ptFile ? (
                            <span style={{ color: '#8b5cf6', fontSize: '10px' }}>CT+PT</span>
                          ) : (
                            <span style={{ color: '#3b82f6', fontSize: '10px' }}>CT only</span>
                          )}
                          {c.status === 'processing' && (
                            <span style={{ color: '#f59e0b', fontSize: '10px' }}>
                              segmenting...
                            </span>
                          )}
                          {c.status === 'success' && (
                            <span style={{ color: '#10b981', fontSize: '10px' }}>
                              done
                            </span>
                          )}
                          {c.status === 'failed' && (
                            <span style={{ color: '#ef4444', fontSize: '10px' }} title={c.error}>
                              error
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

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
                      Tumor 분할 실행
                    </span>
                  </label>
                </div>

                {/* Patient Name (base) */}
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', color: '#a0aec0' }}>
                    Patient Name (base):
                  </label>
                  <input
                    type="text"
                    value={patientName}
                    onChange={(e) => setPatientName(e.target.value)}
                    style={inputStyle}
                    disabled={isProcessing}
                    placeholder="BatchStudy"
                  />
                  <div style={{ fontSize: '10px', color: '#64748b' }}>
                    Each case will be named: {sanitizePatientName(patientName || 'BatchStudy')}_[case_id]
                  </div>
                </div>

                {/* Upload Batch Button */}
                <button
                  onClick={runBatch}
                  disabled={isProcessing || detectedCases.length === 0}
                  style={isProcessing || detectedCases.length === 0 ? buttonDisabledStyle : { ...buttonStyle, backgroundColor: '#10b981' }}
                >
                  {isProcessing ? 'Processing Batch...' : `Upload & Run Batch (${detectedCases.length} cases)`}
                </button>

                {/* Batch Progress */}
                {batchProgress && (
                  <div style={{
                    marginTop: '12px',
                    padding: '12px',
                    backgroundColor: '#0f172a',
                    borderRadius: '6px',
                  }}>
                    <div style={{ fontSize: '12px', marginBottom: '8px', color: '#e2e8f0' }}>
                      Processing: {batchProgress.completed + batchProgress.failed} / {batchProgress.total}
                      {batchProgress.failed > 0 && (
                        <span style={{ color: '#ef4444', marginLeft: '8px' }}>
                          ({batchProgress.failed} failed)
                        </span>
                      )}
                    </div>

                    {/* Progress Bar */}
                    <div style={{
                      width: '100%',
                      height: '8px',
                      backgroundColor: '#1e293b',
                      borderRadius: '4px',
                      overflow: 'hidden',
                      marginBottom: '8px',
                    }}>
                      <div style={{
                        height: '100%',
                        width: `${((batchProgress.completed + batchProgress.failed) / batchProgress.total) * 100}%`,
                        backgroundColor: batchProgress.failed > 0 ? '#f59e0b' : '#10b981',
                        borderRadius: '4px',
                        transition: 'width 0.3s ease',
                      }} />
                    </div>

                    {batchProgress.current && (
                      <div style={{ fontSize: '11px', color: '#94a3b8' }}>
                        Current: {batchProgress.current}
                      </div>
                    )}

                    {!batchProgress.current && batchProgress.completed + batchProgress.failed === batchProgress.total && (
                      <div style={{
                        fontSize: '12px',
                        color: batchProgress.failed > 0 ? '#f59e0b' : '#10b981',
                        fontWeight: 'bold',
                      }}>
                        Batch complete — {batchProgress.completed} succeeded, {batchProgress.failed} failed
                      </div>
                    )}
                  </div>
                )}

                {/* Info Box */}
                <div style={{
                  marginTop: '16px',
                  padding: '12px',
                  backgroundColor: '#0f172a',
                  borderRadius: '6px',
                  fontSize: '11px',
                  color: '#64748b',
                }}>
                  <div style={{ marginBottom: '4px', color: '#94a3b8' }}>Supported Folder Formats:</div>
                  <ul style={{ margin: '0 0 8px 0', paddingLeft: '16px' }}>
                    <li>Top-level NIfTI files (.nii, .nii.gz)</li>
                    <li>Top-level DICOM ZIP files (.zip)</li>
                    <li>Subfolders with CT NIfTI (ct.nii.gz, image.nii.gz, etc.)</li>
                  </ul>
                  <div style={{ marginBottom: '4px', color: '#94a3b8' }}>Note:</div>
                  <ul style={{ margin: '0', paddingLeft: '16px' }}>
                    <li>Raw DICOM folders are not supported — please ZIP them first</li>
                    <li>Cases are uploaded sequentially to avoid overloading the server</li>
                  </ul>
                </div>
              </div>
            )}
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
