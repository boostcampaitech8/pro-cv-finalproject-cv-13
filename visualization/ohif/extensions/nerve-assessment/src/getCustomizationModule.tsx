import React from 'react';
import UploadPanel from './Panels/UploadPanel';

/**
 * Custom About modal - replaces default OHIF About with project info
 */
function NerveAssessmentAbout() {
  return (
    <div style={{
      width: '400px',
      padding: '24px',
      color: 'white',
      textAlign: 'center',
    }}>
      <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#10b981', marginBottom: '8px' }}>
        Team NextCT
      </div>
      <div style={{ fontSize: '14px', color: '#94a3b8', marginBottom: '24px' }}>
        Nerve Assessment Viewer
      </div>

      <div style={{
        textAlign: 'left',
        background: '#1e293b',
        borderRadius: '8px',
        padding: '16px',
        fontSize: '13px',
        lineHeight: '1.8',
        color: '#cbd5e1',
      }}>
        <div><strong style={{ color: '#e2e8f0' }}>Project:</strong> Surgical Nerve Risk Assessment</div>
        <div><strong style={{ color: '#e2e8f0' }}>Features:</strong> CT segmentation, nerve estimation, 3D rendering</div>
        <div><strong style={{ color: '#e2e8f0' }}>Based on:</strong> OHIF Viewer v3.11.0</div>
      </div>
    </div>
  );
}

// MenuComponentCustomization properties for OHIF header menu
(NerveAssessmentAbout as any).menuTitle = 'About';
(NerveAssessmentAbout as any).title = 'About';
(NerveAssessmentAbout as any).containerClassName = 'max-w-md';

/**
 * Customization Module for Nerve Assessment Extension
 */
function getCustomizationModule({
  servicesManager,
  extensionManager,
  commandsManager,
}: {
  servicesManager: any;
  extensionManager: any;
  commandsManager: any;
}) {
  // Wrapper component for modal display
  const UploadModalContent = ({ dataSource, onComplete, onStarted }) => {
    return (
      <div style={{ width: '400px', maxHeight: '80vh', overflow: 'auto' }}>
        <UploadPanel
          commandsManager={commandsManager}
          extensionManager={extensionManager}
          servicesManager={servicesManager}
        />
      </div>
    );
  };

  return [
    {
      name: 'default',
      value: {
        // Replace default DICOM upload component with our 3-tab panel
        dicomUploadComponent: UploadModalContent,
        // Replace default OHIF About modal with project info
        'ohif.aboutModal': NerveAssessmentAbout,
      },
    },
  ];
}

export default getCustomizationModule;
