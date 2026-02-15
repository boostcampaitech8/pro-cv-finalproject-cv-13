import React from 'react';
import UploadPanel from './Panels/UploadPanel';
import StudyListPanel from './Panels/StudyListPanel';
import AnalysisPanel from './Panels/AnalysisPanel';

/**
 * Panel Module for Nerve Assessment Extension
 * Following OHIF v3.8.0 standard pattern
 */
function getPanelModule({
  commandsManager,
  extensionManager,
  servicesManager,
}: {
  commandsManager: any;
  extensionManager: any;
  servicesManager: any;
}) {
  const wrappedUploadPanel = () => {
    return (
      <UploadPanel
        commandsManager={commandsManager}
        extensionManager={extensionManager}
        servicesManager={servicesManager}
      />
    );
  };

  const wrappedStudyListPanel = () => {
    return (
      <StudyListPanel
        commandsManager={commandsManager}
        extensionManager={extensionManager}
        servicesManager={servicesManager}
      />
    );
  };

  const wrappedAnalysisPanel = () => {
    return (
      <AnalysisPanel
        commandsManager={commandsManager}
        extensionManager={extensionManager}
        servicesManager={servicesManager}
      />
    );
  };

  return [
    {
      name: 'nerveUploadPanel',
      iconName: 'tab-patient-info',
      iconLabel: 'Upload',
      label: 'CT Upload',
      component: wrappedUploadPanel,
    },
    {
      name: 'nerveStudyListPanel',
      iconName: 'tab-studies',
      iconLabel: 'Studies',
      label: 'Study List',
      component: wrappedStudyListPanel,
    },
    {
      name: 'analysisPanel',
      iconName: 'tab-segmentation',
      iconLabel: 'Analysis',
      label: 'Nerve Analysis',
      component: wrappedAnalysisPanel,
    },
  ];
}

export default getPanelModule;
