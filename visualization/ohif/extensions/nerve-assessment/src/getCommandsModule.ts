/**
 * Commands Module for Nerve Assessment Extension
 */

// Backend URL for Orthanc API calls
const BACKEND_URL = '/api';

function getCommandsModule({
  servicesManager,
  commandsManager,
}: {
  servicesManager: any;
  commandsManager: any;
}) {
  const { displaySetService, uiNotificationService } = servicesManager.services;

  return {
    definitions: {
      openNerveUploadPanel: {
        commandFn: () => {
          console.log('[NerveAssessment] Open upload panel command');
        },
      },
      refreshStudyList: {
        commandFn: () => {
          console.log('[NerveAssessment] Refresh study list command');
        },
      },
      deleteSeriesFromOrthanc: {
        // Called from menuContentCustomization via commandsManager.runAsync
        // Receives displaySetInstanceUID in the options
        commandFn: async ({ displaySetInstanceUID }) => {
          console.log('[NerveAssessment] Delete series from Orthanc command');
          console.log('[NerveAssessment] displaySetInstanceUID:', displaySetInstanceUID);

          if (!displaySetInstanceUID) {
            console.error('[NerveAssessment] No displaySetInstanceUID provided');
            alert('Error: No series selected');
            return;
          }

          // Get displaySet from service
          const displaySet = displaySetService.getDisplaySetByUID(displaySetInstanceUID);
          if (!displaySet) {
            console.error('[NerveAssessment] DisplaySet not found:', displaySetInstanceUID);
            alert('Error: Could not find series');
            return;
          }

          const { SeriesInstanceUID, Modality } = displaySet;
          console.log(`[NerveAssessment] Deleting ${Modality} series: ${SeriesInstanceUID}`);

          // Confirm deletion
          const confirmMessage = `Delete this ${Modality} series from Orthanc?\n\nThis action cannot be undone.`;
          if (!confirm(confirmMessage)) {
            console.log('[NerveAssessment] Deletion cancelled by user');
            return;
          }

          try {
            // Call backend API to delete series from Orthanc
            const response = await fetch(`${BACKEND_URL}/delete-series/${SeriesInstanceUID}`, {
              method: 'DELETE',
            });

            if (!response.ok) {
              const errorText = await response.text();
              throw new Error(errorText);
            }

            const result = await response.json();
            console.log('[NerveAssessment] Delete result:', result);

            // Show notification
            if (uiNotificationService?.show) {
              uiNotificationService.show({
                title: 'Series Deleted',
                message: `${Modality} series deleted from Orthanc`,
                type: 'success',
              });
            }

            // Refresh the page to update study browser
            window.location.reload();
          } catch (error: any) {
            console.error('[NerveAssessment] Error deleting series:', error);

            if (uiNotificationService?.show) {
              uiNotificationService.show({
                title: 'Delete Failed',
                message: error.message || 'Failed to delete series',
                type: 'error',
              });
            } else {
              alert(`Error deleting series: ${error.message || error}`);
            }
          }
        },
      },
    },
    // No defaultContext - commands will be available in all contexts
  };
}

export default getCommandsModule;
