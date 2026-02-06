import { id } from './id';
import getPanelModule from './getPanelModule';
import getCommandsModule from './getCommandsModule';
import getLayoutTemplateModule from './getLayoutTemplateModule';
import getCustomizationModule from './getCustomizationModule';
import { loadLabelmapForStudy, cleanupLabelmap } from './utils/labelmapIntegration';
import { initVolume3DRendering, cleanupVolumeInit, subscribeToLayoutChanges, unsubscribeFromLayoutChanges } from './utils/surfaceLoader';
import nerveAssessmentHangingProtocol from './hangingProtocol';

/**
 * Nerve Assessment Extension for OHIF Viewer v3.11.0
 *
 * Labelmap-only mode:
 * - Uses NIfTI multi-label labelmap with polySeg for segmentation rendering
 * - Supports all views: axial, sagittal, coronal
 * - On-demand CONTOUR conversion to avoid memory issues
 */

const extension = {
  id,

  /**
   * Hanging Protocol Module - MPR + 3D (2x2 layout)
   */
  getHangingProtocolModule({ servicesManager, extensionManager }: any = {}) {
    console.log('[NerveAssessment] getHangingProtocolModule - MPR + 3D layout');
    return [
      {
        name: nerveAssessmentHangingProtocol.id,
        protocol: nerveAssessmentHangingProtocol,
      },
    ];
  },

  /**
   * Pre-registration hook
   */
  async preRegistration({
    servicesManager,
    commandsManager,
    configuration = {},
  }: {
    servicesManager: any;
    commandsManager: any;
    configuration?: any;
  }) {
    console.log('[NerveAssessment] Extension pre-registration v3.11.0');
  },

  /**
   * Called when the extension is activated in a mode
   * Sets up CT viewports and loads NIfTI labelmap for polySeg rendering
   */
  onModeEnter({ servicesManager, extensionManager }: { servicesManager: any; extensionManager?: any }) {
    if (!window.location.pathname.includes('nerve-assessment')) {
      console.log('[NerveAssessment] Not in nerve-assessment mode, skipping');
      return;
    }

    console.log('[NerveAssessment] onModeEnter - Labelmap-only mode');

    // Hanging protocol handles CT display set assignment with correct orientations
    // (axial, sagittal, coronal, volume3d) - no manual override needed

    setTimeout(async () => {
      try {
        const result = await loadLabelmapForStudy(servicesManager);
        if (result.success) {
          console.log('[NerveAssessment] Labelmap loaded');
        }
      } catch (error: any) {
        console.warn('[NerveAssessment] Labelmap loading failed:', error?.message || error);
      }

      initVolume3DRendering(servicesManager);

      const params = new URLSearchParams(window.location.search);
      const studyUID = params.get('StudyInstanceUIDs') || '';
      if (studyUID) {
        subscribeToLayoutChanges(servicesManager, studyUID);
      }
    }, 500);
  },

  /**
   * Called when leaving the mode
   */
  onModeExit() {
    cleanupLabelmap();
    cleanupVolumeInit();
    unsubscribeFromLayoutChanges();
  },

  getPanelModule,
  getCommandsModule,
  getLayoutTemplateModule,
  getCustomizationModule,
};

export default extension;
