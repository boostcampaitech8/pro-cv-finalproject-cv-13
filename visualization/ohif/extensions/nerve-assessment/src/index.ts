import { id } from './id';
import getPanelModule from './getPanelModule';
import getCommandsModule from './getCommandsModule';
import getLayoutTemplateModule from './getLayoutTemplateModule';
import getCustomizationModule from './getCustomizationModule';
import { loadLabelmapForStudy, cleanupLabelmap } from './utils/labelmapIntegration';
import { setVolume3DPreset, setVolumeHUThreshold, subscribeToLayoutChanges, unsubscribeFromLayoutChanges } from './utils/surfaceLoader';
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

    // Load NIfTI labelmap for polySeg rendering (all views: axial/sagittal/coronal)
    setTimeout(async () => {
      try {
        console.log('[NerveAssessment] Loading NIfTI labelmap...');
        const result = await loadLabelmapForStudy(servicesManager);

        if (result.success) {
          console.log('[NerveAssessment] ✓ Labelmap loaded - polySeg ready for all views');
        } else {
          console.log('[NerveAssessment] Labelmap not available:', result.error);
        }
      } catch (error: any) {
        console.warn('[NerveAssessment] Labelmap loading failed:', error?.message || error);
      }

      // Set CT volume rendering preset on volume3d viewport
      // Retry with increasing delay since volume3d may not have CT loaded yet
      const applyPresetAndThreshold = (attempt: number) => {
        try {
          const presetSet = setVolume3DPreset(servicesManager, 'CT-Bone');
          if (presetSet) {
            console.log(`[NerveAssessment] ✓ Volume3D CT-Bone preset applied (attempt ${attempt})`);
            const huSet = setVolumeHUThreshold(servicesManager, 300, 3000, 0.15, 100);
            if (huSet) {
              console.log('[NerveAssessment] ✓ Default HU threshold applied (min=300)');
            } else if (attempt < 5) {
              console.log('[NerveAssessment] HU threshold not applied (no volume actor), retrying...');
              setTimeout(() => applyPresetAndThreshold(attempt + 1), 2000);
            }
          } else if (attempt < 5) {
            console.log(`[NerveAssessment] Volume3D not ready, retry ${attempt}/5...`);
            setTimeout(() => applyPresetAndThreshold(attempt + 1), 2000);
          } else {
            console.log('[NerveAssessment] Volume3D preset failed after 5 attempts');
          }
        } catch (e: any) {
          console.warn('[NerveAssessment] Volume3D preset error:', e?.message);
          if (attempt < 5) {
            setTimeout(() => applyPresetAndThreshold(attempt + 1), 2000);
          }
        }
      };
      applyPresetAndThreshold(1);

      // Subscribe to layout changes to reapply meshes when user switches layouts
      const params = new URLSearchParams(window.location.search);
      const studyUID = params.get('StudyInstanceUIDs') || '';
      if (studyUID) {
        subscribeToLayoutChanges(servicesManager, studyUID);
      }
    }, 2000);
  },

  /**
   * Called when leaving the mode
   */
  onModeExit() {
    console.log('[NerveAssessment] onModeExit');
    cleanupLabelmap();
    unsubscribeFromLayoutChanges();
  },

  getPanelModule,
  getCommandsModule,
  getLayoutTemplateModule,
  getCustomizationModule,
};

export default extension;
